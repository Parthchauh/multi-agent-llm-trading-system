"""Offline reproduction commands; none authorize certification or execution."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from backtesting.models import BacktestConfig
from data.snapshots import DatasetIdentity, create_snapshot, record_replay, verify_replay
from research_validation.campaign import CampaignConfig, FoldResearchCampaign
from risk.profiles import risk_profile
from strategies.schema import StrategySchema


def register(app: typer.Typer) -> None:
    @app.command('monitor-simulation')
    def monitor_simulation(session_id: str, reference_bundle: Path,
                           database: Path = Path('results/simulations.sqlite')) -> None:
        """Compare saved simulation paths to frozen replay evidence under a persistent drift policy."""
        from paper_trading.operations import PaperSimulationStore
        report = PaperSimulationStore(database).monitor_drift(session_id,
            json.loads(reference_bundle.read_text(encoding='utf-8')))
        typer.echo(json.dumps(report, indent=2, allow_nan=False))

    @app.command('normalize-prices')
    def normalize_prices(prices: Path, factors: Path, as_of: str, basis: str,
                         source: str, method_version: str, output: Path) -> None:
        """Apply explicitly available vendor price/volume factors, without inferring events."""
        from data.adjustments import normalize_with_factors
        normalized, trace = normalize_with_factors(
            pd.read_csv(prices, index_col=0, parse_dates=True),
            pd.read_csv(factors, index_col=0, parse_dates=True), as_of=pd.Timestamp(as_of),
            basis=basis, source=source, method_version=method_version)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as handle:
            normalized.to_csv(handle)
        with output.with_suffix(output.suffix+'.adjustments.json').open('x', encoding='utf-8') as handle:
            json.dump(trace, handle, indent=2, allow_nan=False)
        typer.echo(json.dumps({'status': 'NORMALIZED', 'adjustment_sha256': trace['sha256']}))

    @app.command('compare-architectures')
    def compare_architectures(experiment_ids: list[str] = typer.Argument(...),
                             database: Path = Path('research.sqlite')) -> None:
        """Compare already-recorded matched treatments on validation evidence only."""
        from evaluation.ablation import compare_architectures as compare
        from evaluation.platform import SQLiteResearchRepository
        repository = SQLiteResearchRepository(database)
        typer.echo(json.dumps(compare([repository.load_report(identity) for identity in experiment_ids]),
                              indent=2, allow_nan=False))

    @app.command('backup-state')
    def backup_state(sources: Path, destination: Path) -> None:
        """Back up explicit named SQLite/artifact files; stop writers for a consistent set."""
        from research_service.backups import create_backup
        inputs = {name: Path(path) for name, path in json.loads(sources.read_text(encoding='utf-8')).items()}
        typer.echo(json.dumps(create_backup(destination, inputs), sort_keys=True))

    @app.command('restore-state')
    def restore_state(backup: Path, destination: Path) -> None:
        """Verify and restore into a new directory, revoking saved service sessions."""
        from research_service.backups import restore_backup
        typer.echo(json.dumps(restore_backup(backup, destination), sort_keys=True))

    @app.command("freeze-candidate")
    def freeze_candidate(manifest: Path, registry: Path = Path("results/registry.sqlite")) -> None:
        """Freeze strategy/config/risk/evidence identities. Paper promotion remains disabled."""
        from evaluation.registry import FrozenCandidateRegistry
        from risk.models import RiskPolicy
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        candidate_id = FrozenCandidateRegistry(registry).freeze(
            strategy=StrategySchema.model_validate(payload['strategy']),
            config=BacktestConfig.model_validate(payload['config']),
            risk_policy=RiskPolicy.model_validate(payload['risk_policy']), evidence=payload['evidence'])
        typer.echo(json.dumps({'candidate_id': candidate_id, 'paper_eligible': False}))

    @app.command("simulate-paper")
    def simulate_paper(candidate_id: str, symbol: str, prices: Path,
                       registry: Path = Path("results/registry.sqlite"),
                       database: Path = Path("results/simulations.sqlite")) -> None:
        """Replay saved bars with persistent operational controls, without market connections."""
        from evaluation.registry import FrozenCandidateRegistry
        from paper_trading.operations import PaperSimulationStore
        frame = pd.read_csv(prices, index_col=0, parse_dates=True)
        if frame.index.tz is None:
            raise typer.BadParameter('Simulation CSV timestamps must include a timezone.')
        store = PaperSimulationStore(database)
        session = store.create(registry=FrozenCandidateRegistry(registry), candidate_id=candidate_id,
                               symbol=symbol, sessions=frame.index)
        for end in range(1, len(frame)+1):
            response = store.process(session, frame.iloc[:end], now=frame.index[end-1].to_pydatetime())
            if response['status'] != 'SIMULATED':
                raise typer.BadParameter('Simulation rejected a saved bar; inspect its persisted incidents.')
        typer.echo(json.dumps({'session_id': session, **store.status(session)}, allow_nan=False))

    @app.command("kill-simulation")
    def kill_simulation(session_id: str, database: Path = Path("results/simulations.sqlite")) -> None:
        """Persistently block new simulated entries; valid next-bar exits remain enabled."""
        from paper_trading.operations import PaperSimulationStore
        typer.echo(json.dumps(PaperSimulationStore(database).kill(session_id)))

    @app.command("provision-service-user")
    def provision_user(tenant: str, username: str, role: str,
                       access_database: Path = Path("results/service-access.sqlite")) -> None:
        """Local administrative provisioning. Password is read privately, never as a CLI argument."""
        import getpass

        from research_service.access import AccessStore
        password = getpass.getpass('New service password (at least 16 characters): ')
        if password != getpass.getpass('Repeat password: '):
            raise typer.BadParameter('Passwords did not match.')
        AccessStore(access_database).create_user(tenant, username, password, role)
        typer.echo('Service user provisioned.')

    @app.command("claim-evidence")
    def claim_evidence(experiment_id: str, tenant: str,
                       access_database: Path = Path("results/service-access.sqlite"),
                       research_database: Path = Path("research.sqlite")) -> None:
        """Local administrator assigns existing evidence to one tenant, once."""
        from evaluation.platform import SQLiteResearchRepository
        from research_service.access import AccessStore
        SQLiteResearchRepository(research_database).load_report(experiment_id)
        AccessStore(access_database).claim_resource(experiment_id, tenant)
        typer.echo('Evidence ownership recorded.')

    @app.command("serve-evidence-api")
    def serve_api(certificate: Path, private_key: Path,
                  access_database: Path = Path("results/service-access.sqlite"),
                  research_database: Path = Path("research.sqlite"),
                  port: int = typer.Option(8443, min=1, max=65535)) -> None:
        """Serve the separate TLS-only authenticated, tenant-scoped read API on loopback."""
        from research_service.server import serve
        serve(access_database=access_database, research_database=research_database,
              certificate=certificate, private_key=private_key, port=port)

    @app.command("snapshot-dataset")
    def snapshot_dataset(prices: Path, metadata: Path,
                         store: Path = Path("results/datasets"),
                         sessions: Path | None = None) -> None:
        """Save canonical CSV and explicit provider identity as a hashed offline dataset."""
        try:
            identity = DatasetIdentity.model_validate_json(metadata.read_text(encoding="utf-8"))
            frame = pd.read_csv(prices, index_col=0, parse_dates=True)
            expected = (pd.DatetimeIndex(pd.to_datetime(json.loads(sessions.read_text())))
                        if sessions else None)
            digest = create_snapshot(store, frame, identity, expected_sessions=expected)
        except (OSError, ValueError, KeyError) as exc:
            raise typer.BadParameter(f"Snapshot rejected ({type(exc).__name__}).") from None
        typer.echo(json.dumps({"status": "SNAPSHOT_SAVED", "dataset_sha256": digest}))

    @app.command("record-replay")
    def replay_record(dataset_sha256: str, strategy: Path, config: Path,
                      output: Path, store: Path = Path("results/datasets")) -> None:
        """Record a low-level deterministic backtest; this is not canonical certification."""
        try:
            bundle = record_replay(snapshot_root=store, dataset_sha256=dataset_sha256,
                strategy=StrategySchema.model_validate_json(strategy.read_text(encoding="utf-8")),
                config=BacktestConfig.model_validate_json(config.read_text(encoding="utf-8")),
                source_root=Path.cwd(), output=output)
        except (OSError, ValueError, KeyError) as exc:
            raise typer.BadParameter(f"Replay recording rejected ({type(exc).__name__}).") from None
        typer.echo(json.dumps({"status": "REPLAY_RECORDED", "bundle_sha256": bundle["bundle_sha256"],
                               "statistical_certification": False, "paper_eligible": False}))

    @app.command("verify-replay")
    def replay_verify(bundle: Path, store: Path = Path("results/datasets")) -> None:
        """Check input/bundle integrity and require identical deterministic output."""
        try:
            report = verify_replay(bundle, store, Path.cwd())
        except (OSError, ValueError, KeyError) as exc:
            raise typer.BadParameter(f"Replay verification failed ({type(exc).__name__}).") from None
        typer.echo(json.dumps(report, sort_keys=True))

    @app.command("fold-research")
    def fold_research(prices: Path, configuration: Path, responses: Path, output: Path) -> None:
        """Replay per-fold recorded candidate responses on DEVELOPMENT data only.

        The JSON array is folds -> attempts -> strategy schema. Invalid responses
        remain in the search record. This mode makes no live model calls.
        """
        try:
            payload = json.loads(configuration.read_text(encoding="utf-8"))
            candidates = json.loads(responses.read_text(encoding="utf-8"))
            settings = CampaignConfig.model_validate(payload["campaign"])
            backtest = BacktestConfig.model_validate(payload["backtest"])
            def factory(train, fold, attempt):
                return candidates[fold-1][attempt-1]
            report = FoldResearchCampaign().run(
                development_data=pd.read_csv(prices, index_col=0, parse_dates=True),
                config=settings, backtest_config=backtest,
                risk_policy=risk_profile(payload.get("risk_profile", "research_standard")),
                candidate_factory=factory)
            report["candidate_source"] = "RECORDED_RESPONSES"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            raise typer.BadParameter(f"Fold research rejected ({type(exc).__name__}).") from None
        typer.echo(json.dumps({"status": report["status"], "tested_folds": report["tested_folds"],
                               "statistical_certification": False, "paper_eligible": False}))
