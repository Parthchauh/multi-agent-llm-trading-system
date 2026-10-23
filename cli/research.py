"""Research-only command line workbench for validated strategy experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig
from cli.reproducibility import register as _register_reproducibility
from data.schema import validate_ohlcv
from evaluation.experiments import (
    ExperimentArchitecture,
    ExperimentConfig,
    ModelRequestConfig,
    TransactionCostConfig,
)
from evaluation.platform import ExperimentRunner, SQLiteResearchRepository, report_markdown
from research_dashboard.server import serve_dashboard
from strategies.compiler import StrategyCompiler
from strategies.schema import StrategySchema
from strategies.validator import StrategyValidator

app = typer.Typer(
    name="research",
    help="Auditable research and backtesting commands. No live-trading actions are provided.",
    no_args_is_help=True,
)


def _load_strategy(path: Path) -> StrategySchema:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter("Strategy must be readable JSON.") from exc
    validation = StrategyValidator().validate_dict(raw)
    if not validation.is_valid:
        raise typer.BadParameter("Strategy validation failed: " + "; ".join(validation.errors))
    return StrategySchema.model_validate(raw)


def _load_prices(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        return validate_ohlcv(frame)
    except Exception as exc:
        raise typer.BadParameter("CSV must contain canonical OHLCV data with a datetime index.") from exc


@app.command("validate-strategy")
def validate_strategy(strategy: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Validate a declarative strategy JSON through the canonical Sprint 1 path."""
    parsed = _load_strategy(strategy)
    typer.echo(json.dumps(parsed.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("backtest")
def backtest(
    strategy: Path = typer.Argument(..., exists=True, readable=True),
    prices: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(Path("backtest-result.json"), "--output", "-o"),
    initial_capital: float = typer.Option(100_000.0, min=1.0),
    commission_pct: float = typer.Option(0.001, min=0.0),
    slippage_pct: float = typer.Option(0.0005, min=0.0),
) -> None:
    """Run the deterministic compiler/backtester and write an auditable JSON result."""
    parsed = _load_strategy(strategy)
    frame = _load_prices(prices)
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        position_size_pct=parsed.position_sizing.value,
        maximum_holding_days=parsed.exit.maximum_holding_days,
    )
    result = BacktestEngine(config).run(frame, StrategyCompiler().compile(parsed, frame))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(f"Backtest complete: {output.resolve()}")


@app.command("experiment")
def experiment(
    symbol: str = typer.Option(..., "--symbol", help="One ticker symbol for this run."),
    architecture: ExperimentArchitecture = typer.Option(
        ExperimentArchitecture.MULTI_AGENT_FULL,
        "--architecture",
        help="Controlled architecture to execute.",
    ),
    database: Path = typer.Option(Path("research.sqlite"), "--database", "-d"),
    output_dir: Path = typer.Option(Path("results/e2e"), "--output-dir", "-o"),
    start_date: str = typer.Option("2020-01-01", "--start-date"),
    train_end: str = typer.Option("2023-12-31", "--train-end"),
    validation_end: str = typer.Option("2024-12-31", "--validation-end"),
    test_end: str = typer.Option("2025-12-31", "--test-end"),
    provider: str = typer.Option("google", "--provider"),
    model: str = typer.Option("gemini-2.5-flash", "--model"),
    seed: int = typer.Option(42, "--seed"),
    development_only: bool = typer.Option(False, '--development-only', help='Keep test and final holdout sealed during architecture studies.'),
    repetitions: int = typer.Option(1, "--repetitions", min=1, max=100),
    debate_rounds: int | None = typer.Option(None, "--debate-rounds", min=0, max=3),
    refinement_rounds: int | None = typer.Option(None, "--refinement-rounds", min=0, max=5),
    initial_capital: float = typer.Option(100_000.0, "--initial-capital", min=1.0),
    commission_pct: float = typer.Option(0.001, "--commission-pct", min=0.0),
    slippage_pct: float = typer.Option(0.0005, "--slippage-pct", min=0.0),
    position_size_pct: float = typer.Option(10.0, "--position-size-pct", min=0.000_001, max=100.0),
) -> None:
    """Run and persist one controlled, cost-aware research experiment.

    This command may fetch historical data and call the configured LLM provider.
    It never submits orders or invokes the legacy trading graph.
    """

    resolved_debate_rounds = (
        debate_rounds
        if debate_rounds is not None
        else (0 if architecture in {ExperimentArchitecture.SINGLE_AGENT, ExperimentArchitecture.MULTI_AGENT_NO_DEBATE} else 1)
    )
    resolved_refinement_rounds = (
        refinement_rounds
        if refinement_rounds is not None
        else (1 if architecture is ExperimentArchitecture.MULTI_AGENT_FULL else 0)
    )
    config = ExperimentConfig(
        experiment_id=(
            f"{architecture.value}-{symbol.upper()}-{start_date}-{test_end}-seed{seed}"
        ),
        architecture=architecture,
        provider=provider,
        model=model,
        dataset_id=f"yahoo:{symbol.upper()}:{start_date}:{test_end}",
        start=start_date,
        end=test_end,
        symbols=(symbol,),
        seed=seed,
        development_only=development_only,
        model_settings=ModelRequestConfig(temperature=0.2, timeout_seconds=30),
        transaction_costs=TransactionCostConfig(
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            position_size_pct=position_size_pct,
        ),
        repetitions=repetitions,
        debate_rounds=resolved_debate_rounds,
        refinement_rounds=resolved_refinement_rounds,
    )

    # Kept lazy: validation/backtest/report commands do not need provider or
    # LangGraph imports unless the user explicitly starts an experiment.
    from scripts.run_research_e2e import run_e2e_research

    selected_mode = architecture.value.upper()

    def executor(_config: ExperimentConfig, run_symbol: str, repetition: int) -> dict[str, object]:
        artifact_dir = output_dir / config.experiment_id / run_symbol / f"repetition-{repetition}"
        result = run_e2e_research(
            symbol=run_symbol,
            start_date=start_date,
            train_end=train_end,
            validation_end=validation_end,
            test_end=test_end,
            initial_capital=initial_capital,
            commission=commission_pct,
            slippage=slippage_pct,
            position_size_pct=position_size_pct,
            model_provider=provider,
            model_name=model,
            max_debate_rounds=resolved_debate_rounds,
            max_refinement_rounds=resolved_refinement_rounds,
            seed=seed + repetition - 1,
            mode=selected_mode,
            development_only=development_only,
            output_dir=artifact_dir,
            campaign_id=config.experiment_id,
            exposure_ledger_path=database.resolve().parent / 'holdout_exposure.sqlite',
        )
        return {
            "status": result["status"],
            "strategy_id": result["selected_strategy_id"],
            "validation_metrics": result["validation_metrics"],
            'train_metrics': result.get('train_metrics', {}),
            'test_metrics': result['test_metrics'],
            'final_holdout_metrics': result.get('final_holdout_metrics', {}),
            'certification': result.get('certification', {}),
            'selected_strategy': result.get('selected_strategy'),
            'risk': result.get('risk'),
            'provenance': result.get('provenance'),
            'lineage': result.get('lineage'),
            'execution_config': result.get('execution_config'),
            'invocation_audit': result.get('invocation_audit'),
            'development_only': result.get('development_only', development_only),
            "artifact_directory": result["output_dir"],
        }

    report = ExperimentRunner(SQLiteResearchRepository(database)).run(config, executor)
    completed = sum(run.status == "COMPLETED" for run in report.runs)
    typer.echo(
        f"Experiment persisted: {config.experiment_id} ({completed}/{len(report.runs)} completed)"
    )


@app.command("list-runs")
def list_runs(
    database: Path = typer.Option(Path("research.sqlite"), "--database", "-d"),
    experiment_id: str | None = typer.Option(None, "--experiment-id"),
) -> None:
    """List persisted research runs, including failures."""
    for run in SQLiteResearchRepository(database).list_runs(experiment_id):
        typer.echo(f"{run.run_id}\t{run.symbol}\t{run.repetition}\t{run.status}")


@app.command("show-run")
def show_run(
    run_id: str = typer.Argument(...),
    database: Path = typer.Option(Path("research.sqlite"), "--database", "-d"),
) -> None:
    """Show one persisted run as canonical JSON."""
    for run in SQLiteResearchRepository(database).list_runs():
        if run.run_id == run_id:
            typer.echo(run.model_dump_json(indent=2))
            return
    raise typer.BadParameter("No persisted run has that run_id.")


@app.command("export-report")
def export_report(
    experiment_id: str = typer.Argument(...),
    output: Path = typer.Option(Path("experiment-report.md"), "--output", "-o"),
    database: Path = typer.Option(Path("research.sqlite"), "--database", "-d"),
) -> None:
    """Render a human-readable Markdown report from the canonical persisted JSON."""
    report = SQLiteResearchRepository(database).load_report(experiment_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report_markdown(report), encoding="utf-8")
    typer.echo(f"Report exported: {output.resolve()}")


@app.command("serve-dashboard")
def serve_dashboard_command(
    database: Path = typer.Option(Path("research.sqlite"), "--database", "-d"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port", min=0, max=65_535),
) -> None:
    """Serve a local, read-only dashboard for persisted research reports."""
    serve_dashboard(database, host=host, port=port, announce=typer.echo)


_register_reproducibility(app)

if __name__ == "__main__":
    app()
