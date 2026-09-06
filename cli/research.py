"""Research-only command line workbench for validated strategy experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from backtesting.engine import BacktestEngine
from backtesting.models import BacktestConfig
from data.schema import validate_ohlcv
from evaluation.platform import SQLiteResearchRepository, report_markdown
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


if __name__ == "__main__":
    app()
