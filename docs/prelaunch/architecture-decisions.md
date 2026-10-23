# Prelaunch architecture decisions

Retain the existing package layout and supported `trading-research` CLI. Preserve
the upstream graph as explicitly legacy reference code. The Docker entry point
now selects the research CLI. No broker integration is included.

Reuse TradeExecutor and Portfolio for transaction costs and whole-share sizing.
Extract the existing bar execution order into backtesting/kernel.py. Both the
historical engine and paper adapter call it. Paper records every transaction,
including an entry followed by a protective exit on the same bar. The existing
single-fill return is retained for compatibility; callers needing all fills use
bar_fills or portfolio.fills. Numeric state is not rounded per fill.

RiskPolicy remains the low-level schema and retains compatibility for exploratory
unit-level analysis. Launch entry points require complete named profiles. The
new default thresholds are proposed research screening settings, not empirically
validated guarantees. Twenty trades, nonnegative return, Sharpe >= 0.5, drawdown
<= 20%, and turnover <= 2000% are conservative exclusion criteria for this MVP;
passing them alone does not prove statistical significance or release readiness.
Turnover is cumulative over the evaluated split, not annualized.

Benchmark behavior changes intentionally: buy_and_hold now means 100% allocation;
the previous allocation-matched interpretation has the explicit separate key
exposure_matched_buy_and_hold. Both charge the same configured transaction costs,
retain residual cash, use whole shares, and do not rebalance. Technical baselines
retain their stop/target overlays. Artifact consumers must account for the new
benchmark and separate final_holdout_metrics.json.

Later statistical certification, normalization, monitoring, and production CI
work is deferred to separate packages after the P0 release prerequisites.
Only the provenance and invocation-audit helpers needed for current evidence
are retained from the initial later-phase exploration.

The root main.py and both supported command aliases invoke the research CLI.
The original root demo is preserved in examples/legacy_tradingagents_demo.py,
and the inherited CLI is explicitly tradingagents-legacy.
