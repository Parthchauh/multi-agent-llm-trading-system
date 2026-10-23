# Supported research architecture

The V1 entry point is `python -m cli.research experiment` (installed command:
`trading-research experiment`). It runs scripts/run_research_e2e.py and the
structured research graphs. The legacy `tradingagents` CLI and main.py are
upstream demos, not the supported certification path.

The provider-independent data layer validates chronological OHLCV. Generation
receives training context. The canonical StrategySchema and semantic validator
bound every hypothesis; StrategyCompiler translates validated conditions into
causal signals. BacktestEngine and PaperTradingRunner use UnifiedExecutionKernel,
TradeExecutor, and Portfolio for fills, protective exits, costs, and sizing.

Refinement uses validation evidence. A selected candidate freezes before the
confirmatory test. Final holdout is released only after the confirmatory screen
and risk gate pass. The shared exposure database records overlapping holdout
windows across strategy versions and campaigns. Previously exposed data cannot
be represented as fresh confirmation by choosing a new strategy ID. Preserve
the ledger with the research history; deleting records cannot remove knowledge.

Artifacts distinguish train, validation, confirmatory test, and final holdout,
and include strategy, cost, risk, and data identity. Statistical viability and
paper eligibility remain false until additional robustness and release gates
are explicitly satisfied. Screening success is not proof of profitability.

The existing dashboard serves persisted reports on loopback with read-only
endpoints. Public hosting, authentication, team isolation, real-time paper beta,
and broker execution are separate release programs. The repository does not
activate real-money execution.

See prelaunch/execution-semantics.md for numeric and bar-order contracts and
prelaunch/implementation-status.md for actual verification and unmet gates.
