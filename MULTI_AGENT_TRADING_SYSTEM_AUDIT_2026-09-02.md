# Multi-Agent LLM Trading System — Codebase Audit

**Audit date:** 2026-09-02  
**Repository:** `D:\TradingAgents`  
**Audited revision:** `10c136f49c82e11f0e324c9c50cda1638a8ed5a7` plus the pre-existing dirty working tree described below  
**Method:** source inspection, repository-wide searches, import/runtime probes, and two pytest executions. README and sprint labels were treated as claims, not evidence.

## 1. Executive Summary

The repository contains a credible deterministic core for declarative, long-only daily strategies: strict-extra Pydantic strategy models, explicit indicator and operator allow-lists, causal vectorized indicator calculations, compiled boolean signals, next-bar-open execution, basic portfolio accounting, and a typed market-data contract. It also contains a newer Sprint 4-style LangGraph pipeline with structured regime and strategy agents, validation/compilation gates, bounded generation attempts, typed run statuses, and a tested path to backtest metrics.

That code genuinely runs. The complete pytest invocation collected 229 tests and all 229 passed. A second tests-directory-only run also passed all 229 tests. However, passing tests do not make the present pipeline research-valid.

Three P0 findings currently invalidate historical research results in affected paths:

1. The Sprint 4 graph computes a regime snapshot from the **last bar of the same dataset** that it subsequently uses to backtest the generated strategy from the first bar. Strategy construction is therefore informed by the evaluation sample's future endpoint.
2. ATR stops are set at the next-bar open using ATR calculated with that execution bar's complete High/Low/Close. Those values are not known at the open. A runtime probe held the signal and execution open constant while changing only the execution bar's later range; the stop changed from `104.8061` to `-37.6224`.
3. The legacy dataflow back-fills price data before applying the historical cutoff, and several legacy fundamental/news/insider paths expose current or insufficiently point-in-time data during backdated analysis.

The new deterministic and structured-agent implementation is also almost entirely absent from Git history: before this report, 58 files were untracked, including all `strategies/`, `backtesting/`, `data/`, and `regime/` files and 21 tests. Three tracked files were modified. A clean checkout of the audited HEAD cannot reproduce the implementation or the 229-test baseline.

**Deterministic pipeline classification: PROTOTYPE ONLY.** It is functionally integrated and substantially better controlled than the legacy prose architecture, but the sample-boundary and execution-time leakage defects prevent `RESEARCH-READY` classification.

**Recommended Next Sprint:** a short, blocking **Correctness & Research Protocol Hardening sprint (Sprint 3.5)**. It should repair the two deterministic leakage paths, establish immutable/finite contracts, correct metric semantics, introduce explicit research/evaluation data partitions, and place the implementation under version control before more agent, ranking, or refinement work.

## 2. Repository State

### 2.1 Git and environment state

| Item | Verified state |
|---|---|
| Branch | `main`, tracking `origin/main` |
| HEAD | `10c136f49c82e11f0e324c9c50cda1638a8ed5a7` |
| Tracked modifications before audit | `pyproject.toml`, `tradingagents/agents/__init__.py`, `tradingagents/graph/__init__.py` |
| Untracked files before audit | 58: 21 tests, 10 under `tradingagents/`, 9 strategies, 6 backtesting, 5 regime, 4 data, 3 result reports |
| Tracked tests | Only `tests/test_google_api_key.py`, `tests/test_model_validation.py`, `tests/test_ticker_symbol_handling.py`, plus root `test.py` |
| Python | 3.13.9 |
| Pydantic | 2.12.4 |
| pandas | 2.3.3 |
| pytest | 8.4.2 |
| Dependency health | `python -m pip check`: no broken requirements |
| Repository instructions | No `AGENTS.md` found |

`pyproject.toml` has been modified to include `strategies*`, `backtesting*`, `data*`, and `regime*` in package discovery, but the local `build/lib` tree and `tradingagents.egg-info/top_level.txt` are stale and contain only the legacy `tradingagents` and `cli` packages. No wheel/sdist build was performed during this read-only audit, so final artifact contents are **UNVERIFIED**.

The new code directly imports `pydantic` and `numpy`, but neither is declared as a direct project dependency. They are currently available transitively. There is no declared development dependency group for pytest, coverage, linting, or type checking, and there are no pytest, coverage, Ruff, or mypy settings in `pyproject.toml`.

### 2.2 High-level directory map

| Path | Purpose and important components | Maturity | Architecture lineage |
|---|---|---|---|
| `strategies/` | `StrategySchema`, recursive conditions, registries, causal indicators, semantic validator, `StrategyCompiler`, `CompiledStrategy` | Substantial; critical contract gaps remain | New deterministic core |
| `backtesting/` | `BacktestEngine`, `TradeExecutor`, `Portfolio`, typed result records, basic metrics | Functional; leakage and metric issues block research use | New deterministic core |
| `data/` | Canonical OHLCV validation, provider protocol, Yahoo adapter, in-memory copy-safe cache | Functional first provider layer; metadata/persistence absent | New deterministic core |
| `regime/` | Causal feature frame, frozen regime contracts, deterministic fallback classifier | Functional component; orchestration uses it with an unsafe sample boundary | New Sprint 4 work |
| `tradingagents/agents/strategy_generator.py` | Structured LLM proposal envelope, allow-list/schema prompt context, validation and bounded attempts | Functional with fake clients; real-provider wiring unverified | New Sprint 4 work |
| `tradingagents/agents/regime_analyst.py` | Structured regime interpretation with deterministic fallback | Functional with fake clients | New Sprint 4 work |
| `tradingagents/agents/structured_output.py` | Provider-neutral protocol and LangChain structured-output adapter | Useful abstraction; not wired through the legacy factory/CLI | New Sprint 4 work |
| `tradingagents/graph/research_graph.py` | Typed staged graph: data → regime → generation → validation → compile → backtest | End-to-end functional but sample-leaking | New Sprint 4 work |
| `tradingagents/graph/trading_graph.py`, `setup.py` | Original analyst/debate/trader/risk/portfolio LangGraph | Operational prose workflow, not a deterministic strategy research pipeline | Legacy TradingAgents architecture |
| `tradingagents/agents/analysts/` | Market, news, social, fundamental analysts with read-only data tools | Reusable only as non-authoritative context generators after point-in-time fixes | Legacy |
| `tradingagents/agents/researchers/` | Bull/bear prose debate | Optional qualitative critique; no structured strategy contract | Legacy |
| `tradingagents/agents/trader/` | Free-form investment proposal with BUY/HOLD/SELL instruction | Not reusable as deterministic compiler input | Legacy |
| `tradingagents/agents/risk_mgmt/` | Aggressive, neutral, conservative prose debate | Not an enforceable risk layer | Legacy |
| `tradingagents/agents/managers/` | Research synthesis and final prose portfolio decision | Qualitative only; README overstates implemented quantitative risk controls | Legacy |
| `tradingagents/llm_clients/` | Provider factory for OpenAI-compatible, Anthropic, and Google models | Reusable, but separate from the new structured client adapter | Legacy/shared candidate |
| `tradingagents/dataflows/` | Yahoo/Alpha Vantage stock, indicator, news, fundamentals routing and caching | Broad coverage but historically unsafe and heavily vendor-shaped | Legacy |
| `tradingagents/agents/utils/memory.py` | In-process BM25 retrieval over prior situations | Non-persistent, unversioned qualitative memory | Legacy |
| `cli/`, `main.py` | Interactive CLI and example runner | Connected only to `TradingAgentsGraph`, not the deterministic research graph | Legacy |
| `results/` | Local prose reports and logs | Ad hoc local artifacts; no deterministic run store | Legacy output |
| `tests/` | 24 Python files, 21 of them untracked; synthetic deterministic and fake-client tests | Good breadth around new code, weak point-in-time/OOS/provider coverage | Mixed |
| `build/`, `tradingagents.egg-info/` | Stale local build and installation metadata | Should not be treated as current deliverables | Generated legacy artifacts |

There are two overlapping systems, not one unified production architecture. The legacy graph produces narrative analysis and a rating. The new research graph produces a schema-backed strategy and deterministic metrics. They share some package exports and potential LLM clients, but the CLI, legacy tools, memory, result persistence, and report flow remain separate.

## 3. Test Baseline

### 3.1 Complete repository invocation

Command: `python -m pytest -ra`

```text
Total collected tests: 229
Passed: 229
Failed: 0
Skipped: 0
Warnings: 0 reported by pytest
Runtime: 124.19 seconds
```

### 3.2 Tests-directory control run

Command: `python -m pytest tests -ra`

```text
Total collected tests: 229
Passed: 229
Failed: 0
Skipped: 0
Warnings: 0 reported by pytest
Runtime: 35.56 seconds
```

The large runtime difference is meaningful. Root `test.py` has import-time executable code, calls `get_stock_stats_indicators_window()` for AAPL, and therefore attempts live Yahoo Finance access during pytest collection even though it defines no test. It is not a controlled integration test and has no assertions. The complete suite is therefore not hermetic or reliably offline.

### 3.3 Test-suite quality assessment

Strengths:

- Schema and validator tests cover many valid/invalid payloads.
- Compiler tests cover all seven operators, nested `ALL`/`ANY`, NaN-to-false behavior, alignment, deterministic repeatability, missing volume, duplicates, and the SMA-200 warm-up regression.
- Backtest tests cover next-bar fills, stops, targets, gaps, collision priority, holding limits, end-of-data closure, costs, and basic accounting.
- Future perturbation tests demonstrate causal indicator/signal calculations before a cutoff.
- Sprint 4 graph tests cover success, retry, validation rejection, compiler failure, backtest failure, data failure, and deterministic regime fallback.

Limitations:

- The LLM and graph tests use queue-based fakes. A real structured-output provider, provider-specific schema behavior, timeouts, token limits, and retry semantics are **UNVERIFIED**.
- The Yahoo adapter test injects a fake downloader. Real Yahoo schema changes, multi-index behavior, timezone behavior, empty results, and rate limiting are **UNVERIFIED**.
- The look-ahead tests do not combine an ATR strategy with next-open execution, which is why the execution-bar ATR leak passed.
- The Sprint 4 golden test uses one dataset both for the final regime snapshot and the entire backtest and asserts only that the stages completed; it codifies the unsafe boundary rather than detecting it.
- The future-data engine test uses a signal provider that ignores prices and only compares an early entry fill, so it does not validate price-dependent signals, stops, or full trade invariance.
- Metric tests intentionally expect gross-PnL profit factor while other trade classification uses net PnL; they do not cover gross-positive/net-negative trades, all-loss `largest_win`, all-win `largest_loss`, negative/zero equity, infinities, or NaNs.
- No tests attempt nested mutation of frozen Pydantic models. A runtime probe successfully appended a condition to a supposedly immutable strategy.
- The broad legacy graph, its date/tool arguments, point-in-time fundamentals/news behavior, persistence, and signal extraction have almost no tests.
- No property-based, randomized invariant, multi-asset, long-duration performance, or OOS tests exist.
- Statement and branch coverage are **UNVERIFIED**: the `coverage` module and coverage configuration are absent.
- Flakiness is **UNVERIFIED** from a single complete run. Most new tests are deterministic, but root `test.py` introduces external variability.

## 4. Architecture Overview

### 4.1 New deterministic path

```text
OHLCV DataFrame
  → data.validate_ohlcv
  → RegimeFeatureEngine.snapshot(last bar)
  → RegimeAnalystAgent (structured or deterministic fallback)
  → StrategyGeneratorAgent (structured draft)
  → StrategyValidator.validate_dict
  → StrategySchema.model_validate
  → StrategyCompiler
      → collect required indicators
      → compute causal indicator Series
      → apply allow-listed operators
      → CompiledStrategy signals
  → BacktestEngine
      → next-bar-open execution
      → Portfolio / TradeExecutor
  → calculate_metrics
  → ResearchRunResult
```

The bridge exists in `tradingagents/graph/research_graph.py`. The graph copies strategy position size and maximum holding period into an effective `BacktestConfig` at lines 279–284, so those schema fields are honored in this path. Direct users of `StrategyCompiler` plus `BacktestEngine` must reproduce that adapter manually because the compiled signal-provider protocol does not carry sizing or holding policy.

### 4.2 Legacy path

```text
Ticker/date
  → tool-calling market/news/social/fundamental analysts
  → free-form reports
  → bull/bear debate
  → research-manager prose plan
  → trader prose recommendation
  → three-way risk debate
  → portfolio-manager prose decision
  → second LLM extracts a rating string
  → local Markdown/JSON/log output
```

No strategy schema, compiler, backtester, metrics object, or OOS contract is part of this path. It has no broker/exchange integration, so it does not directly place trades, but it returns an unvalidated actionable rating that an external caller could misuse.

### 4.3 Dependency direction

The new deterministic packages mostly have sound dependency direction: `strategies` and `backtesting` do not import agents; the compiler depends on the canonical data contract; agents depend on regime/strategy contracts; the research graph orchestrates them. The main architectural weaknesses are duplicated data/indicator logic, public bypass surfaces, generic top-level package naming (`data`), and co-location with a separate legacy architecture whose imports and CLI remain dominant.

## 5. Sprint 1 Audit

| Feature | Status | Evidence | Files | Tests | Risks |
|---|---|---|---|---|---|
| Pydantic v2 strategy schemas | COMPLETE | Uses `ConfigDict`, `field_validator`, `model_validator`, `model_validate`; installed Pydantic is 2.12.4 | `strategies/schema.py` | `test_strategy_schema.py` | Pydantic is not a direct declared dependency |
| Frozen/immutable models | PARTIAL | Field reassignment is blocked, but `ConditionGroup.conditions` is a mutable `list` at line 185. Runtime append succeeded | `strategies/schema.py:179-192` | No nested-mutation test | Post-validation mutation harms reproducibility and contract claims |
| Unknown fields rejected | COMPLETE for strategy models | Every strategy model uses `extra="forbid"` | `strategies/schema.py` | Schema boundary/adversarial tests | `BacktestConfig` does not forbid extras |
| Strategy registry | COMPLETE for the fixed vocabulary | 19 names and seven operators are frozen sets | `strategies/registry.py` | Schema/operator/compiler tests | Other legacy indicator vocabularies remain separate |
| 19 technical indicators | COMPLETE as 19 accepted series names | Five raw OHLCV series plus 14 derived series; all have explicit functions/specs | `indicator_registry.py`, `indicators.py` | `test_indicators.py` | “19 technical indicators” overstates derived count because five are raw fields |
| Seven operators | COMPLETE | Five comparisons and two crossover operators; explicit vectorized functions | `registry.py`, `operator_registry.py` | `test_operator_registry.py` | Operator allow-list and implementation map can drift until tests run |
| Nested groups | COMPLETE with a robustness caveat | Recursive `ConditionGroup` supports `Condition` or nested group; compiler recursively combines | `schema.py`, `compiler.py` | Nested compile and round-trip tests | No explicit maximum depth/complexity bound for untrusted LLM payloads |
| Indicator-vs-value / indicator-vs-indicator structure | COMPLETE | `ConditionValue = Union[float, str]`; string values must be registered indicators | `schema.py:42-44,126-141` | Schema/compiler tests | Numeric constraints accept non-finite values |
| Invalid indicator parameters rejected | PARTIAL | Parameters are not modeled; periods are encoded in fixed names. Extra parameter fields are rejected rather than validated | `schema.py`, `registry.py` | Unknown-field/name tests | No extensible typed parameter contract |
| Semantic validation | PARTIAL | Duplicate warnings and local direct contradictions exist; compiler calls semantic validator | `validator.py`, `compiler.py:177-184` | `test_strategy_validator.py`, compiler audit | Does not reason across nested groups or equivalent expressions |
| Contradiction detection | PARTIAL / INCORRECT EDGE | Detects obvious bounds, but uses `max_lower >= min_upper` without strictness. It rejects satisfiable `RSI >= 50 AND RSI <= 50`; runtime probe confirmed | `validator.py:145-184` | Only obvious contradictions tested | False rejection and misleading error text normalize inclusive operators to strict ones |
| Operator validation | COMPLETE in canonical path | Schema membership check plus compiler resolution to explicit map | `schema.py`, `operator_registry.py`, `compiler.py` | Adversarial and bypass tests | Direct `CompiledStrategy` construction skips semantic validation |
| `validate_dict()` pipeline | COMPLETE but inefficient | Parses JSON/dict, structurally validates, then semantically validates | `validator.py:261-298` | Validator and generator tests | Returns no parsed object; generator reparses the same dict at line 127 |
| Strict schema | PARTIAL | Extra fields are forbidden and literals/allow-lists are strong; models are not `strict=True` and coerce values (`"5"` became `5.0`) | `schema.py` | Type/error tests | NaN/Infinity and coercion undermine a research-grade contract |

### Sprint 1 conclusion

The schema boundary is real and useful, but “immutable” and “strict” are only partially true. A runtime probe verified that `StopLoss(value=NaN)` and `PositionSizing(value=NaN)` are accepted because comparisons such as `v <= 0` are false for NaN. These defects should be repaired before strategy IDs, hashing, caching, or lineage are introduced.

## 6. Sprint 2 Audit

| Feature | Status | Evidence | Files | Tests | Risks |
|---|---|---|---|---|---|
| Bar-by-bar execution | COMPLETE | Explicit `for i in range(n_bars)` loop with documented stage order | `backtesting/engine.py:232-326` | Engine tests | Python loop plus repeated slicing may be slow at scale |
| Historical sliding-window isolation | PARTIAL | Engine passes `df.iloc[:i+1]`; `CompiledStrategy` ignores the supplied slice and reads a precomputed full-frame signal by integer index | `engine.py:246,320`; `compiler.py:78-92` | Structural slice test | Safety depends on compiler causality, not solely structural isolation |
| Next-bar-open execution | COMPLETE | Signals queue `_PendingOrder`; next iteration fills at Open; final-bar signals are not evaluated | `engine.py:239-266,319-326` | Engine/look-ahead tests | None found for ordinary signal fills |
| Slippage | PARTIAL | Buy/sell fills worsen price consistently | `execution.py:36-43` | Execution/portfolio tests | Strategy stop/target levels use raw Open, not actual slipped entry fill |
| Commissions | COMPLETE for percentage-per-leg model | Entry and exit commissions deducted and included in net PnL | `portfolio.py` | Portfolio/engine tests | No minimum/fixed/tiered fees |
| Gap handling | COMPLETE for long stops/targets | Gap down fills at open; gap up target fills at open | `execution.py:65-72` | Dedicated execution tests | Assumes daily OHLC path and unlimited liquidity |
| Stop-loss | PARTIAL | Percentage and ATR-multiple levels exist and are enforced | compiler/executor/engine | Unit tests | ATR execution-bar leak; raw-vs-fill basis; gaps can exceed intended max loss |
| Take-profit | PARTIAL | Percentage and risk/reward targets exist | compiler/executor | Unit tests | Raw-vs-fill basis; no partial fills |
| Conservative SL/TP collision | COMPLETE | Stop wins if both intraday thresholds touch | `execution.py:78-82` | Dedicated collision test | Conservative assumption is explicit and deterministic |
| Portfolio accounting | PARTIAL | Whole-share cash accounting and entry/exit costs are arithmetically coherent | `portfolio.py` | Portfolio and end-to-end tests | Non-finite/over-100% cost inputs can corrupt fills; records are mutable |
| Equity curve | COMPLETE for single asset | One close snapshot per bar, rewritten after EOD closure | `engine.py:303-346` | Engine tests | Exposure observes end-of-bar state, missing intraday-only exposure |
| Total return | COMPLETE under valid positive inputs | Final equity vs initial capital | `metrics.py:35-39` | Hand-calculated test | No independent result invariant test |
| CAGR | PARTIAL | Calendar-day annualization, returns `None` for short or non-positive ending equity | `metrics.py:84-92` | Limited | Negative/zero-equity semantics are untested |
| Max drawdown | COMPLETE under positive finite equity | Running-peak drawdown | `metrics.py:56-60` | Hand-calculated test | Zero/negative/non-finite equity can divide incorrectly |
| Sharpe / Sortino | PARTIAL | Annualized daily equity returns; zero-vol/downside return `None` | `metrics.py:62-81` | Basic edge coverage | No risk-free rate, irregular-calendar handling, or non-finite guard |
| Win rate / expectancy | COMPLETE for documented net-PnL definitions | Classifies and averages net PnL | `metrics.py:115-134` | Hand-calculated test | Zero net PnL is counted as a loss |
| Profit factor | PARTIAL / INCONSISTENT | Uses gross PnL while win/loss classification and expectancy use net PnL | `metrics.py:115-129` | Tests encode gross definition | A gross winner that becomes a net loser is profit in PF but a loss elsewhere |
| Exposure | PARTIAL | Percent of end-of-bar equity points marked invested | `metrics.py:97-99` | One simple test | Understates positions opened and closed intraday or at the open |
| Open position at dataset end | COMPLETE | Forced sell at final Close with slippage, reason `END_OF_DATA` | `engine.py:328-346` | Engine/look-ahead tests | Research protocol must disclose this synthetic liquidation assumption |

### Hidden leakage finding

At a pending entry, the engine calls `get_stop_price(open_p, i, df.iloc[:i+1])` (`engine.py:246-247`). `CompiledStrategy.get_stop_price` retrieves `atr_14.iloc[current_index]` (`compiler.py:121`). ATR at index `i` uses that bar's High, Low, and Close, even though the order fills at its Open. This is direct within-bar future leakage for ATR-multiple stops.

### Edge cases

- Zero trades: handled with `None` for undefined trade metrics.
- One trade: basic behavior works; ratio stability is only lightly tested.
- Flat equity: Sharpe/Sortino become `None`.
- Negative equity: possible with pathological accepted cost inputs; metric behavior is not robustly defined.
- No downside deviation: Sortino becomes `None`.
- Divide-by-zero: common paths are guarded, but zero/non-finite equity arrays are not comprehensively guarded.
- NaNs: canonical data rejects them, but standalone engine validation and configs do not reject all non-finite values.
- Open position at end: deterministically closed.

## 7. Sprint 3 Audit

| Feature | Status | Evidence | Files | Tests | Risks |
|---|---|---|---|---|---|
| Deterministic `StrategyCompiler` | COMPLETE for signal generation | Validates type/semantics/data, computes indicators, emits fixed signals | `strategies/compiler.py` | 30+ compiler/operator/indicator tests | Public `CompiledStrategy` can be constructed directly |
| Explicit indicator registry | COMPLETE | Immutable mapping to explicit functions and warm-up counts | `indicator_registry.py` | Registry/indicator tests | Other modules duplicate formulas/vocabularies |
| Explicit operator registry | COMPLETE | MappingProxyType map; no expression evaluation | `operator_registry.py` | All operators tested | No import-time equality assertion against `SUPPORTED_OPERATORS` |
| Nested `ALL` / `ANY` | COMPLETE | Recursive combination with index checks | `compiler.py:58-71` | Nested tests | No maximum nesting complexity |
| Indicator-vs-value comparisons | COMPLETE | Numeric constant converted to aligned Series | `compiler.py:82-89` | Compiler tests | NaN/Infinity constants not forbidden at schema boundary |
| Indicator-vs-indicator comparisons | COMPLETE | Both operands resolve to registered columns | `compiler.py:76-85` | Compiler tests | None beyond registry drift |
| Crossover / crossunder | COMPLETE | Current strict crossing plus prior inclusive opposite relation; prior-only shift(1) | `operator_registry.py:34-49` | Crossover tests | Definition should be versioned because crossover conventions differ |
| Warm-up metadata | COMPLETE | Each indicator has `warmup_bars`; rows before it forced NaN | `indicator_registry.py`, `indicators.py:174-176` | Warm-up tests | Formula definitions are not versioned |
| SMA-200 bug fix | COMPLETE and regression-tested | `sma_200` warm-up is 200; first 199 signals false; bar 199 becomes available | registry/indicators | `test_sprint3_compiler_audit.py:85-93`, regime warm-up test | Legitimate fix |
| NaN → false | COMPLETE in operator layer | Validity mask plus `fillna(False).astype(bool)` | `operator_registry.py` | Operator/compiler tests | Silent NaNs are correct during warm-up but input data NaNs must remain rejected |
| Typed exceptions | COMPLETE for compiler/data failures | Dedicated compilation and market-data exception hierarchies | `strategies/exceptions.py`, `data/schema.py` | Negative-path tests | Graph often catches broad `Exception` and reduces detail to strings |
| Canonical OHLCV validation | COMPLETE in new path | Case-insensitive canonicalization, finite/positive values, ordered unique DatetimeIndex | `data/schema.py` | Market-data tests | Backtest engine maintains a weaker duplicate validator |
| Market-data provider abstraction | COMPLETE as an interface | Runtime-checkable provider Protocol | `data/provider.py` | Fake provider tests | Only one concrete new adapter |
| Yahoo Finance adapter | COMPLETE at code level; live behavior **UNVERIFIED** | Injectible downloader, canonical validation | `data/provider.py` | Fake-downloader test | No live/recorded contract test or provider metadata |
| In-memory copy-safe cache | COMPLETE | Locked store; deep copy on set/get | `data/cache.py` | Mutation test | No TTL, capacity, disk persistence, or corporate-action versioning |
| Timezone-preserving indexes | COMPLETE | Canonical copy retains index; optional timezone requirement | `data/schema.py` | Timezone test | Provider-specific timezone semantics unverified live |
| No `eval()` / `exec()` | COMPLETE in deterministic path | AST scan found no calls to dangerous built-ins; `compile` matches are graph/compiler methods | New source tree | Adversarial tests | Keyword filtering is not the main safety boundary; schema is |
| Future-data perturbation tests | COMPLETE for causal feature/signal prefixes | Future changes do not alter earlier indicator, regime-feature, or signal prefixes | Sprint 3/regime tests | Perturbation tests | Does not cover execution-time ATR or sample selection |

### Duplication and drift risks

- `strategies/registry.py`, `indicator_registry.py`, and the retained `REGISTRY_TO_STOCKSTATS` map represent one deterministic vocabulary in three structures. Two import-time assertions protect indicator key equality.
- The legacy market analyst, Yahoo stockstats implementation, and Alpha Vantage indicator adapter have different indicator lists and definitions.
- `regime/features.py` independently recomputes SMA series and imports only the ATR helper.
- `data/schema.py` and `backtesting/engine.py::_validate_and_normalize_data` enforce different contracts. The compiler requires Volume, unique timestamps, finite positive prices; the standalone engine does not enforce all of these.
- `compute_indicators(strict=False)` retains a silent-skip/NaN compatibility mode. The compiler correctly uses `strict=True`, but other callers could silently degrade.

## 8. End-to-End Deterministic Pipeline

### 8.1 Verified trace

1. Structured client returns `StrategyProposalDraft`, whose outer fields are Pydantic-validated.
2. `StrategyGeneratorAgent.validate_candidate()` calls `StrategyValidator.validate_dict()` on the raw strategy dict.
3. A valid candidate is reparsed into `StrategySchema` and promoted to `StrategyProposal`.
4. `Sprint4ResearchGraph` passes that exact strategy to `StrategyCompiler.compile()`.
5. Compiler re-runs semantic validation, validates canonical OHLCV, resolves all required indicators, and creates aligned boolean entry/exit Series.
6. The graph creates an effective backtest config using strategy position size and maximum holding days.
7. `BacktestEngine` consumes the compiled strategy, creates trades/equity, and calls deterministic metrics.
8. `ResearchRunResult` exposes the authoritative `BacktestMetrics` from `BacktestResult`.

This path is exercised by `tests/test_sprint4_integration.py` and failure-routing tests.

### 8.2 Missing/manual bridges

- The graph accepts a DataFrame directly; it does not accept a `MarketDataProvider`, capture provider identity, or fetch a versioned dataset.
- The LLM client factory returns general chat models. No production composition function wraps one in `LangChainStructuredOutputClient` and constructs the new research graph.
- The CLI and `main.py` instantiate only the legacy graph.
- There is no public one-call raw Strategy JSON → deterministic result API outside the LLM graph.
- Direct compiler/backtester users must map position sizing and holding days into `BacktestConfig` manually.
- No persistence or report builder consumes `ResearchRunResult`.
- No explicit train/context/validation/test data types enforce sample separation.

### 8.3 Classification

**PROTOTYPE ONLY.** The path is functional, typed in important places, and deterministically testable, but it is not research-ready because strategy generation sees the evaluation endpoint, ATR stop placement leaks within the execution bar, metrics are not yet ranking-safe, and provenance/OOS controls are absent.

## 9. Legacy Multi-Agent Architecture

| Component | Current behavior | Classification | Recommendation |
|---|---|---|---|
| Market analyst | Tool-calling LLM produces a long Markdown technical report | EXISTING BUT NEEDS REFACTOR | Retain only as optional context after server-side date enforcement; never as signal authority |
| News analyst | Tool-calling LLM produces prose from company/global news | EXISTING BUT NEEDS REFACTOR | Require point-in-time article timestamps and bounded server-set dates |
| Social media analyst | Uses company news/search tool and outputs prose sentiment report | EXISTING BUT NEEDS REFACTOR | Optional qualitative feature, with provenance and temporal validation |
| Fundamentals analyst | Uses live overview and fiscal-period-filtered statements; outputs prose | EXISTING BUT NEEDS REFACTOR | Replace with as-of filing/release-date data before historical use |
| Bull/bear researchers | Debate accumulated prose and memories | EXISTING & REUSABLE only as critique | Convert to typed critiques of a schema candidate; no direct mutations |
| Research manager | Chooses a stance and produces an investment plan in prose | EXISTING BUT NEEDS REFACTOR | Do not treat as deterministic strategy selection or score |
| Trader | Produces BUY/HOLD/SELL prose | SHOULD NOT BE REUSED for strategy compilation | Supersede with schema generator and deterministic evaluator |
| Aggressive/neutral/conservative risk agents | Persuasive prose debate; no calculations or enforced limits | EXISTING BUT NEEDS REFACTOR | May generate bounded critiques, but deterministic risk engine must decide constraints |
| Portfolio manager | Free-form rating and plan | SHOULD NOT BE REUSED as risk authority | May summarize validated outputs only |
| Graph/router | LangGraph routing, bounded debate counts, tool loops | EXISTING & REUSABLE conceptually | Prefer the new research graph; avoid merging free-form state into authoritative state |
| LLM client factory | Provider abstraction for several model APIs | EXISTING & REUSABLE | Add an explicit structured-client composition layer |
| BM25 memory | In-process retrieval of past prose/reflections | PARTIAL | Do not use for research optimization until data scope, persistence, and lineage are defined |
| Legacy dataflows | Multiple vendors and tool wrappers | EXISTING BUT NEEDS REFACTOR | Do not feed historical research until point-in-time correctness is proven |

All legacy analyst, researcher, trader, risk, and manager outputs are strings. `AgentState` is a TypedDict, but its financially authoritative fields are unvalidated prose. `SignalProcessor` asks another LLM to return one rating word and returns `response.content` without enum validation. There is no direct brokerage code, so agents have no in-repository order execution authority. They do have authority to call the finite read-only data tool set and choose date arguments; those dates are not centrally clamped to `trade_date`.

**DOCUMENTATION MISMATCH:** README line 93 claims the risk team evaluates volatility, liquidity, and other risk factors and adjusts strategies. The implementation supplies prose prompts and reports; it contains no deterministic liquidity model, exposure constraint, sizing engine, or portfolio-risk calculation.

## 10. Look-Ahead & Leakage Audit

### 10.1 Repository-wide search results

| Pattern/risk | Result | Context verdict |
|---|---|---|
| `shift(-n)` | No production matches | No direct future shift found |
| `rolling(..., center=True)` | No production matches | No centered-window leak found |
| `iloc[i + 1]` / `iloc[t + 1]` | No production future-index matches | Execution advances by loop state, not future reads |
| `.bfill()` / `.backfill()` | One production match in `stockstats_utils.py:42` | **UNSAFE** in historical features, especially because cutoff occurs afterward |
| Positive `shift(1)` | Crossover prior operands and ATR prior close | **SAFE**, causal use |
| Forward fill | Paired with backfill in legacy cleanup | Forward fill can be defensible; the following backfill is not |
| Future joins | No matches found | No identified join leakage |
| Full-dataset normalization/scaler fit | No matches found | Not currently implemented |
| Full-dataset parameter optimizer | No matches found | Not currently implemented |
| Ranking on holdout | No ranking/holdout implementation | Missing rather than leaking yet |

### 10.2 Unsafe occurrences and architectural leakage

**A. Same-sample regime/strategy generation and backtest — unsafe.**  
`research_graph.py:140` calls `feature_engine.snapshot(state["market_data"])`; snapshot always selects `features.iloc[-1]`. At `research_graph.py:288`, the engine backtests that generated strategy on the same full `state["market_data"]`. The LLM receives the last-bar market summary and then its strategy is evaluated on all earlier bars. Even though feature formulas are causal, the research protocol is not.

**B. ATR stop at next-open — unsafe.**  
At the entry Open, the stop uses ATR at the execution-bar index. ATR incorporates that day's full range and close. Runtime perturbation proved the stop changes when only post-open values on the execution bar change.

**C. Legacy price cleanup — unsafe.**  
`stockstats_utils.py:42` performs `.ffill().bfill()` on data downloaded through today. `load_ohlcv` applies `Date <= curr_date` only afterward at line 88. Backfill can substitute a later observation into an earlier missing value, including across the historical cutoff.

**D. Legacy fundamentals — unsafe for point-in-time research.**  
Yahoo `ticker.info` explicitly ignores `curr_date` (`y_finance.py:248-255`) and exposes current TTM, forward estimates, price averages, and ratios. Statement columns are filtered by fiscal period end, not by public filing/release timestamp. Alpha Vantage uses the same fiscal-end logic, and company overview is unfiltered. This creates reporting-lag leakage.

**E. Legacy news and insider data — partial/unsafe.**  
Flat Yahoo news items are assigned `pub_date=None` and bypass date filtering. Company news permits a boundary up to `end + 1 day`. Insider-transaction functions accept no historical cutoff. LLM-chosen tool dates are not clamped server-side.

**F. Cache boundaries.**  
The new cache keys symbol/start/end/interval and deep-copies frames, so mutation and cross-range reuse were not found. It does not record dataset version, retrieval time, corporate-action settings, or split role. The legacy cache stores a current rolling history then filters it, which compounds the backfill issue.

### 10.3 Rating

**Look-Ahead Safety Rating: 4/10.** Causal indicator formulas and next-bar signal execution are strong, but the current end-to-end research graph is sample-aware, ATR stops leak within the fill bar, and legacy point-in-time data is not safe. Those are result-invalidating defects, not minor methodology preferences.

## 11. LLM Safety Audit

An AST scan across 91 source files (10,704 lines) found no calls to Python's dangerous `eval`, `exec`, `compile`, or `__import__` built-ins. The three `compile` call-name matches are ordinary methods on LangGraph workflows or `StrategyCompiler`. No model-generated Python is executed, no dynamic module path comes from model output, and the new structured agents have no tools or trading authority.

### New strategy pathway

- Native/provider structured output is requested with a Pydantic schema.
- The proposal envelope forbids extra fields.
- The strategy remains a raw dict until `StrategyValidator.validate_dict()` succeeds.
- It is reparsed into `StrategySchema`, then semantically revalidated by `StrategyCompiler`.
- Indicators and operators resolve only through explicit maps.
- Explanations are strings and are never executed.

This pathway has no identified schema bypass in `Sprint4ResearchGraph`. Public APIs can still be misused: callers can instantiate `CompiledStrategy` directly or create invalid Pydantic objects with `model_construct`/unchecked `model_copy`, but the compiler catches unsupported indicators/operators and the graph uses the proper gates.

### Legacy pathways

- Analyst tool use is allow-listed, but the LLM controls arguments such as dates and lookback lengths.
- Prose from one agent is embedded unescaped into later prompts. Prompt injection can propagate through reports/news content.
- Trader, risk, and portfolio decisions are not schema-validated.
- `SignalProcessor` returns arbitrary model text without enum parsing.
- No in-repository broker order placement exists.

The keyword blacklist in `StrategyProposalDraft` is defense-in-depth only and is easy to evade, but this does not become code execution because those fields are never evaluated. The real protection is the declarative schema plus explicit compiler.

**LLM Execution Safety Rating: 7/10.** The new strategy execution boundary is strong. The score is reduced for the parallel legacy free-form decision path, prompt-injection propagation, unbounded data-tool arguments, and lack of a validated final action contract.

## 12. Testing Quality

| Area | Assessment |
|---|---|
| Unit coverage breadth | Good for new schema/compiler/backtest mechanics |
| Integration coverage | One synthetic golden graph and several fake failure routes; no live provider/model composition |
| Edge cases | Moderate; important non-finite, inclusive-bound, cost-basis, same-bar ATR, and negative-equity cases missing |
| Regression tests | SMA-200 warm-up is properly regression-tested |
| Determinism | Repeated indicators/compiler/backtest covered; LLM seed enforcement is not |
| Mocking | Appropriate for unit tests but too dominant for provider/model integration claims |
| Legacy coverage | Very low |
| OOS/research protocol | Absent |
| Hermeticity | Broken by root `test.py` import-time live call |
| Coverage measurement | **UNVERIFIED**; tooling absent |

Tests are meaningful enough to support “functional prototype,” but not the stronger claims of look-ahead safety or research readiness. The most important new tests are: execution-bar perturbation for ATR stops, separate context/evaluation sample enforcement, point-in-time vendor fixtures, non-finite contract rejection, trade/accounting conservation invariants, and holdout access denial.

## 13. Engineering Quality

### 13.1 Scores

```text
Architecture: 6/10
Correctness: 5/10
Testing: 6/10
Maintainability: 5/10
Reproducibility: 3/10
Security: 6/10
Quant Research Quality: 4/10
```

### 13.2 Rationale

**Architecture — 6/10.** The deterministic layers are cohesive and mostly point inward toward plain contracts. The score is constrained by two parallel architectures, duplicate registries/validators/features, public bypass surfaces, generic top-level package names, and missing production composition.

**Correctness — 5/10.** Basic signal, fill, cash, and indicator behavior is sound under ordinary inputs. Two direct leakage defects and metric/contract edge errors are material.

**Testing — 6/10.** There are 229 passing tests with useful negative and regression coverage. Real providers/models, OOS logic, property invariants, legacy behavior, and the discovered defects are not covered; collection is not hermetic.

**Maintainability — 5/10.** Code is readable and typed in the new core, with sensible exceptions. Duplication, broad catches, stale generated artifacts, wildcard imports, mutable default lists, and lack of lint/type/coverage automation lower confidence.

**Reproducibility — 3/10.** Most audited work is untracked; seeds are recorded but not enforced by the structured adapter; dataset hashes/provider versions/config snapshots are absent; current vendor data is mutable.

**Security — 6/10.** No generated-code execution exists and secrets are environment-based with `.env` ignored. Legacy prompt injection, unvalidated final ratings, possible ticker path traversal in legacy logging, and unredacted exception strings remain.

**Quant Research Quality — 4/10.** Causal indicators and next-bar fills are a strong base. There is no valid sample protocol, no OOS/walk-forward/holdout layer, limited metrics, no ranking/viability controls, and no sensitivity/Monte Carlo analysis.

### 13.3 Performance

Positive aspects include vectorized indicator computation, computation of only referenced indicators, one-time compiled signals, and copy-safe cache access. Costs include repeated validation/deep copies at graph, feature, compiler, and engine boundaries; repeated rolling calculations for related Bollinger/MACD/regime features; an unbounded cache; per-bar DataFrame slicing even though `CompiledStrategy` ignores the slice; and no benchmark or scale test. Large-universe/multi-year performance is **UNVERIFIED**.

## 14. Critical Findings

### P0 — Critical

#### ID: P0-01

- **Severity:** P0
- **Title:** Strategy generation sees the evaluation sample's endpoint
- **Affected files:** `tradingagents/graph/research_graph.py:140,288`; `regime/features.py:91-120`
- **Problem:** The regime snapshot uses the final row of the same frame later backtested in full.
- **Why it matters:** The generated strategy is conditioned on future information relative to most simulated trades. Reported performance is not a valid historical evaluation.
- **Evidence:** Direct call-flow trace; the golden integration test passes the same 300-bar frame through both stages.
- **Recommended fix:** Introduce typed `ResearchDatasetSplit` inputs. Generate only from training/context data ending at or before a declared decision timestamp. Evaluate only on a subsequent validation segment; keep the final holdout inaccessible to agents and ranking.
- **Should block next sprint?** YES

#### ID: P0-02

- **Severity:** P0
- **Title:** ATR stop uses execution bar's future High/Low/Close at its Open
- **Affected files:** `backtesting/engine.py:242-250`; `strategies/compiler.py:112-122`; `strategies/indicator_registry.py:57-76`
- **Problem:** ATR stop placement at next-bar Open reads ATR for that full bar.
- **Why it matters:** The stop distance and resulting PnL depend on price action that occurs after the simulated decision/fill time.
- **Evidence:** Runtime perturbation with the same signal and execution Open changed only execution-bar range; stop changed from `104.8061` to `-37.6224`.
- **Recommended fix:** At next-open entry, use indicator state from the signal bar (`i-1`) or a separately modeled pre-open feature. Add an execution-bar future-perturbation regression test.
- **Should block next sprint?** YES

#### ID: P0-03

- **Severity:** P0
- **Title:** Legacy historical inputs are not point-in-time safe
- **Affected files:** `tradingagents/dataflows/stockstats_utils.py:35-42,48-88`; `y_finance.py:248-299,305-419`; `yfinance_news.py:40-48,81-88`; `alpha_vantage_fundamentals.py:4-18`
- **Problem:** Price gaps are back-filled before cutoff; live overview/insider fields ignore historical date; statements use fiscal end rather than availability date; undated news bypasses filtering.
- **Why it matters:** Any backdated legacy agent analysis can incorporate information unavailable at the stated trade date.
- **Evidence:** Source paths and ordering listed above.
- **Recommended fix:** Quarantine legacy dataflows from research; replace with point-in-time adapters using availability timestamps, server-enforced cutoff dates, and no backward fill.
- **Should block next sprint?** YES if legacy context will be reused

### P1 — High

#### ID: P1-01

- **Severity:** P1
- **Title:** Risk levels are based on raw Open, not actual slipped entry fill
- **Affected files:** `backtesting/engine.py:247-255`; `backtesting/portfolio.py:70-94`; `strategies/compiler.py:97-135`
- **Problem:** Stop and target callbacks receive `open_p`, then the portfolio buys at `open_p * (1 + slippage)`. A 5% stop is therefore not 5% below actual cost.
- **Why it matters:** Risk/reward and realized loss assumptions are systematically distorted whenever slippage is nonzero.
- **Evidence:** Call order and formulas in the cited files.
- **Recommended fix:** Compute the executable entry fill first, derive risk levels from that fill using only information available at the open, then commit the position atomically.
- **Should block next sprint?** YES

#### ID: P1-02

- **Severity:** P1
- **Title:** Metrics are not internally consistent enough for ranking
- **Affected files:** `backtesting/metrics.py:115-133`; `backtesting/models.py:379-403`
- **Problem:** Profit factor uses gross PnL while wins/losses use net PnL. `largest_win` and `largest_loss` are max/min over all trades even when no winning/losing trade exists.
- **Why it matters:** A fee-eroded gross winner can count as profit in one metric and a loss in another. Rankings and viability gates would be misleading.
- **Evidence:** Probe with two net losers returned `winning_trades=0`, `profit_factor=1.0`, and `largest_win=-1`.
- **Recommended fix:** Version metric definitions, consistently use net economics (or expose explicitly named gross and net variants), and return `None` when a directional subset is empty.
- **Should block next sprint?** YES, especially Sprint 5

#### ID: P1-03

- **Severity:** P1
- **Title:** Strategy/config contracts accept non-finite values and are not deeply immutable
- **Affected files:** `strategies/schema.py:179-192,279-405`; `backtesting/models.py:119-176`
- **Problem:** NaN passes comparison validators; nested condition lists are mutable; BacktestConfig ignores unknown fields and accepts NaN/Infinity. Cost percentages have no realistic upper bound.
- **Why it matters:** Invalid inputs can corrupt quantities, fills, equity, hashes, or reproducibility after validation.
- **Evidence:** Runtime probes accepted NaN StopLoss, PositionSizing, and initial capital; appended to frozen conditions; silently ignored a misspelled commission field.
- **Recommended fix:** Use finite-number constraints, strict models, `extra="forbid"`, immutable tuples, and explicit cost bounds. Revalidate copied/constructed models at trust boundaries.
- **Should block next sprint?** YES

#### ID: P1-04

- **Severity:** P1
- **Title:** Audited implementation is not reproducible from Git
- **Affected files:** all new deterministic/Sprint 4 packages and tests; three modified exports/config files
- **Problem:** 58 files were untracked before audit; only three old tests are tracked.
- **Why it matters:** A clone of HEAD cannot reproduce features, failures, or the passing test baseline. Faculty review and rollback lack an auditable change set.
- **Evidence:** `git ls-files --others --exclude-standard` and `git status` inventory.
- **Recommended fix:** After correcting P0/P1 issues, review and commit source/tests in coherent changes; exclude generated result artifacts; regenerate package artifacts in CI.
- **Should block next sprint?** YES

#### ID: P1-05

- **Severity:** P1
- **Title:** No enforceable train/validation/test or holdout protocol exists
- **Affected files:** new research graph and all future evaluation/refinement modules (currently absent)
- **Problem:** Data is an undifferentiated DataFrame. There are no chronological split types, access controls, walk-forward runner, or untouched holdout.
- **Why it matters:** Evaluation, ranking, and refinement can accidentally optimize on the same data, as the current graph already does.
- **Evidence:** No split/holdout/walk-forward/Monte Carlo code found; graph accepts one frame.
- **Recommended fix:** Make split provenance a required type and graph input before building ranking or refinement.
- **Should block next sprint?** YES

### P2 — Medium

#### ID: P2-01

- **Severity:** P2
- **Title:** Inclusive-bound contradiction logic rejects a valid equality case
- **Affected files:** `strategies/validator.py:145-184`
- **Problem:** `x >= a AND x <= a` is flagged as impossible.
- **Why it matters:** Valid strategies can be rejected and feedback to the LLM is factually wrong.
- **Evidence:** Runtime probe with RSI inclusive bounds at 50 returned invalid.
- **Recommended fix:** Track bound inclusivity and reject equal endpoints only when at least one bound is strict.
- **Should block next sprint?** NO, but fix in Sprint 3.5

#### ID: P2-02

- **Severity:** P2
- **Title:** Indicator and data contracts are duplicated across architectures
- **Affected files:** `strategies/*registry.py`, `strategies/indicators.py`, `regime/features.py`, `tradingagents/dataflows/*`, `data/schema.py`, `backtesting/engine.py`
- **Problem:** Multiple vocabularies, formulas, and validators can drift.
- **Why it matters:** The same named research concept may produce different values or accept different data depending on path.
- **Evidence:** Three new registry structures, separate legacy lists, regime SMA duplication, and two OHLCV validators.
- **Recommended fix:** Establish versioned canonical indicator/data contracts and adapters for legacy display names.
- **Should block next sprint?** NO

#### ID: P2-03

- **Severity:** P2
- **Title:** Run metadata is incomplete and seed metadata is not enforced
- **Affected files:** `tradingagents/agents/structured_output.py:31-56`; `tradingagents/graph/research_models.py:36-55`; `research_graph.py:338-355`
- **Problem:** The adapter stores `seed` but does not configure the underlying model. There are no run/strategy IDs, schema version, dataset hash/row count/timezone/provider, dependency version, or full configuration snapshot.
- **Why it matters:** Two nominally identical runs cannot be independently reproduced or compared with confidence.
- **Evidence:** Metadata field inventory and adapter implementation.
- **Recommended fix:** Generate IDs, capture canonical JSON config and data fingerprint, and pass supported deterministic parameters into model construction while recording provider support.
- **Should block next sprint?** NO, but required before results are retained

#### ID: P2-04

- **Severity:** P2
- **Title:** Public deterministic APIs permit weaker or bypassed contracts
- **Affected files:** `strategies/compiler.py:41-56`; `backtesting/engine.py:118-170`; `strategies/indicators.py:98-179`
- **Problem:** `CompiledStrategy` is publicly constructible without semantic validation; standalone engine validation is weaker than canonical validation; non-strict indicator mode silently skips unknowns.
- **Why it matters:** Alternate call paths can evade guarantees documented for the canonical pipeline.
- **Evidence:** Source API behavior; tests deliberately use model construction bypasses.
- **Recommended fix:** Make unsafe constructors/internal modes private or explicitly named, and require one canonical market-data validator.
- **Should block next sprint?** NO

#### ID: P2-05

- **Severity:** P2
- **Title:** New research graph has no production entry point or persistence
- **Affected files:** `cli/main.py`, `main.py`, `tradingagents/graph/research_graph.py`
- **Problem:** CLI and examples instantiate only `TradingAgentsGraph`; no provider/client composition or result-store/report path exists for `Sprint4ResearchGraph`.
- **Why it matters:** Passing fake-client tests do not prove a usable application workflow.
- **Evidence:** Reference search found new graph usage only in package exports and tests.
- **Recommended fix:** After research-boundary repairs, add a controlled composition root and non-interactive CLI/API command.
- **Should block next sprint?** NO; complete during Sprint 4/8

#### ID: P2-06

- **Severity:** P2
- **Title:** Test collection performs uncontrolled live network work
- **Affected files:** root `test.py`
- **Problem:** Module-level code fetches Yahoo data during collection and has no assertions.
- **Why it matters:** Baseline runtime and pass/fail behavior depend on external state; offline CI may hang or mislead.
- **Evidence:** Complete run 124.19 s versus tests-only 35.56 s; source inspection.
- **Recommended fix:** Move to an explicit manual example or a marked integration test with mocked/recorded data and assertions.
- **Should block next sprint?** NO

#### ID: P2-07

- **Severity:** P2
- **Title:** Legacy final action and output path are insufficiently validated
- **Affected files:** `tradingagents/graph/signal_processing.py:13-33`; `trading_graph.py:256-267`; `cli/utils.py:41-43`
- **Problem:** Rating output is arbitrary LLM text; ticker normalization allows path separators and is used as a result directory segment.
- **Why it matters:** Downstream automation may trust an invalid action, and package/API callers can influence filesystem paths.
- **Evidence:** No enum parsing; only `strip().upper()`; ticker joined directly into `results_dir`.
- **Recommended fix:** Parse a strict action enum and validate symbol syntax or escape it to a safe stable identifier.
- **Should block next sprint?** NO if legacy execution remains disconnected

#### ID: P2-08

- **Severity:** P2
- **Title:** Exposure and holding-period semantics are underspecified
- **Affected files:** `backtesting/metrics.py:97-99`; `portfolio.py:133-151`; `models.py:262-264`
- **Problem:** Exposure counts end-of-bar invested flags; same-day/open exits disappear. Documentation says holding days are inclusive, code uses date difference and can return zero.
- **Why it matters:** Metrics can be misinterpreted in evaluation and reports.
- **Evidence:** Formula and documentation mismatch.
- **Recommended fix:** Define bar exposure, capital-time exposure, and holding duration precisely; add tests for same-day trades.
- **Should block next sprint?** NO, but resolve before Sprint 5 scoring

### P3 — Low

#### ID: P3-01

- **Severity:** P3
- **Title:** Documentation and generated package artifacts are stale or overstated
- **Affected files:** `README.md`, `build/`, `tradingagents.egg-info/`, module docstrings
- **Problem:** README does not document the deterministic graph and overstates risk capabilities; build metadata omits new packages; stockstats docstring says 15 years while code downloads five; backtest result records are called immutable but dataclasses are mutable.
- **Why it matters:** Reviewers and users cannot infer actual guarantees from documentation.
- **Evidence:** Source/document comparison and runtime mutation probe.
- **Recommended fix:** Update documentation only after contracts stabilize; remove/regenerate build artifacts.
- **Should block next sprint?** NO

#### ID: P3-02

- **Severity:** P3
- **Title:** Developer-quality automation is absent
- **Affected files:** repository configuration
- **Problem:** No CI definition, lint/type/coverage config, dev dependency group, or benchmark suite was found.
- **Why it matters:** Quality currently depends on local discipline and one environment.
- **Evidence:** Repository inventory and `pyproject.toml` inspection.
- **Recommended fix:** Add pinned development tooling and CI gates after the working tree is brought under version control.
- **Should block next sprint?** NO

## 15. Verified Work Completed Till Date

### Foundation / Architecture

- **Implemented, tested, audited:** separate deterministic packages for strategy, data, and backtest logic with no agent dependency.
- **Implemented, tested, audited:** canonical new OHLCV contract with typed failures and copy preservation.
- **Implemented, tested, audited:** explicit operator/indicator execution maps with no arbitrary generated-code execution.
- **Implemented, partially verified:** provider-independent market-data interface, Yahoo adapter, and copy-safe cache. Live provider behavior is **UNVERIFIED**.
- **Implemented, tested, audited:** a parallel typed Sprint 4 research graph, currently prototype-only because of sample leakage.

### Sprint 1

- **Implemented and tested:** Pydantic v2 declarative strategy model, nested conditions, long-only daily scope, risk fields, and extra-field rejection.
- **Implemented and tested:** 19 accepted series names (five raw, 14 derived) and seven operators.
- **Implemented and tested:** combined JSON/dict structural plus semantic validation.
- **Partially verified:** frozen model behavior. Top-level fields are frozen; nested lists remain mutable.
- **Partially verified:** contradiction detection. Useful for simple strict bounds, incorrect for inclusive equal bounds.

### Sprint 2

- **Implemented and tested:** bar loop, next-open signal orders, final-bar policy, gap fills, commissions, slippage, long-only single-position cash accounting, stops/targets, collision priority, EOD liquidation, and equity curve.
- **Implemented and tested:** total return, CAGR, drawdown, Sharpe, Sortino, trade counts, win rate, profit factor, expectancy, average/largest trades, holding days, and exposure.
- **Partially verified:** research correctness of ATR stops and metric semantics; both have audited defects.
- **Partially verified:** look-ahead isolation. Signal formulas are causal; full pipeline isolation is not.

### Sprint 3

- **Implemented, tested, audited:** deterministic compiler, nested Boolean compilation, value/indicator comparisons, crossover/crossunder, warm-up metadata, NaN-to-false, typed compiler failures, and canonical data validation.
- **Implemented, tested, audited:** SMA-200 warm-up fix; first 199 rows unavailable and regression-protected.
- **Implemented, tested, audited:** future-prefix perturbation tests for indicator/signal causality.
- **Implemented and tested with fakes:** Yahoo adapter contract and cache mutation safety.
- **Partially verified:** real vendor behavior and packaged distribution.

### Existing Legacy Components That Can Be Reused

- LangGraph routing patterns and bounded debate counters.
- Provider/model client factory after adaptation to structured output.
- Selected analysts as non-authoritative context generators after point-in-time remediation.
- BM25 mechanics as an optional retrieval primitive after lineage/scope/persistence design.
- Markdown CLI display and report formatting ideas, not the current authoritative data model.

## 16. Remaining Development Work

### Sprint 4 — Safe Agent Orchestration & Market Regime

- **Objective:** complete a production-composable structured graph without contaminating evaluation data.
- **Current existing support:** regime features/contracts/classifier, structured regime and strategy agents, bounded attempts, validation/compile/backtest gates, typed statuses/events, fake-client integration tests.
- **Missing components:** explicit context/train versus evaluation inputs; decision timestamp; real client composition; data provider composition; run/strategy IDs; dataset provenance; correctly enforced seed/temperature; safe serialized result; stage-specific error contracts.
- **Required files/modules:** revise `research_graph.py`, `research_state.py`, `research_models.py`; add a research dataset/split contract and structured-client factory/composition root.
- **Key tests:** no overlap between context and evaluation; final evaluation bars absent from prompts; deterministic fallback; real provider-adapter contract fixture; JSON serialization; metadata completeness.
- **Dependencies:** Sprint 3.5 correctness fixes and split protocol.
- **Complexity:** HIGH

### Sprint 5 — Evaluation & Strategy Ranking

- **Objective:** create deterministic, transparent candidate viability and ranking.
- **Current existing support:** total return, CAGR, drawdown, Sharpe, Sortino, trade count, win rate, profit factor, expectancy, average win/loss, largest win/loss, holding duration, exposure.
- **Missing components:** Calmar ratio, recovery factor, payoff ratio, turnover, corrected profit factor, minimum trade count, viability gates, comparison model, deterministic scoring, score decomposition, uncertainty/insufficient-sample handling, performance analyst contract.
- **Required files/modules:** preferably a new `evaluation/` package with versioned metric definitions, `models.py`, `gates.py`, `ranking.py`; carefully extend `backtesting/metrics.py` only for execution-derived primitives.
- **Key tests:** independently hand-calculated metrics, scale invariance, fee-eroded trades, zero/all-win/all-loss cases, deterministic tie-breaking, transparent decomposition sums, minimum-sample rejection.
- **Dependencies:** corrected metrics and validation-only evaluation data.
- **Complexity:** HIGH

### Sprint 6 — Risk Management & Refinement

- **Objective:** enforce quantitative risk constraints and a bounded, auditable refinement loop.
- **Current existing support:** required stop/target, fixed-percentage sizing, one concurrent long position, basic ATR stop, bounded initial generation attempts, legacy qualitative risk debate.
- **Missing components:** formal max-loss budget, gap-risk policy, max portfolio exposure, volatility/ATR sizing, liquidity/capacity, multi-position concurrency, concentration, portfolio covariance/risk, drawdown controls, typed risk decisions, bounded refinement, lineage/parent-child IDs, overfit rejection.
- **Required files/modules:** new `risk/` calculation/constraint models and `refinement/` lineage/loop modules; do not reuse prose debaters as enforcement.
- **Key tests:** risk limits under gaps/costs, sizing invariants, liquidity caps, multi-asset concentration, drawdown halt, bounded loop termination, immutable lineage, rejection when validation improves but OOS degrades.
- **Dependencies:** Sprint 5 viability plus Sprint 7 OOS protocol before LLM refinement is enabled.
- **Complexity:** VERY HIGH

### Sprint 7 — Out-of-Sample Validation

- **Objective:** make research conclusions resistant to selection and overfitting.
- **Current existing support:** causal calculations and prefix perturbation tests only.
- **Missing components:** chronological train/validation/test split, untouched holdout, walk-forward runner, parameter/cost/slippage sensitivity, Monte Carlo/trade skipping, regime-level and rolling performance, multiple-testing/selection controls, split-aware caching and provenance.
- **Required files/modules:** new `validation/` package with `splits.py`, `walk_forward.py`, `sensitivity.py`, `monte_carlo.py`, and protocol/result contracts.
- **Key tests:** strict timestamp ordering/no overlap, holdout access denial, embargo/warm-up handling, stable seeded simulations, split-aware cache keys, no rank/refinement call receives holdout metrics.
- **Dependencies:** repaired graph boundary and stable metrics.
- **Complexity:** VERY HIGH

### Sprint 8 — Integration, Observability & Hardening

- **Objective:** make every run reproducible, inspectable, persistable, and operable.
- **Current existing support:** Python logging for stage outcomes, Pydantic run result/status/event objects, model/prompt labels, backtest config, legacy Markdown output, environment-based secrets, CLI framework.
- **Missing components:** run/strategy IDs, schema versions, canonical config snapshots, data hash/provider/retrieval/corporate-action metadata, dependency/git version, enforced random seeds, execution-assumption version, lineage, JSON artifact store, Markdown/HTML/PDF research reports, database/object persistence, secret redaction, retry/timeout/cancellation policy, deterministic CLI/API, CI and end-to-end production tests.
- **Required files/modules:** `observability/`, `persistence/`, and `reporting/` packages plus a new CLI/API composition entry point.
- **Key tests:** byte-stable canonical artifacts where appropriate, migration/version tests, secret-redaction tests, restart/reload, report rendering, full offline golden run, controlled provider failure.
- **Dependencies:** stable contracts from Sprints 3.5–7.
- **Complexity:** HIGH

## 17. Sprint 4–8 Gap Analysis

### Sprint 4 capability classification

| Capability | Classification | Audit note |
|---|---|---|
| Graph orchestration | EXISTING BUT NEEDS REFACTOR | Staged graph is real; data partition is unsafe |
| Typed agent input/output | EXISTING BUT NEEDS REFACTOR | Pydantic contracts exist; graph state also carries opaque DataFrames/compiled objects |
| Structured LLM strategy generation | EXISTING & REUSABLE | Strict proposal envelope and canonical strategy promotion |
| Regime classification | EXISTING BUT NEEDS REFACTOR | Causal features/fallback exist; same-sample use is invalid |
| Bounded retry handling | EXISTING & REUSABLE | Generation attempts are capped and tested |
| Validation gates | EXISTING & REUSABLE | Invalid strategies retry/terminate before compiler |
| Compiler gates | EXISTING & REUSABLE | Compiler failure terminates before backtest |
| Error propagation | PARTIAL | Typed statuses/events exist; broad catches and fallback can flatten error semantics |
| Agent metadata logging | PARTIAL | Prompt/model labels and events exist; IDs/data fingerprints/persistence absent |
| Model abstraction | EXISTING BUT NEEDS REFACTOR | Structured protocol exists but is not composed with the legacy factory |
| Strategy-generation contracts | EXISTING & REUSABLE | Proposal draft and promoted proposal are distinct |

**Minimum Sprint 4 completion:** first accept separate, non-overlapping context and evaluation datasets; then add a real structured-client composition root, canonical run/strategy identifiers, dataset/config provenance, enforced provider settings, and one offline recorded-provider plus one opt-in live smoke test. Do not add more agents.

### Sprint 5 capability classification

| Capability | State |
|---|---|
| Average win/loss | EXISTS, but directional largest-value semantics need correction |
| Calmar / recovery / payoff / turnover | MISSING |
| Minimum trades / viability gates | MISSING |
| Strategy comparison / ranking | MISSING |
| Scoring and score decomposition | MISSING |
| Deterministic performance analyst input | PARTIAL: `BacktestResult.to_dict()` exists; analyst contract does not |

### Sprint 6 capability classification

| Capability | State |
|---|---|
| Max loss per trade | PARTIAL: stop exists; gap loss is unbounded and no capital-risk budget exists |
| Max exposure / fixed position sizing | PARTIAL: one asset and fixed percent only |
| Volatility/ATR sizing | MISSING |
| Concurrent positions | PARTIAL by hard-coded single position, not a configurable portfolio constraint |
| Liquidity / concentration / portfolio risk | MISSING |
| Drawdown controls | MISSING |
| Refinement, bounded iterations, lineage | MISSING; initial generation retry is not refinement |
| Overfit rejection | MISSING |

### Sprint 7 capability classification

All requested capabilities are **MISSING** except causal rolling calculations and prefix perturbation tests. The current one-frame graph is an active optimization/test contamination risk. There is no safe final holdout.

### Sprint 8 capability classification

Structured logging, model labels, prompt versions, result objects, environment-based secrets, and legacy Markdown output are **PARTIAL** foundations. Run/strategy IDs, schema/data/execution versions, deterministic seed enforcement, lineage, persistent new-run storage, HTML/PDF, new CLI/API, and hardened end-to-end tests are **MISSING**.

## 18. Recommended Roadmap

The original Sprint 4–8 sequence should not remain unchanged. Refinement before a protected OOS protocol would create an automated overfitting loop, and ranking before metric repair would formalize inconsistent scores.

1. **Sprint 3.5 — Correctness & Research Protocol Hardening.** Fix P0/P1 deterministic defects, finite/deep-immutable contracts, metric semantics, canonical validator use, hermetic tests, and Git/package reproducibility.
2. **Sprint 4 — Complete Safe Orchestration.** Introduce separate context and validation frames, a decision timestamp, production structured-client/provider composition, IDs, and provenance. Keep legacy prose agents outside the authoritative path.
3. **Sprint 5A — Deterministic Evaluation Primitives.** Add versioned metrics, minimum-trade/viability gates, and transparent score components without LLM judgment.
4. **Sprint 7 — OOS and Walk-Forward Protocol.** Implement chronological splits, untouched holdout, sensitivity, seeded Monte Carlo/trade skipping, and split-access controls.
5. **Sprint 5B — Candidate Comparison/Ranking.** Rank on training/validation outputs only. Final holdout remains unavailable until a selection is frozen.
6. **Sprint 6 — Risk Enforcement, then Refinement.** Build deterministic sizing/portfolio constraints first. Add bounded LLM refinement only with lineage and validation/OOS degradation rejection.
7. **Sprint 8 — Persistence, Reporting, CLI/API, Observability, CI.** Harden the stable contracts and produce faculty/research artifacts.

**Recommended Next Sprint: Sprint 3.5 — Correctness & Research Protocol Hardening.** It is the shortest dependency-correct move: every later score, agent decision, OOS result, and report depends on trustworthy fills, metrics, immutable inputs, and data boundaries.

## 19. Technical Debt

### Architecture and contracts

- Two separate graphs and result models with no explicit legacy quarantine boundary.
- Public unsafe construction/bypass paths.
- Generic `data` top-level package can collide with other packages.
- Mutable dataclasses despite “immutable” documentation.
- Pydantic models use coercion and incomplete finite constraints.
- No schema/indicator/operator/execution version model.

### Quant/data

- Same-frame context/evaluation.
- Execution-bar ATR leak and raw-vs-fill risk basis.
- Duplicate indicator computations and definitions.
- No point-in-time fundamental/news contract.
- Metrics mix gross and net economics.
- Exposure/holding semantics are not research-grade.
- No benchmark, calendar, corporate-action, survivorship, delisting, or universe contract.

### Agents and safety

- Legacy prose is recursively trusted as prompt input.
- Final rating has no enum validation.
- Tool dates are chosen by the model rather than clamped by the system.
- Structured adapter does not enforce recorded seed.
- Error strings are truncated but not secret-redacted.

### Testing and operations

- New implementation/tests are untracked.
- Root test performs collection-time network access.
- No coverage/lint/type/CI gates.
- No live or recorded external provider/model contract suite.
- No scale/performance benchmarks.
- Local build and egg metadata are stale.
- New research result has no durable store or report renderer.

## 20. Final Readiness Assessment

```text
Strategy Schema Layer: MOSTLY READY
Strategy Compiler: MOSTLY READY
Market Data Layer: PARTIAL
Backtesting Engine: PARTIAL
Metrics Layer: PARTIAL
Agent Architecture: PARTIAL
Risk Layer: PARTIAL
OOS Validation: NOT READY
Observability: PARTIAL
Reporting: PARTIAL
End-to-End Integration: PARTIAL
```

**Overall Project Completion: approximately 45%.**

This estimate is dependency-weighted rather than sprint-counted:

| Capability group | Weight | Estimated maturity | Weighted contribution |
|---|---:|---:|---:|
| Strategy schema | 10% | 80% | 8.0% |
| Compiler and indicators | 12% | 80% | 9.6% |
| Market data | 8% | 65% | 5.2% |
| Backtesting/execution | 15% | 60% | 9.0% |
| Metrics/evaluation/ranking | 10% | 30% | 3.0% |
| Agents/orchestration | 15% | 45% | 6.75% |
| Risk/refinement | 10% | 15% | 1.5% |
| OOS validation | 10% | 0% | 0% |
| Observability/reporting/persistence/CLI | 10% | 20% | 2.0% |
| **Total** | **100%** |  | **45.05% ≈ 45%** |

```text
Current Stage: Integrated deterministic/structured-agent prototype
Biggest Strength: Explicit allow-listed strategy compilation with causal indicator signals and next-bar execution
Biggest Technical Risk: Evaluation-sample leakage into strategy generation, compounded by execution-bar ATR leakage
Most Important Missing Component: Enforceable chronological research/OOS data protocol with an untouched holdout
Recommended Next Sprint: Sprint 3.5 — Correctness & Research Protocol Hardening
```

## 21. Recommended Next Step

Freeze feature expansion and create one reviewed correction branch/change set containing only:

1. A typed, non-overlapping context/train/validation/test data contract with decision timestamps.
2. ATR stop computation from information available before the execution Open.
3. Fill-price-based stop/target construction.
4. Finite, strict, deeply immutable strategy/backtest models and corrected inclusive-bound logic.
5. Versioned, internally consistent net-of-cost metric definitions.
6. Regression tests for every item above, including holdout access denial and execution-bar perturbation.
7. Removal or isolation of collection-time network code.
8. A coherent Git commit and package-build verification for all new modules/tests.

Only after those acceptance criteria pass should Sprint 4 orchestration be called complete or Sprint 5 ranking begin.
