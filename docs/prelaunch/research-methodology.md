# Research diagnostics and their limits

These tools generate development evidence. They do not grant statistical
certification, paper eligibility, investment recommendations or execution rights.
The canonical certification boundary remains disabled. Historical baseline gaps
in baseline-results.md remain unchanged.

## Chronological campaigns

`research_validation.campaign.FoldResearchCampaign` accepts DEVELOPMENT data only.
Each fold generates candidates from an isolated copy of training bars, ranks
semantically valid candidates on validation total return, breaks ties by strategy
SHA256, freezes the complete strategy/config/risk identity, and evaluates the
winner on the next test window. Test windows cannot overlap. Rolling and
expanding history are supported. Later folds may train on earlier development
test windows; these windows are not a reusable final confirmatory holdout.

The candidate factory is trusted application code, not an arbitrary-code sandbox.
`TrainOnlyAgentFactory` derives features exclusively from its supplied training
frame and uses the existing structured agents with one generation request per
candidate attempt. Schema failures, duplicate outputs and provider failures all
consume budget. Provider failures are distinct from rejected schemas. The
recorded-response CLI makes no live provider calls. A planned global budget
overflow fails before invoking the factory.

`WalkForwardEvaluator` remains a FIXED_STRATEGY_DIAGNOSTIC and discloses overlapping
validation windows. It cannot establish fold-specific candidate generation.

Campaign reports disclose every attempt, selected frozen identity, full fold
metrics, validation/test distributions, risk assessments, cost multipliers
1/1.5/2 and start offsets 0/5/10 bars. All sensitivity runs retain the same
canonical execution kernel and past-only feature warm-up. Existing Sprint 7
parameter, delay, regime and subperiod diagnostics remain separate components.

## Uncertainty

PSR uses nonannualized Sharpe and reference Sharpe at the same bar frequency:
`Phi((SR-SR_reference)*sqrt(T-1)/sqrt(1-skew*SR+(kurtosis-1)*SR^2/4))`.
Kurtosis is Pearson kurtosis, not excess kurtosis. Fewer than 20 observations or
zero variation returns INSUFFICIENT_EVIDENCE. Serial dependence and moment
estimation uncertainty can invalidate the approximation; lag-one autocorrelation
is reported, and no PSR threshold automatically approves a strategy.

DSR is unavailable unless the caller supplies comparable trial Sharpe estimates
and a justified independent-trial count. Candidate count is not silently treated
as independent search count. Its expected-null-maximum benchmark uses trial
dispersion and the normal-order-statistic approximation described in
[Bailey and Lopez de Prado, The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).
The underlying PSR assumptions are described in their
[Sharpe Ratio Efficient Frontier paper](https://www.davidhbailey.com/dhbpapers/sharpe-frontier.pdf).

The seeded circular block bootstrap reports 95% quantile intervals for mean bar
return, compounded return and maximum drawdown. Blocks retain dependence within
a block, not between blocks. The default block length is five observations;
methodology must be specified before interpreting results. Completed-trade
bootstrap and permutation hold net trade P&L fixed. They do not reconstruct market
paths, sizing feedback, liquidity or future execution. Fewer than 20 completed
trades is insufficient evidence.

## Data and comparison boundaries

Saved snapshots bind provider/version, query, retrieval time, adjustment basis,
timezone convention, supplied session coverage and normalized values. Hashes
detect changes; they do not prove vendor claims. Corporate-action normalization
uses supplied price and volume factors with availability timestamps. It refuses
future-unavailable factors. Rebuild factors at each research cutoff; applying
today's adjusted history to all past folds can introduce hindsight. Exchange
calendars, point-in-time universes, delistings and data redistribution permissions
need independently reviewed sources.

Architecture comparisons use validation evidence only, require matched
data/model/seed/cost/split conditions and actual fingerprints, and retain failed
runs and rejections. Missing usage/prices remain null. A descriptive improvement
does not establish a statistically significant agent benefit. The canonical CLI
executes all five architecture treatments. They share regime preprocessing and
one initial strategy generator. No-debate adds performance/critic review; debate
adds bounded Bull/Bear/synthesis; risk adds an explanatory risk agent; full adds
bounded refinement. Deterministic risk enforcement remains enabled in every
treatment. Review-only treatments do not fabricate revised strategies or claim
performance improvement merely because more agents ran. Use --development-only
for architecture studies so confirmatory test and final holdout stay sealed.

Paper drift uses frozen reference evidence and predeclared persistent thresholds.
Low observation/trade counts never produce a drift pass. Simulated historical
bars do not satisfy elapsed real paper observation requirements.
