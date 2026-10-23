# Authoritative execution semantics

The engine is long-only and holds at most one position for one symbol. A signal
evaluated on a bar close executes at the next bar open. Strategy feature windows
contain no later bars. Stop and target levels are frozen using the entry fill
and preceding-bar context, including ATR from that context.

Each bar executes a pending signal, then checks maximum calendar holding days,
then gap and intrabar protective exits. A new position's protections are active
on its entry bar after the opening fill. A gap below the stop sells at the open;
a gap above the target sells at the open. Otherwise an intrabar stop wins when
both levels are touched. Pending signal exits precede protective exits.

Whole-share sizing floors allocated cash divided by slipped price including
commission. An unaffordable single share produces no fill. No leverage, shorts,
partial fills, or partial exits are supported. Duplicate fill IDs, oversells,
and invalid exits are rejected without mutating the paper ledger.

Slippage is incorporated in execution price exactly once. Cash changes by
executed notional plus commission for buys, and executed notional minus commission
for sells. Slippage attribution is informational. Round-trip realized P&L
includes both entry and exit commissions. Equity is cash plus marked holdings.
Binary floating point is retained; execution values are not rounded before ledger
updates. Display snapshots round to four decimals; ledger comparisons use 1e-8
absolute tolerance or pytest's numeric tolerance for accumulated results.

Backtest EOF closes an open position at the final close with sell costs. Paper
streams have no implicit EOF; explicit finalize() gives the same accounting
close for replay comparisons and disallows subsequent bars. The historical
adapter ignores the final bar's signal because no next open exists.

Paper history must append one unrevised bar per call; duplicate timestamps and
changed history are rejected. The first call may include indicator warm-up.
Checkpoints atomically save replay inputs and immutable strategy/configuration.
Restore reconstructs the ledger and verifies cash and every fill before use.
This is local replay recovery, not a claim of broker-order idempotency.
