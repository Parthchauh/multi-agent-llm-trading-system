"""Durable holdout exposure audit shared across strategy versions and campaigns.

An exposure is recorded before data is released, including failed evaluations.
Overlapping windows for the same dataset family are previously exposed evidence.
Deleting or changing the audit database cannot undo human knowledge; preserve
this database with research artifacts and use fresh data for new confirmation.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class HoldoutExposureLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS exposures (
                id INTEGER PRIMARY KEY, dataset_family TEXT NOT NULL,
                campaign_id TEXT NOT NULL, strategy_hash TEXT NOT NULL,
                data_hash TEXT NOT NULL, start TEXT NOT NULL, end TEXT NOT NULL,
                exposed_at TEXT NOT NULL, classification TEXT NOT NULL)""")

    def claim(self, *, dataset_family, campaign_id, strategy_hash, fingerprint):
        if not all((dataset_family, campaign_id, strategy_hash)):
            raise ValueError("Exposure requires dataset, campaign, and frozen strategy identity.")

        def utc_text(stamp):
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            return stamp.astimezone(timezone.utc).isoformat()

        start = utc_text(fingerprint.start)
        end = utc_text(fingerprint.end)
        # Canonical families are SYMBOL:provider. A provider adapter rename
        # must not make the same symbol/date window unseen. Include historical
        # rows without rewriting their original attribution.
        symbol_prefix = dataset_family.split(":", 1)[0] + ":"
        with sqlite3.connect(self.path, timeout=30.0) as connection:
            connection.execute("BEGIN IMMEDIATE")
            seen = connection.execute(
                "SELECT id FROM exposures WHERE (dataset_family=? OR substr(dataset_family,1,?)=?) AND start<=? AND end>=? LIMIT 1",
                (dataset_family, len(symbol_prefix), symbol_prefix, end, start),
            ).fetchone()
            classification = "PREVIOUSLY_EXPOSED_EXPLORATORY" if seen else "UNSEEN_CONFIRMATORY"
            connection.execute(
                "INSERT INTO exposures(dataset_family,campaign_id,strategy_hash,data_hash,start,end,exposed_at,classification) VALUES(?,?,?,?,?,?,?,?)",
                (
                    dataset_family,
                    campaign_id,
                    strategy_hash,
                    fingerprint.data_sha256,
                    start,
                    end,
                    datetime.now(timezone.utc).isoformat(),
                    classification,
                ),
            )
        return classification
