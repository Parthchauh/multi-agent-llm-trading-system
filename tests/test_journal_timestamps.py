from datetime import datetime, timedelta, timezone

import pytest

from paper_trading.journal import PaperJournal


@pytest.mark.parametrize('field', ['timestamp', 'signal_time', 'fill_time'])
def test_utc_json_timestamps_are_supported_on_every_declared_python_version(field):
    journal = PaperJournal()
    journal.record_event('FIXTURE', {field: '2026-10-23T12:34:56Z'})
    assert journal.entries[0].timestamp == datetime(2026, 10, 23, 12, 34, 56, tzinfo=timezone.utc)
    assert journal.entries[0].data[field].endswith('Z')


def test_journal_preserves_explicit_offset():
    journal = PaperJournal()
    journal.record_event('FIXTURE', {'timestamp': '2026-10-23T18:04:56+05:30'})
    assert journal.entries[0].timestamp.utcoffset() == timedelta(hours=5, minutes=30)
    assert journal.entries[0].timestamp == datetime(2026, 10, 23, 12, 34, 56, tzinfo=timezone.utc)
