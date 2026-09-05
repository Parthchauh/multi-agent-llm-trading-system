"""
tests/conftest.py
=================
pytest configuration: marker registration and collection guards.
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: marks tests that require live network access (deselect with -m 'not live')",
    )
