"""Read-only local dashboard for persisted quantitative-research reports."""

from research_dashboard.server import create_dashboard_server, serve_dashboard

__all__ = ["create_dashboard_server", "serve_dashboard"]
