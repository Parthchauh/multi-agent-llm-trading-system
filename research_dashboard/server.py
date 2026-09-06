"""Local, read-only research dashboard with no external web-framework dependency.

The dashboard reads only canonical reports already written by
``SQLiteResearchRepository``.  It has no order, broker, LLM, strategy
generation, or backtest-execution endpoints; those controls remain in the
authoritative research pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from evaluation.platform import (
    ExperimentReport,
    ResearchPlatformError,
    SQLiteResearchRepository,
    report_markdown,
)


class ResearchDashboardServer(ThreadingHTTPServer):
    """Threading HTTP server that owns only a report repository reference."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        repository: SQLiteResearchRepository,
    ) -> None:
        self.repository = repository
        super().__init__(server_address, handler_class)


class _DashboardHandler(BaseHTTPRequestHandler):
    """Serve local report data and a compact browser interface."""

    server: ResearchDashboardServer

    def do_GET(self) -> None:  # noqa: N802 - required stdlib handler name.
        path = urlsplit(self.path).path
        if path == "/" or path == "/index.html":
            self._send_html(_DASHBOARD_HTML)
            return
        if path == "/health":
            self._send_json({"status": "ok", "service": "research-dashboard"})
            return
        if path == "/api/experiments":
            self._send_json([_report_summary(report) for report in self.server.repository.list_reports()])
            return
        if path.startswith("/api/experiments/"):
            experiment_id = unquote(path.removeprefix("/api/experiments/"))
            self._send_report_json(experiment_id)
            return
        if path.startswith("/reports/") and path.endswith(".md"):
            experiment_id = unquote(path.removeprefix("/reports/")[:-3])
            self._send_report_markdown(experiment_id)
            return
        self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - required stdlib handler name.
        self._send_json(
            {
                "error": "read_only_dashboard",
                "detail": "Use the authoritative research runner to create experiments.",
            },
            status=HTTPStatus.METHOD_NOT_ALLOWED,
        )

    def _send_report_json(self, experiment_id: str) -> None:
        try:
            report = self.server.repository.load_report(experiment_id)
        except ResearchPlatformError:
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json(report.model_dump(mode="json"))

    def _send_report_markdown(self, experiment_id: str) -> None:
        try:
            report = self.server.repository.load_report(experiment_id)
        except ResearchPlatformError:
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        body = report_markdown(report).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, page: str) -> None:
        body = page.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Keep a local dashboard quiet; audit events already live in SQLite."""


def create_dashboard_server(
    database: str | Path = "research.sqlite",
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> ResearchDashboardServer:
    """Create a local-only report dashboard server without starting it."""

    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("The research dashboard may listen only on a loopback address.")
    if not 0 <= port <= 65_535:
        raise ValueError("port must be between 0 and 65535.")
    repository = SQLiteResearchRepository(database)
    return ResearchDashboardServer((host, port), _DashboardHandler, repository=repository)


def serve_dashboard(
    database: str | Path = "research.sqlite",
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    announce: Callable[[str], None] = print,
) -> None:
    """Run the local read-only dashboard until the user interrupts it."""

    server = create_dashboard_server(database, host=host, port=port)
    address, resolved_port = server.server_address[:2]
    announce(f"Research dashboard: http://{address}:{resolved_port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _report_summary(report: ExperimentReport) -> dict[str, Any]:
    completed = sum(run.status == "COMPLETED" for run in report.runs)
    experiment_id = report.experiment_config.experiment_id
    return {
        "experiment_id": experiment_id,
        "architecture": report.experiment_config.architecture.value,
        "dataset_id": report.experiment_config.dataset_id,
        "symbols": list(report.experiment_config.symbols),
        "runs": len(report.runs),
        "completed_runs": completed,
        "failed_runs": len(report.failures),
        "report_json_url": f"/api/experiments/{quote(experiment_id, safe='')}",
        "report_markdown_url": f"/reports/{quote(experiment_id, safe='')}.md",
    }


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trading Strategy Research Dashboard</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { background: #0b1020; color: #e7edf8; margin: 0; padding: 2rem; }
    main { max-width: 1000px; margin: auto; }
    h1 { margin-bottom: .3rem; } p { color: #b9c6db; }
    table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; }
    th, td { text-align: left; padding: .75rem; border-bottom: 1px solid #27324a; }
    th { color: #8fb5ff; } a { color: #82d2ff; }
    .notice { border-left: 4px solid #65c89a; padding: .8rem 1rem; background: #121b30; }
  </style>
</head>
<body><main>
  <h1>Research Dashboard</h1>
  <p>Read-only local view of canonical persisted experiment reports.</p>
  <p class="notice">LLM interpretations are not execution authority. All shown results originate from validated deterministic research runs.</p>
  <table><thead><tr><th>Experiment</th><th>Architecture</th><th>Dataset</th><th>Runs</th><th>Failures</th><th>Reports</th></tr></thead>
  <tbody id="experiments"><tr><td colspan="6">Loading reports…</td></tr></tbody></table>
  <script>
    const cell = (value) => { const node = document.createElement('td'); node.textContent = value; return node; };
    fetch('/api/experiments').then(response => response.json()).then(reports => {
      const body = document.getElementById('experiments'); body.replaceChildren();
      if (!reports.length) { const row = document.createElement('tr'); const empty = cell('No persisted experiment reports yet.'); empty.colSpan = 6; row.append(empty); body.append(row); return; }
      reports.forEach(report => { const row = document.createElement('tr'); row.append(cell(report.experiment_id), cell(report.architecture), cell(report.dataset_id), cell(`${report.completed_runs}/${report.runs}`), cell(report.failed_runs)); const links = document.createElement('td'); const json = document.createElement('a'); json.href = report.report_json_url; json.textContent = 'JSON'; const markdown = document.createElement('a'); markdown.href = report.report_markdown_url; markdown.textContent = 'Markdown'; links.append(json, document.createTextNode(' · '), markdown); row.append(links); body.append(row); });
    }).catch(() => { document.getElementById('experiments').innerHTML = '<tr><td colspan="6">Unable to load reports.</td></tr>'; });
  </script>
</main></body></html>"""


__all__ = ["ResearchDashboardServer", "create_dashboard_server", "serve_dashboard"]
