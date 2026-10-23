"""TLS-only, loopback-bound authenticated evidence API. No model/job execution."""
from __future__ import annotations

import json
import re
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from evaluation.platform import ResearchPlatformError, SQLiteResearchRepository
from research_service.access import AccessStore


def handler(access: AccessStore, repository: SQLiteResearchRepository):
    class ResearchHandler(BaseHTTPRequestHandler):
        server_version = 'ResearchEvidence'
        sys_version = ''

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass  # Access audit contains decisions, never credentials, tokens or bodies.

        def _reply(self, status, body):
            payload = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Strict-Transport-Security', 'max-age=31536000')
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if self.path != '/v1/session':
                self._reply(404, {'error': 'NOT_FOUND'})
                return
            try:
                access.rate_limit('login-ip:'+self.client_address[0], maximum=30)
                size = int(self.headers.get('Content-Length', '0'))
                if (not 1 <= size <= 4096 or self.headers.get('Transfer-Encoding')
                        or len(self.headers.get_all('Content-Length', [])) != 1):
                    raise ValueError('Invalid request length.')
                body = json.loads(self.rfile.read(size))
                if set(body) != {'tenant', 'username', 'password'}:
                    raise ValueError('Unexpected login fields.')
                token = access.login(**body)
                self._reply(200, {'access_token': token, 'token_type': 'Bearer', 'expires_in': 900})  # nosec B105 - OAuth scheme name, not a credential.
            except PermissionError as exc:
                self._reply(429 if str(exc) == 'RATE_LIMITED' else 401, {'error': 'ACCESS_DENIED'})
            except (ValueError, TypeError):
                self._reply(400, {'error': 'INVALID_REQUEST'})
            except TimeoutError:
                self._reply(408, {'error': 'REQUEST_TIMEOUT'})

        def do_GET(self):
            match = re.fullmatch(r'/v1/experiments/([A-Za-z0-9_.:-]{1,160})', self.path)
            if not match:
                self._reply(404, {'error': 'NOT_FOUND'})
                return
            try:
                scheme, _, token = self.headers.get('Authorization', '').partition(' ')
                if scheme != 'Bearer':
                    raise PermissionError('SESSION_REQUIRED')
                access.authorize(token, match[1])
                report = repository.load_report(match[1])
                self._reply(200, report.model_dump(mode='json'))
            except PermissionError as exc:
                self._reply(429 if str(exc) == 'RATE_LIMITED' else 403, {'error': 'ACCESS_DENIED'})
            except (KeyError, ValueError, ResearchPlatformError):
                self._reply(404, {'error': 'NOT_FOUND'})

        def do_DELETE(self):
            scheme, _, token = self.headers.get('Authorization', '').partition(' ')
            if self.path != '/v1/session' or scheme != 'Bearer':
                self._reply(404, {'error': 'NOT_FOUND'})
                return
            try:
                access.authenticate(token)
                access.revoke(token)
                self._reply(200, {'status': 'SESSION_REVOKED'})
            except PermissionError:
                self._reply(403, {'error': 'ACCESS_DENIED'})
    return ResearchHandler


def serve(*, access_database: Path, research_database: Path, certificate: Path,
          private_key: Path, port=8443):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certificate, private_key)
    server = ThreadingHTTPServer(('127.0.0.1', port),
        handler(AccessStore(access_database), SQLiteResearchRepository(research_database)))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
