import ipaddress
import json
import ssl
import threading
from datetime import datetime, timedelta, timezone
from http.client import HTTPSConnection
from http.server import ThreadingHTTPServer

import pytest

from research_service.access import AccessStore
from research_service.server import handler


def test_real_tls_login_tenant_boundaries_malformed_requests_and_revocation(tmp_path):
    x509 = pytest.importorskip('cryptography.x509')
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), critical=False)
        .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path/'test-cert.pem', tmp_path/'test-key.pem'
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption()))
    access = AccessStore(tmp_path/'access.sqlite')
    password = 'tls-test-password-123456'
    access.create_user('alpha', 'reader', password, 'read_only')
    access.claim_resource('alpha-evidence', 'alpha')
    access.claim_resource('beta-evidence', 'beta')

    class Report:
        def model_dump(self, **kwargs):
            return {'status': 'RESEARCH_REJECTION', 'paper_eligible': False}
    class Repository:
        def __init__(self):
            self.reads = []
        def load_report(self, identity):
            self.reads.append(identity)
            return Report()
    repository = Repository()
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(access, repository))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(method, path, payload=None, token=None):
        connection = HTTPSConnection('127.0.0.1', server.server_port,
            context=ssl.create_default_context(cafile=str(cert_path)), timeout=10)
        headers = {'Authorization': 'Bearer '+token} if token else {}
        connection.request(method, path, body=json.dumps(payload) if payload is not None else None, headers=headers)
        response = connection.getresponse()
        body = json.loads(response.read())
        assert response.getheader('Cache-Control') == 'no-store'
        connection.close()
        return response.status, body
    try:
        assert request('GET', '/v1/experiments/alpha-evidence')[0] == 403
        assert request('POST', '/v1/session', {'wrong': 'fields'})[0] == 400
        assert request('POST', '/v1/session', {'tenant': 'alpha', 'username': 'reader', 'password': 'wrong'})[0] == 401
        status, login = request('POST', '/v1/session', {'tenant': 'alpha', 'username': 'reader', 'password': password})
        assert status == 200
        token = login['access_token']
        assert request('GET', '/v1/experiments/alpha-evidence', token=token)[0] == 200
        assert request('GET', '/v1/experiments/beta-evidence', token=token)[0] == 403
        assert request('GET', '/v1/experiments/unknown', token=token)[0] == 403
        assert request('GET', '/v1/experiments/../secret', token=token)[0] == 404
        assert repository.reads == ['alpha-evidence']
        assert request('DELETE', '/v1/session', token=token)[0] == 200
        assert request('GET', '/v1/experiments/alpha-evidence', token=token)[0] == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
