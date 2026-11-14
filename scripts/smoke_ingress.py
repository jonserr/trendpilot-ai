"""Exercise rendered Traefik policies in disposable local containers, without Kubernetes.

Requires docker, kubectl, openssl, and an existing API image (for PyYAML and bcrypt).
Uses Traefik's file provider; Kubernetes Secret/CRD reconciliation still needs a
cluster integration test. No production credentials or running services are used.
"""

import argparse
import base64
import http.client
import json
from pathlib import Path
import socket
import ssl
import subprocess
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def command(*args, input=None):
    return subprocess.run(
        args, input=input, capture_output=True, text=True, check=True
    ).stdout.strip()


def request(port, host, path="/", auth=None, secure=True, cert=None, websocket=False, verify=True):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    if secure:
        context = (
            ssl.create_default_context(cafile=str(cert))
            if verify
            else ssl._create_unverified_context()
        )
        connection.sock = context.wrap_socket(
            socket.create_connection(("127.0.0.1", port), timeout=3), server_hostname=host
        )
    headers = {"Host": host}
    if auth:
        headers["Authorization"] = "Basic " + base64.b64encode(auth.encode()).decode()
    if websocket:
        headers.update(
            {
                "Connection": "Upgrade",
                "Upgrade": "websocket",
                "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
                "Sec-WebSocket-Version": "13",
            }
        )
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return (
            response.status,
            {name.lower(): value for name, value in response.getheaders()},
            response.read(),
        )
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-image", default="trendpilot-ai-api:latest")
    parser.add_argument("--traefik-image", default="traefik:v3.7.15")
    args = parser.parse_args()
    rendered = command("kubectl", "kustomize", str(ROOT / "deploy/k8s/overlays/prod"))
    resources = json.loads(
        command(
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            args.app_image,
            "python",
            "-c",
            "import json,sys,yaml; print(json.dumps(list(yaml.safe_load_all(sys.stdin))))",
            input=rendered,
        )
    )
    # Only disposable test credentials, never the user's API key or dashboard password.
    user_hash = command(
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        args.app_image,
        "python",
        "-c",
        "import crypt; print('smoke:' + crypt.crypt('smoke-password', crypt.mksalt(crypt.METHOD_BLOWFISH)))",
    )
    prefix = "trendpilot-ingress-" + uuid.uuid4().hex[:10]
    network, backend, proxy = prefix, prefix + "-backend", prefix + "-proxy"
    containers = []
    checks = ROOT / ".local-checks"
    checks.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ingress-", dir=checks) as folder:
        directory = Path(folder)
        command(
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(directory / "test.key"),
            "-out",
            str(directory / "test.crt"),
            "-subj",
            "/CN=trendpilot.example.com",
            "-addext",
            "subjectAltName=DNS:trendpilot.example.com,DNS:api.trendpilot.example.com",
        )
        (directory / "users").write_text(user_hash + "\n")
        (directory / "backend.py").write_text("""
from http.server import BaseHTTPRequestHandler, HTTPServer
import base64, hashlib, json
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get("Upgrade", "").lower() == "websocket":
            self.send_response(101)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            accept = hashlib.sha1((self.headers["Sec-WebSocket-Key"] + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
            self.send_header("Sec-WebSocket-Accept", base64.b64encode(accept).decode())
            self.end_headers()
            self.close_connection = True
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"authorization_seen": bool(self.headers.get("Authorization"))}).encode())
HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
""")
        config = {
            "http": {"routers": {}, "services": {}, "middlewares": {}},
            "tls": {
                "certificates": [
                    {"certFile": "/fixtures/test.crt", "keyFile": "/fixtures/test.key"}
                ],
                "options": {},
            },
        }
        for obj in resources:
            kind, name, spec = obj["kind"], obj["metadata"]["name"], obj.get("spec", {})
            if kind == "Middleware":
                middleware = json.loads(json.dumps(spec))
                if "basicAuth" in middleware:
                    middleware["basicAuth"].pop("secret")
                    middleware["basicAuth"]["usersFile"] = "/fixtures/users"
                config["http"]["middlewares"][name] = middleware
            elif kind == "TLSOption":
                config["tls"]["options"][name] = spec
            elif kind == "Ingress":
                annotations = obj["metadata"]["annotations"]
                router = {
                    "rule": " || ".join(f"Host(`{rule['host']}`)" for rule in spec["rules"]),
                    "entryPoints": annotations[
                        "traefik.ingress.kubernetes.io/router.entrypoints"
                    ].split(","),
                    "service": "backend",
                }
                if annotations.get("traefik.ingress.kubernetes.io/router.tls") == "true":
                    ref = annotations["traefik.ingress.kubernetes.io/router.tls.options"]
                    router["tls"] = {"options": ref.removeprefix("trendpilot-").split("@")[0]}
                references = annotations.get("traefik.ingress.kubernetes.io/router.middlewares", "")
                if references:
                    router["middlewares"] = [
                        ref.removeprefix("trendpilot-").split("@")[0]
                        for ref in references.split(",")
                    ]
                config["http"]["routers"][name] = router
        config["http"]["services"]["backend"] = {
            "loadBalancer": {"servers": [{"url": "http://backend:8000"}]}
        }
        try:
            command("docker", "network", "create", network)
            containers.append(backend)
            command(
                "docker",
                "run",
                "-d",
                "--name",
                backend,
                "--network",
                network,
                "--network-alias",
                "backend",
                "--mount",
                f"type=bind,src={directory},dst=/fixtures,readonly",
                args.app_image,
                "python",
                "/fixtures/backend.py",
            )
            for case in ("valid", "missing-secret", "missing-middleware", "missing-certificate"):
                candidate = json.loads(json.dumps(config))
                if case == "missing-secret":
                    candidate["http"]["middlewares"]["dashboard-auth"]["basicAuth"][
                        "usersFile"
                    ] = "/fixtures/missing-users"
                elif case == "missing-middleware":
                    candidate["http"]["middlewares"].pop("dashboard-auth")
                elif case == "missing-certificate":
                    candidate["tls"]["certificates"] = []
                (directory / "dynamic.yaml").write_text(json.dumps(candidate))
                containers.append(proxy)
                command(
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    proxy,
                    "--network",
                    network,
                    "-p",
                    "127.0.0.1::8000",
                    "-p",
                    "127.0.0.1::8443",
                    "--mount",
                    f"type=bind,src={directory},dst=/fixtures,readonly",
                    args.traefik_image,
                    "--providers.file.filename=/fixtures/dynamic.yaml",
                    "--entrypoints.web.address=:8000",
                    "--entrypoints.websecure.address=:8443",
                )
                http_port = int(command("docker", "port", proxy, "8000/tcp").rsplit(":", 1)[1])
                https_port = int(command("docker", "port", proxy, "8443/tcp").rsplit(":", 1)[1])
                for attempt in range(50):
                    try:
                        status, _, _ = request(http_port, "trendpilot.example.com", secure=False)
                        if status in (301, 302, 307, 308):
                            break
                    except (OSError, http.client.HTTPException):
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError(f"Traefik did not start for {case}")
                if case == "missing-certificate":
                    try:
                        request(
                            https_port,
                            "trendpilot.example.com",
                            cert=directory / "test.crt",
                            verify=False,
                        )
                    except ssl.SSLError:
                        pass
                    else:
                        raise AssertionError("Missing TLS certificate was accepted")
                elif case != "valid":
                    status, _, _ = request(
                        https_port,
                        "trendpilot.example.com",
                        auth="smoke:smoke-password",
                        cert=directory / "test.crt",
                    )
                    assert status in (401, 404, 503), (case, status)
                else:
                    for host in ("trendpilot.example.com", "api.trendpilot.example.com"):
                        status, headers, _ = request(http_port, host, path="/check", secure=False)
                        assert (
                            status in (301, 308) and headers["location"] == f"https://{host}/check"
                        )
                    for auth in (None, "smoke:wrong"):
                        assert (
                            request(
                                https_port,
                                "trendpilot.example.com",
                                auth=auth,
                                cert=directory / "test.crt",
                            )[0]
                            == 401
                        )
                    status, _, body = request(
                        https_port,
                        "trendpilot.example.com",
                        auth="smoke:smoke-password",
                        cert=directory / "test.crt",
                    )
                    assert status == 200 and json.loads(body)["authorization_seen"] is False
                    status, headers, _ = request(
                        https_port,
                        "trendpilot.example.com",
                        auth="smoke:smoke-password",
                        cert=directory / "test.crt",
                        websocket=True,
                    )
                    assert (
                        status == 101
                        and headers["sec-websocket-accept"] == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
                    )
                print(f"Traefik {case}: passed", flush=True)
                command("docker", "rm", "-f", proxy)
                containers.remove(proxy)
        except Exception as exc:
            logs = subprocess.run(["docker", "logs", proxy], capture_output=True, text=True)
            raise RuntimeError(f"Ingress smoke failed: {logs.stdout}\n{logs.stderr}") from exc
        finally:
            for container in reversed(containers):
                subprocess.run(["docker", "rm", "-f", container], capture_output=True)
            subprocess.run(["docker", "network", "rm", network], capture_output=True)


if __name__ == "__main__":
    main()
