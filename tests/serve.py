"""A tiny local HTTP server for the fetcher tests: Range requests, HEAD, injected failures.

Every request is recorded on `server.requests` as (method, path, range_header). Any
path is served from `directory / <basename of the path>`, query string ignored, so a
cutout-style URL with parameters maps to one fixture file.
"""

from __future__ import annotations

import threading
import urllib.parse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class FixtureServer(ThreadingHTTPServer):
    directory: Path
    requests: list
    fail_queue: list
    support_range: bool


class _Handler(BaseHTTPRequestHandler):
    server: FixtureServer

    def log_message(self, *args) -> None:      # keep pytest output clean
        pass

    def do_HEAD(self) -> None:
        self._serve(head=True)

    def do_GET(self) -> None:
        self._serve()

    def _serve(self, head: bool = False) -> None:
        rng = self.headers.get("Range")
        self.server.requests.append((self.command, self.path, rng))
        if self.server.fail_queue:
            self.send_error(self.server.fail_queue.pop(0))
            return
        path = self.server.directory / Path(urllib.parse.urlparse(self.path).path).name
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        if rng and self.server.support_range:
            start = int(rng.split("=", 1)[1].split("-", 1)[0])
            if start >= len(data):
                self.send_error(416)
                return
            body = data[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        if not head:
            self.wfile.write(body)


@contextmanager
def serve(directory: Path, *, fail_queue=(), support_range: bool = True):
    """Yield (server, base_url) for the lifetime of the block."""
    server = FixtureServer(("127.0.0.1", 0), _Handler)
    server.directory = Path(directory)
    server.requests = []
    server.fail_queue = list(fail_queue)
    server.support_range = support_range
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
