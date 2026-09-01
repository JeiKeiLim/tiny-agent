"""Local training-run dashboard server (TASK-010).

Serves ``scripts/dashboard.html`` plus a small JSON API over HTTP so a browser
can watch live ``run.jsonl`` files (the per-step trainer log written by
``train/trainer.py``) and overlay multiple runs for comparison:

- ``GET /``                              -> dashboard.html
- ``GET /api/runs``                      -> JSON list of run.jsonl files under --root
- ``GET /api/run?path=REL&offset=N``     -> raw JSONL tail from byte offset N

Stdlib only. The server binds to localhost by default and rejects any path
that escapes the root directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

DASHBOARD_FILE = Path(__file__).with_name("dashboard.html")
RUN_LOG_NAME = "run.jsonl"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_ROOT = Path("checkpoints")


def find_run_logs(root: Path) -> list[dict[str, float | str]]:
    """List every ``run.jsonl`` under ``root`` as ``{path, size, mtime}``.

    Paths are relative to ``root`` (POSIX separators). Newest first, with the
    relative path as a tie-breaker so the live run sorts to the top.
    """
    if not root.is_dir():
        return []
    runs: list[dict[str, float | str]] = []
    for path in root.rglob(RUN_LOG_NAME):
        if not path.is_file():
            continue
        stat = path.stat()
        runs.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": float(stat.st_size),
                "mtime": stat.st_mtime,
            }
        )
    runs.sort(key=lambda run: (-cast(float, run["mtime"]), cast(str, run["path"])))
    return runs


def resolve_run_path(root: Path, rel: str) -> Path | None:
    """Resolve ``rel`` against ``root``.

    Returns the resolved file path, or ``None`` when the path escapes the
    root, is a directory, or does not exist.
    """
    root_resolved = root.resolve()
    try:
        candidate = (root / rel).resolve()
    except (OSError, ValueError):
        return None
    if candidate == root_resolved or root_resolved not in candidate.parents:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _build_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._serve_dashboard()
            elif parsed.path == "/api/runs":
                self._send_json({"root": root.as_posix(), "runs": find_run_logs(root)})
            elif parsed.path == "/api/run":
                self._serve_run(parse_qs(parsed.query))
            else:
                self._send_json({"error": "not found"}, status=404)

        def _serve_dashboard(self) -> None:
            if not DASHBOARD_FILE.is_file():
                self._send_json({"error": "dashboard.html not found"}, status=404)
                return
            self._send_bytes(DASHBOARD_FILE.read_bytes(), "text/html; charset=utf-8")

        def _serve_run(self, query: dict[str, list[str]]) -> None:
            rel = query.get("path", [""])[0]
            if not rel:
                self._send_json({"error": "missing path"}, status=400)
                return
            try:
                offset = int(query.get("offset", ["0"])[0])
            except ValueError:
                self._send_json({"error": "invalid offset"}, status=400)
                return
            if offset < 0:
                self._send_json({"error": "invalid offset"}, status=400)
                return
            path = resolve_run_path(root, rel)
            if path is None:
                self._send_json({"error": "not found"}, status=404)
                return
            with path.open("rb") as fh:
                fh.seek(offset)
                self._send_bytes(fh.read(), "application/x-ndjson")

        def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: object, status: int = 200) -> None:
            self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

    return DashboardHandler


class DashboardServer(ThreadingHTTPServer):
    """Threading HTTP server that serves the dashboard for one root directory."""

    def __init__(self, addr: tuple[str, int], root: Path) -> None:
        super().__init__(addr, _build_handler(root))
        self.root = root
        self.daemon_threads = True


def create_server(
    root: Path, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> DashboardServer:
    return DashboardServer((host, port), root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the Kestrel training-run dashboard.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="directory scanned for run.jsonl files (default: checkpoints/)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port (default: 8787)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"error: root directory not found: {root}", file=sys.stderr)
        return 1
    server = create_server(root, args.host, args.port)
    port = server.server_address[1]
    print(f"Kestrel training dashboard: http://{args.host}:{port}/ (root: {root})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
