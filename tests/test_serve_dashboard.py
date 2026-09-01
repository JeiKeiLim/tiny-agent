"""Tests for the training-run dashboard server (scripts/serve_dashboard.py).

The server is a thin stdlib HTTP shell; the code under test is the run-log
discovery, the root-escaped path resolution, and the request routing.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from serve_dashboard import create_server, find_run_logs, main, resolve_run_path

LINE1 = '{"step": 1, "train_loss": 1.0, "lr": 0.0}\n'
LINE2 = '{"step": 2, "train_loss": 1.5, "lr": 0.0, "val_loss": 1.2}\n'


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "a" / "run.jsonl").write_text(LINE1 + LINE2, encoding="utf-8")
    (root / "b" / "run.jsonl").write_text(LINE1, encoding="utf-8")
    (root / "other.jsonl").write_text("not a run log\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("top secret\n", encoding="utf-8")
    return root


@pytest.fixture()
def server(root: Path):
    srv = create_server(root, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join()
    srv.server_close()


def _get(srv, path: str) -> tuple[int, bytes]:
    host, port = srv.server_address
    url = f"http://{host}:{port}{path}"
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_find_run_logs_lists_run_jsonl_only(root: Path) -> None:
    runs = find_run_logs(root)
    assert sorted(r["path"] for r in runs) == ["a/run.jsonl", "b/run.jsonl"]
    for r in runs:
        assert r["size"] > 0
        assert r["mtime"] > 0


def test_find_run_logs_missing_root(tmp_path: Path) -> None:
    assert find_run_logs(tmp_path / "nope") == []


def test_resolve_run_path_inside_root(root: Path) -> None:
    resolved = resolve_run_path(root, "a/run.jsonl")
    assert resolved is not None
    assert resolved == (root / "a" / "run.jsonl").resolve()


def test_resolve_run_path_rejects_escape(root: Path, tmp_path: Path) -> None:
    assert resolve_run_path(root, "../secret.txt") is None
    assert resolve_run_path(root, "/etc/hosts") is None
    assert resolve_run_path(root, "a") is None  # directory
    assert resolve_run_path(root, "missing/run.jsonl") is None


def test_api_runs_lists_run_logs(server, root: Path) -> None:
    status, body = _get(server, "/api/runs")
    assert status == 200
    payload = json.loads(body)
    assert payload["root"] == root.resolve().as_posix()
    assert sorted(r["path"] for r in payload["runs"]) == ["a/run.jsonl", "b/run.jsonl"]


def test_api_run_returns_full_content(server) -> None:
    status, body = _get(server, "/api/run?path=a/run.jsonl")
    assert status == 200
    assert body.decode("utf-8") == LINE1 + LINE2


def test_api_run_offset_returns_tail(server) -> None:
    status, body = _get(server, f"/api/run?path=a/run.jsonl&offset={len(LINE1.encode())}")
    assert status == 200
    assert body.decode("utf-8") == LINE2


def test_api_run_offset_beyond_eof_is_empty(server) -> None:
    status, body = _get(server, "/api/run?path=a/run.jsonl&offset=999999")
    assert status == 200
    assert body == b""


def test_api_run_rejects_traversal(server) -> None:
    status, body = _get(server, "/api/run?path=../secret.txt")
    assert status == 404
    assert b"top secret" not in body


def test_api_run_missing_path_is_400(server) -> None:
    status, _ = _get(server, "/api/run")
    assert status == 400


def test_api_run_invalid_offset_is_400(server) -> None:
    status, _ = _get(server, "/api/run?path=a/run.jsonl&offset=abc")
    assert status == 400


def test_api_unknown_route_is_404(server) -> None:
    status, _ = _get(server, "/api/nope")
    assert status == 404


def test_serves_dashboard_html(server) -> None:
    status, body = _get(server, "/")
    assert status == 200
    assert b"Kestrel Training Dashboard" in body


def test_main_rejects_missing_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(tmp_path / "nope")]) == 1
    assert "not found" in capsys.readouterr().err
