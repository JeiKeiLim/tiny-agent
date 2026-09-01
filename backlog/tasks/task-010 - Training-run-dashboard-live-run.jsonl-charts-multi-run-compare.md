---
id: TASK-010
title: Training run dashboard (live run.jsonl charts + multi-run compare)
status: Done
assignee:
  - '@agent'
created_date: '2026-09-01 07:13'
updated_date: '2026-09-01 07:36'
labels: []
dependencies: []
references:
  - backlog/docs/doc-001
ordinal: 59000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a local web dashboard to monitor Kestrel training runs live and compare runs. The shared trainer appends one JSON object per step to <output_dir>/run.jsonl ({step, train_loss, lr, optional val_loss}); a browser cannot watch a local file, so a small stdlib-only HTTP server (scripts/serve_dashboard.py) serves a self-contained HTML page (scripts/dashboard.html, no CDN deps) plus a tiny JSON API. The page lists run.jsonl files found under the server root (default checkpoints/), lets the user add one or more runs (checkbox list + manual path input), and overlays their train_loss / val_loss / lr curves on canvas charts (loss panel + LR panel). It auto-refreshes every ~3s (pausable) so a live run keeps updating, and parses incrementally (dedupe by step, last occurrence wins, to absorb resume-induced duplicate lines). Also update README.md and AGENTS.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 scripts/serve_dashboard.py serves the dashboard at GET / and exposes GET /api/runs (lists run.jsonl files under --root, default checkpoints/) and GET /api/run?path=REL&offset=N (raw JSONL tail from byte offset); paths escaping the root are rejected (403/404); stdlib only; --port (default 8787) and --host (default 127.0.0.1) flags
- [x] #2 scripts/dashboard.html is self-contained (no external/CDN assets): run picker from /api/runs + manual path add, per-run colors, overlaid train_loss (solid) and val_loss (dashed) chart plus a separate LR chart, legend with latest step/loss per run, auto-refresh polling (default 3s) with pause, incremental parse with step dedupe (last wins)
- [x] #3 tests/test_serve_dashboard.py covers run discovery, content fetch, offset tail, and path-traversal rejection using the established importlib spec_from_file_location script-import pattern; make check is green
- [x] #4 README.md documents the dashboard command and AGENTS.md status line mentions it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. scripts/serve_dashboard.py: stdlib ThreadingHTTPServer; GET / -> dashboard.html; GET /api/runs -> JSON list of run.jsonl under --root (default checkpoints/); GET /api/run?path=REL&offset=N -> JSONL tail with root-escape rejection; --port 8787 --host 127.0.0.1. 2. scripts/dashboard.html: self-contained canvas charts (no CDN); run picker + manual path add; per-run colors; train_loss solid + val_loss dashed overlay chart; separate LR chart; legend w/ latest step+loss; 3s auto-refresh polling with pause; incremental parse with step dedupe (last wins). 3. tests/test_serve_dashboard.py: discovery, fetch, offset, traversal rejection (importlib script-import pattern). 4. README.md + AGENTS.md updates. 5. make check green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented: scripts/serve_dashboard.py (stdlib ThreadingHTTPServer; GET /, /api/runs, /api/run?path&offset; root-escape rejection; --root/--host/--port), scripts/dashboard.html (self-contained canvas charts, no CDN: run picker + manual add, per-run colors, train solid + val dashed loss chart, LR panel, legend w/ latest step/loss, 3s pausable auto-refresh, byte-offset incremental fetch, step dedupe last-wins, replaced-file reset, log-scale toggle, hover readouts), tests/test_serve_dashboard.py (14 tests). Smoke-tested live against checkpoints/pretrain/50m-3b (growing run.jsonl, offset tail works). make check green (368 tests). README + AGENTS.md updated.

Follow-up (user feedback): train vs val loss were hard to tell apart when multiple runs overlaid (both used the run color). Val loss now renders as a lighter shade (lighten(color, 0.55)) + dashed, train stays solid full-color; legend shows both line styles per run; chart title updated. make check green (368 tests).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a local training-run dashboard: scripts/serve_dashboard.py (stdlib-only HTTP server: GET / serves the page, /api/runs lists run.jsonl under --root, /api/run?path&offset serves byte-offset tails with root-escape rejection) and scripts/dashboard.html (self-contained, no CDN: checkbox run picker + manual path add, per-run colored overlaid train_loss/val_loss chart + separate LR panel, legend with latest step/loss, 3s pausable auto-refresh, incremental byte-offset parsing with step dedupe so resumed runs and live runs both work, log-scale toggle, hover readouts). 14 tests in tests/test_serve_dashboard.py; make check green (368 tests); smoke-tested live against the running 50m-3b pretrain (offset tail tracks the growing file). README.md + AGENTS.md updated.
<!-- SECTION:FINAL_SUMMARY:END -->
