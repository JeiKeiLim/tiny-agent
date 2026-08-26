"""Full training checkpoint helpers (TASK-005.12.02).

A resumable checkpoint directory contains:

- ``weights.npz``: model weights
- ``optimizer.npz``: flattened MLX optimizer state arrays
- ``state.json``: training metadata, dataset state, and artifact hashes
- ``config/``: optional raw + resolved configuration snapshots
- ``run.jsonl``: optional snapshot of the live run log

Checkpoint writes use a temporary directory and rename so a partially written
checkpoint is not exposed as a resumable directory. ``state.json`` is written
last inside the temporary directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

FORMAT_VERSION = 1

_WEIGHTS_FILE = "weights.npz"
_OPTIMIZER_FILE = "optimizer.npz"
_STATE_FILE = "state.json"
_RUN_LOG_FILE = "run.jsonl"
_CONFIG_DIR = "config"
_RESOLVED_DIR = "resolved"


@dataclass
class CheckpointContext:
    """Optional provenance written into every full checkpoint."""

    raw_configs: dict[str, str] = field(default_factory=dict)
    resolved_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    extra_state: dict[str, Any] = field(default_factory=dict)


def _validate_component_name(name: str) -> None:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        msg = f"unsafe checkpoint component name: {name!r}"
        raise ValueError(msg)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_config_context(tmp: Path, context: CheckpointContext | None) -> None:
    if context is None:
        return
    if context.raw_configs:
        raw_dir = tmp / _CONFIG_DIR
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name, text in context.raw_configs.items():
            _validate_component_name(name)
            (raw_dir / name).write_text(text, encoding="utf-8")
    if context.resolved_configs:
        resolved_dir = tmp / _CONFIG_DIR / _RESOLVED_DIR
        resolved_dir.mkdir(parents=True, exist_ok=True)
        for name, data in context.resolved_configs.items():
            _validate_component_name(name)
            _write_json(resolved_dir / f"{name}.json", data)


def save_full_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    path: str | Path,
    state: dict[str, Any],
    context: CheckpointContext | None = None,
    run_log_path: str | Path | None = None,
) -> None:
    """Atomically write a full resumable checkpoint to ``path``."""
    path = Path(path)
    _validate_component_name(path.name)
    output_root = path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    tmp = output_root / f".{path.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        model.save_weights(str(tmp / _WEIGHTS_FILE))
        flat_state = cast(list[tuple[str, mx.array]], tree_flatten(optimizer.state))
        mx.savez(str(tmp / _OPTIMIZER_FILE), **dict(flat_state))
        _save_config_context(tmp, context)
        if run_log_path is not None and Path(run_log_path).is_file():
            shutil.copyfile(run_log_path, tmp / _RUN_LOG_FILE)

        payload: dict[str, Any] = {"format_version": FORMAT_VERSION}
        if context is not None:
            payload.update(context.extra_state)
            payload["artifact_hashes"] = context.artifact_hashes
        payload.update(state)
        _write_json(tmp / _STATE_FILE, payload)

        old = output_root / f".{path.name}.old-{os.getpid()}"
        if old.exists():
            shutil.rmtree(old)
        if path.exists():
            path.rename(old)
        tmp.rename(path)
        if old.exists():
            shutil.rmtree(old)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def read_checkpoint_state(path: str | Path) -> dict[str, Any]:
    """Read and minimally validate ``state.json`` from a full checkpoint."""
    path = Path(path)
    state_path = path / _STATE_FILE
    if not state_path.is_file():
        msg = f"not a full training checkpoint (missing {state_path.name}): {path}"
        raise ValueError(msg)
    state = cast(dict[str, Any], json.loads(state_path.read_text(encoding="utf-8")))
    if state.get("format_version") != FORMAT_VERSION:
        msg = f"unsupported checkpoint format_version: {state.get('format_version')!r}"
        raise ValueError(msg)
    for required in (_WEIGHTS_FILE, _OPTIMIZER_FILE):
        if not (path / required).is_file():
            msg = f"not a full training checkpoint (missing {required}): {path}"
            raise ValueError(msg)
    return state


def load_optimizer_state(optimizer: optim.Optimizer, path: str | Path) -> None:
    """Restore flattened optimizer arrays from ``path/optimizer.npz``."""
    path = Path(path)
    optimizer_path = path / _OPTIMIZER_FILE
    if not optimizer_path.is_file():
        msg = f"not a full training checkpoint (missing {optimizer_path.name}): {path}"
        raise ValueError(msg)

    loaded = cast(dict[str, mx.array], mx.load(str(optimizer_path)))
    flat_state = cast(list[tuple[str, Any]], [(key, value) for key, value in loaded.items()])
    optimizer.state.update(tree_unflatten(flat_state))
    restored = cast(list[tuple[str, Any]], tree_flatten(optimizer.state))
    arrays = [value for _, value in restored if isinstance(value, mx.array)]
    mx.eval(*arrays)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
