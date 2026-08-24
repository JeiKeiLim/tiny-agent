"""Kestrel model I/O (TASK-002.02).

``load(config, checkpoint)`` is the Kestrel model factory: it builds a
``Kestrel`` from ``config`` and, when ``checkpoint`` is given, loads the weights
from that checkpoint directory. ``save(model, path)`` writes a model's weights
to a checkpoint directory. Checkpoints follow the ``checkpoints/<phase>/<name>/``
convention (a directory holding ``weights.npz``).

This is the Kestrel factory only. The Qwen3/pretrained loader is a separate
Track B build (step 7, ``model/pretrained.py``) and does not route through here.
"""

from __future__ import annotations

from pathlib import Path

from kestrel.model.config import ModelConfig
from kestrel.model.kestrel import Kestrel

_WEIGHTS_FILE = "weights.npz"


def save(model: Kestrel, path: str | Path) -> None:
    """Save ``model`` weights to the checkpoint directory ``path``.

    Creates ``path`` (and any parents) if needed, then writes ``weights.npz``
    inside it. ``path`` is expected to follow the
    ``checkpoints/<phase>/<name>/`` convention (e.g.
    ``checkpoints/pretrain/kestrel-50m``).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path / _WEIGHTS_FILE))


def load(config: ModelConfig, checkpoint: str | Path | None = None) -> Kestrel:
    """Build a ``Kestrel`` from ``config``; load weights from ``checkpoint`` if given.

    With ``checkpoint=None`` the returned model is randomly initialized.
    Otherwise its weights are loaded (strictly) from ``checkpoint/weights.npz``.
    """
    model = Kestrel(config)
    if checkpoint is not None:
        model.load_weights(str(Path(checkpoint) / _WEIGHTS_FILE))
    return model
