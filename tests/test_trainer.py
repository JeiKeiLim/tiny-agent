"""Tests for the shared trainer (TASK-005.03)."""

import json
import math
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.optimizers as optim
import pytest
from pydantic import ValidationError

from kestrel.model.config import ModelConfig
from kestrel.model.io import load
from kestrel.model.kestrel import Kestrel
from kestrel.train.checkpoint import (
    load_optimizer_state,
    read_checkpoint_state,
    save_full_checkpoint,
)
from kestrel.train.trainer import (
    ResumeState,
    TrainerConfig,
    TrainResult,
    _prune_step_checkpoints,
    lr_at,
    train,
)

TINY = ModelConfig(
    name="tiny",
    vocab_size=64,
    context_length=16,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    hidden_size=32,
    intermediate_size=64,
)


def _batches(n: int, b: int, t: int, vocab: int, seed: int = 0) -> list[tuple[mx.array, mx.array]]:
    """Learnable counting batches: token at pos i is (start + i) % vocab, so next = cur + 1."""
    mx.random.seed(seed)
    out: list[tuple[mx.array, mx.array]] = []
    for _ in range(n):
        start = mx.random.randint(0, vocab, (b,))
        pos = mx.arange(t)
        x = ((start[:, None] + pos[None, :]) % vocab).astype(mx.int32)
        target = mx.concatenate([x[:, 1:], x[:, -1:]], axis=1)
        out.append((x, target))
    return out


def _tiny_model() -> Kestrel:
    mx.random.seed(0)
    return Kestrel(TINY)


def test_train_loss_decreases(tmp_path: Path) -> None:
    model = _tiny_model()
    train_ds = _batches(30, 2, 16, 64, seed=1)
    val_ds = _batches(5, 2, 16, 64, seed=2)
    cfg = TrainerConfig(
        num_steps=30,
        warmup_steps=5,
        eval_every=10,
        eval_iters=3,
        log_every=100,
        save_every=100,
        output_dir=str(tmp_path),
    )
    result = train(model, train_ds, val_ds, cfg)
    assert isinstance(result, TrainResult)
    assert result.num_steps == 30
    first = result.history[0][1]
    last = result.history[-1][1]
    assert math.isfinite(last)
    assert last < first, f"loss did not decrease: first={first:.4f} last={last:.4f}"


def test_trainer_config_strict() -> None:
    TrainerConfig(lr=1e-3)  # valid
    with pytest.raises(ValidationError):
        TrainerConfig(lr="fast")  # type: ignore[arg-type]  # mistyped
    with pytest.raises(ValidationError):
        TrainerConfig(bogus=1)  # type: ignore[call-arg]  # unknown key


def test_trainer_config_default_betas() -> None:
    assert TrainerConfig().betas == (0.9, 0.95)  # beta2=0.95 is the LLaMA/modern default
    TrainerConfig(betas=(0.9, 0.99))  # custom betas accepted


def test_lr_schedule() -> None:
    cfg = TrainerConfig(lr=1.0, warmup_steps=10, num_steps=50)
    assert lr_at(0, cfg) == 0.0  # starts at 0
    assert lr_at(10, cfg) == pytest.approx(1.0)  # peak at end of warmup
    assert lr_at(50, cfg) == pytest.approx(0.0, abs=1e-9)  # ~0 at the end
    warmup = [lr_at(s, cfg) for s in range(11)]
    assert warmup == sorted(warmup)  # monotonic increase during warmup
    decay = [lr_at(s, cfg) for s in range(10, 51)]
    assert decay == sorted(decay, reverse=True)  # monotonic decrease during decay


def test_lr_schedule_uses_schedule_steps_override() -> None:
    cfg = TrainerConfig(lr=1.0, warmup_steps=10, num_steps=0)
    assert lr_at(0, cfg, schedule_steps=50) == 0.0
    assert lr_at(10, cfg, schedule_steps=50) == pytest.approx(1.0)
    assert lr_at(50, cfg, schedule_steps=50) == pytest.approx(0.0, abs=1e-9)
    assert lr_at(25, cfg, schedule_steps=50) != lr_at(25, cfg, schedule_steps=100)


class _EstimatedBatches:
    def __init__(self, batches: list[tuple[mx.array, ...]], steps: int) -> None:
        self.batches = batches
        self.steps = steps

    def __iter__(self) -> Iterator[tuple[mx.array, ...]]:
        return iter(self.batches)

    def estimated_steps(self) -> int:
        return self.steps


def test_train_auto_num_steps_uses_dataset_estimated_steps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _tiny_model()
    train_ds = _EstimatedBatches(_batches(12, 2, 16, 64, seed=1), steps=7)
    val_ds = _batches(3, 2, 16, 64, seed=2)
    cfg = TrainerConfig(
        num_steps=0,
        warmup_steps=2,
        eval_every=100,
        log_every=100,
        save_every=100,
        output_dir=str(tmp_path),
    )
    result = train(model, train_ds, val_ds, cfg)
    assert result.num_steps == 12
    assert result.schedule_steps == 7
    assert "schedule_steps=7" in capsys.readouterr().out


def test_checkpoint_reloads(tmp_path: Path) -> None:
    model = _tiny_model()
    train_ds = _batches(12, 2, 16, 64, seed=3)
    val_ds = _batches(3, 2, 16, 64, seed=4)
    cfg = TrainerConfig(
        num_steps=10,
        warmup_steps=2,
        save_every=5,
        eval_every=100,
        log_every=100,
        output_dir=str(tmp_path),
    )
    train(model, train_ds, val_ds, cfg)
    reloaded = load(TINY, tmp_path / "final")
    x = train_ds[0][0]
    mx.eval(model.parameters(), reloaded.parameters())
    assert mx.allclose(model(x), reloaded(x))


def test_in_loop_val_loss(tmp_path: Path) -> None:
    model = _tiny_model()
    train_ds = _batches(30, 2, 16, 64, seed=5)
    val_ds = _batches(5, 2, 16, 64, seed=6)
    cfg = TrainerConfig(
        num_steps=30,
        warmup_steps=5,
        eval_every=10,
        eval_iters=3,
        log_every=100,
        save_every=100,
        output_dir=str(tmp_path),
    )
    result = train(model, train_ds, val_ds, cfg)
    val_entries = [h[2] for h in result.history if h[2] is not None]
    assert len(val_entries) == 3  # val ran at steps 10, 20, 30
    assert all(math.isfinite(v) for v in val_entries)
    assert result.best_val_loss == min(val_entries)


def test_trainer_checkpoint_retention_config() -> None:
    cfg = TrainerConfig()
    assert cfg.keep_latest_checkpoints == 3
    assert cfg.keep_best_checkpoint is True

    TrainerConfig(keep_latest_checkpoints=None)
    TrainerConfig(keep_latest_checkpoints=1)
    TrainerConfig(keep_best_checkpoint=False)

    with pytest.raises(ValidationError):
        TrainerConfig(keep_latest_checkpoints=0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TrainerConfig(keep_latest_checkpoints="2")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TrainerConfig(keep_latest_checkpoints=2.0)  # type: ignore[arg-type]


def _make_step_dirs(root: Path, steps: list[int]) -> None:
    base_time = 1_000_000_000.0
    for offset, step in enumerate(steps):
        path = root / f"step_{step:06d}"
        path.mkdir(parents=True)
        mtime = base_time + offset
        os.utime(path, (mtime, mtime))


def test_prune_step_checkpoints_keeps_newest(tmp_path: Path) -> None:
    _make_step_dirs(tmp_path, [1, 2, 3, 4, 5])
    (tmp_path / "best").mkdir()
    (tmp_path / "final").mkdir()
    (tmp_path / "notes").mkdir()

    _prune_step_checkpoints(tmp_path, 2)

    remaining = {path.name for path in tmp_path.iterdir()}
    assert remaining == {"step_000004", "step_000005", "best", "final", "notes"}


def test_prune_step_checkpoints_none_retains_all(tmp_path: Path) -> None:
    _make_step_dirs(tmp_path, [1, 2, 3])

    _prune_step_checkpoints(tmp_path, None)

    remaining = {path.name for path in tmp_path.iterdir()}
    assert remaining == {"step_000001", "step_000002", "step_000003"}


def test_prune_step_checkpoints_missing_dir_is_noop(tmp_path: Path) -> None:
    _prune_step_checkpoints(tmp_path / "missing", 1)


def test_train_prunes_old_step_checkpoints(tmp_path: Path) -> None:
    model = _tiny_model()
    train_ds = _batches(10, 2, 16, 64, seed=7)
    val_ds = _batches(3, 2, 16, 64, seed=8)
    cfg = TrainerConfig(
        num_steps=10,
        warmup_steps=2,
        eval_every=100,
        log_every=100,
        save_every=2,
        keep_latest_checkpoints=2,
        output_dir=str(tmp_path),
    )

    train(model, train_ds, val_ds, cfg)

    step_dirs = sorted(
        path.name for path in tmp_path.iterdir() if path.is_dir() and path.name.startswith("step_")
    )
    assert step_dirs == ["step_000008", "step_000010"]
    assert (tmp_path / "final").is_dir()


def test_train_retains_all_step_checkpoints_when_pruning_disabled(tmp_path: Path) -> None:
    model = _tiny_model()
    train_ds = _batches(10, 2, 16, 64, seed=9)
    val_ds = _batches(3, 2, 16, 64, seed=10)
    cfg = TrainerConfig(
        num_steps=10,
        warmup_steps=2,
        eval_every=100,
        log_every=100,
        save_every=2,
        keep_latest_checkpoints=None,
        output_dir=str(tmp_path),
    )

    train(model, train_ds, val_ds, cfg)

    step_dirs = sorted(
        path.name for path in tmp_path.iterdir() if path.is_dir() and path.name.startswith("step_")
    )
    assert step_dirs == ["step_000002", "step_000004", "step_000006", "step_000008", "step_000010"]


def test_best_checkpoint_written_on_strict_improvement_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    val_values = iter([10.0, 11.0, 9.0])

    def fake_val_loss(
        model: Kestrel,
        val: Iterable[tuple[mx.array, ...]],
        iters: int,
        use_loss_mask: bool = False,
    ) -> float:
        return next(val_values)

    best_writes: list[Path] = []

    def fake_save(
        model: Kestrel,
        optimizer: optim.Optimizer,
        path: str | Path,
        state: dict[str, Any],
        context: Any = None,
        run_log_path: str | Path | None = None,
    ) -> None:
        path = Path(path)
        if path.name == "best":
            best_writes.append(path)
        save_full_checkpoint(
            model, optimizer, path, state, context=context, run_log_path=run_log_path
        )

    monkeypatch.setattr("kestrel.train.trainer.estimate_val_loss", fake_val_loss)
    monkeypatch.setattr("kestrel.train.trainer.save_full_checkpoint", fake_save)

    model = _tiny_model()
    train_ds = _batches(12, 2, 16, 64, seed=11)
    val_ds = _batches(3, 2, 16, 64, seed=12)
    cfg = TrainerConfig(
        num_steps=12,
        warmup_steps=2,
        eval_every=4,
        eval_iters=1,
        log_every=100,
        save_every=100,
        keep_best_checkpoint=True,
        output_dir=str(tmp_path),
    )

    train(model, train_ds, val_ds, cfg)

    assert len(best_writes) == 2
    assert (tmp_path / "best" / "weights.npz").exists()
    assert (tmp_path / "best" / "optimizer.npz").exists()
    assert (tmp_path / "best" / "state.json").exists()


def test_best_checkpoint_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    val_values = iter([10.0, 9.0])

    def fake_val_loss(
        model: Kestrel,
        val: Iterable[tuple[mx.array, ...]],
        iters: int,
        use_loss_mask: bool = False,
    ) -> float:
        return next(val_values)

    monkeypatch.setattr("kestrel.train.trainer.estimate_val_loss", fake_val_loss)

    model = _tiny_model()
    train_ds = _batches(8, 2, 16, 64, seed=13)
    val_ds = _batches(3, 2, 16, 64, seed=14)
    cfg = TrainerConfig(
        num_steps=8,
        warmup_steps=2,
        eval_every=4,
        eval_iters=1,
        log_every=100,
        save_every=100,
        keep_best_checkpoint=False,
        output_dir=str(tmp_path),
    )

    train(model, train_ds, val_ds, cfg)

    assert not (tmp_path / "best").exists()
    assert (tmp_path / "final").is_dir()


def test_best_checkpoint_survives_pruning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    val_values = iter([10.0, 9.0])

    def fake_val_loss(
        model: Kestrel,
        val: Iterable[tuple[mx.array, ...]],
        iters: int,
        use_loss_mask: bool = False,
    ) -> float:
        return next(val_values)

    monkeypatch.setattr("kestrel.train.trainer.estimate_val_loss", fake_val_loss)

    model = _tiny_model()
    train_ds = _batches(10, 2, 16, 64, seed=15)
    val_ds = _batches(3, 2, 16, 64, seed=16)
    cfg = TrainerConfig(
        num_steps=10,
        warmup_steps=2,
        eval_every=5,
        eval_iters=1,
        log_every=100,
        save_every=2,
        keep_latest_checkpoints=2,
        keep_best_checkpoint=True,
        output_dir=str(tmp_path),
    )

    train(model, train_ds, val_ds, cfg)

    step_dirs = sorted(
        path.name for path in tmp_path.iterdir() if path.is_dir() and path.name.startswith("step_")
    )
    assert step_dirs == ["step_000008", "step_000010"]
    assert (tmp_path / "best").is_dir()
    assert (tmp_path / "final").is_dir()


def _tiny_optimizer(cfg: TrainerConfig) -> optim.Optimizer:
    return optim.AdamW(learning_rate=cfg.lr, betas=list(cfg.betas), weight_decay=cfg.weight_decay)


def test_train_writes_run_jsonl(tmp_path: Path) -> None:
    model = _tiny_model()
    train_ds = _batches(5, 2, 16, 64, seed=20)
    val_ds = _batches(2, 2, 16, 64, seed=21)
    cfg = TrainerConfig(
        num_steps=5,
        warmup_steps=2,
        eval_every=100,
        log_every=100,
        save_every=100,
        output_dir=str(tmp_path),
    )

    train(model, train_ds, val_ds, cfg)

    lines = (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]
    assert [entry["step"] for entry in entries] == [1, 2, 3, 4, 5]
    assert all(math.isfinite(entry["train_loss"]) for entry in entries)
    assert all("lr" in entry for entry in entries)


def test_full_checkpoint_contents(tmp_path: Path) -> None:
    model = _tiny_model()
    train_ds = _batches(4, 2, 16, 64, seed=22)
    val_ds = _batches(2, 2, 16, 64, seed=23)
    cfg = TrainerConfig(
        num_steps=4,
        warmup_steps=2,
        eval_every=100,
        log_every=100,
        save_every=2,
        output_dir=str(tmp_path),
    )

    train(model, train_ds, val_ds, cfg)

    for name in ("step_000002", "step_000004", "final"):
        checkpoint = tmp_path / name
        assert (checkpoint / "weights.npz").exists()
        assert (checkpoint / "optimizer.npz").exists()
        assert (checkpoint / "state.json").exists()
        assert (checkpoint / "run.jsonl").exists()

    state = read_checkpoint_state(tmp_path / "final")
    assert state["step"] == 4
    assert state["schedule_steps"] == 4
    assert state["dataset_state"] is None


def test_trainer_resume_continues_global_steps(tmp_path: Path) -> None:
    batches = _batches(10, 2, 16, 64, seed=24)
    val_ds = _batches(3, 2, 16, 64, seed=25)

    first_cfg = TrainerConfig(
        num_steps=5,
        warmup_steps=2,
        eval_every=100,
        log_every=100,
        save_every=5,
        output_dir=str(tmp_path),
    )
    first_model = _tiny_model()
    train(first_model, iter(batches), val_ds, first_cfg, schedule_steps=10)

    checkpoint_dir = tmp_path / "step_000005"
    state = read_checkpoint_state(checkpoint_dir)
    assert state["step"] == 5
    assert state["schedule_steps"] == 10

    second_cfg = TrainerConfig(
        num_steps=10,
        warmup_steps=2,
        eval_every=100,
        log_every=100,
        save_every=5,
        output_dir=str(tmp_path),
    )
    second_model = load(TINY, checkpoint_dir)
    optimizer = _tiny_optimizer(second_cfg)
    load_optimizer_state(optimizer, checkpoint_dir)

    resume = ResumeState(
        step=int(state["step"]),
        schedule_steps=int(state["schedule_steps"]),
        best_val_loss=cast(float | None, state.get("best_val_loss")),
        last_train_loss=cast(float | None, state.get("last_train_loss")),
        last_val_loss=cast(float | None, state.get("last_val_loss")),
        last_eval_step=cast(int | None, state.get("last_eval_step")),
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
    )
    result = train(second_model, iter(batches[5:]), val_ds, second_cfg, resume=resume)

    assert result.num_steps == 10
    assert len(result.history) == 5
    assert [entry[0] for entry in result.history] == [6, 7, 8, 9, 10]
    final_state = read_checkpoint_state(tmp_path / "final")
    assert final_state["step"] == 10


def test_trainer_resume_restores_optimizer_state(tmp_path: Path) -> None:
    batches = _batches(5, 2, 16, 64, seed=26)
    val_ds = _batches(2, 2, 16, 64, seed=27)
    cfg = TrainerConfig(
        num_steps=5,
        warmup_steps=2,
        eval_every=100,
        log_every=100,
        save_every=5,
        output_dir=str(tmp_path),
    )

    train(_tiny_model(), iter(batches), val_ds, cfg)

    optimizer = _tiny_optimizer(cfg)
    load_optimizer_state(optimizer, tmp_path / "step_000005")
    assert int(optimizer.state["step"].item()) == 5
    moment = optimizer.state["embed"]["weight"]["m"]
    assert bool(mx.abs(moment).sum() > 0)
