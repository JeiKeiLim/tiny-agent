"""External benchmark evaluation for Kestrel pretrained checkpoints.

This module evaluates a saved pretrain checkpoint on locally downloaded
benchmark files. It reads the raw evaluation files produced by
``scripts/download_pretrain_eval_datasets.py`` and does not require network
access, Hugging Face dataset caching, or unpacked archives.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import pyarrow.parquet as pq
from mlx.nn.losses import cross_entropy
from tokenizers import Tokenizer

from kestrel.common.config import load_config
from kestrel.model.config import ModelConfig
from kestrel.model.io import load as load_model
from kestrel.train.pretrain import PretrainConfig

LM_KIND = "language_modeling"
MCQ_KIND = "multiple_choice"


@dataclass(frozen=True)
class BenchmarkSpec:
    """Static metadata for one externally downloaded benchmark."""

    name: str
    kind: str
    text_field: str | None = None
    large: bool = False


BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("hellaswag", MCQ_KIND),
    BenchmarkSpec("piqa", MCQ_KIND),
    BenchmarkSpec("arc_easy", MCQ_KIND),
    BenchmarkSpec("arc_challenge", MCQ_KIND),
    BenchmarkSpec("winogrande", MCQ_KIND),
    BenchmarkSpec("openbookqa", MCQ_KIND),
    BenchmarkSpec("boolq", MCQ_KIND),
    BenchmarkSpec("sciq", MCQ_KIND),
    BenchmarkSpec("mmlu", MCQ_KIND),
    BenchmarkSpec("wikitext2", LM_KIND, text_field="page"),
    BenchmarkSpec("wikitext103", LM_KIND, text_field="page"),
    BenchmarkSpec("c4_en_validation", LM_KIND, text_field="text", large=True),
    BenchmarkSpec("pile_test", LM_KIND, text_field="text", large=True),
    BenchmarkSpec("lambada", LM_KIND, text_field="text"),
)

SPEC_BY_NAME: dict[str, BenchmarkSpec] = {spec.name: spec for spec in BENCHMARK_SPECS}


@dataclass(frozen=True)
class BenchmarkResult:
    """Evaluation result for one benchmark."""

    name: str
    kind: str
    status: str
    metrics: dict[str, float | int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "metrics": self.metrics,
            "error": self.error,
        }


@dataclass(frozen=True)
class BenchmarkScorecard:
    """Scorecard for one checkpoint across selected benchmarks."""

    format_version: int
    checkpoint: str
    pretrain_config: str
    data_dir: str
    seed: int
    max_tokens: int | None
    max_examples: int | None
    results: tuple[BenchmarkResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "checkpoint": self.checkpoint,
            "pretrain_config": self.pretrain_config,
            "data_dir": self.data_dir,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "max_examples": self.max_examples,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass
class BpbAccumulator:
    """Token- and byte-weighted language-modeling accumulator."""

    nats: float = 0.0
    tokens: int = 0
    bytes: int = 0
    examples: int = 0

    @property
    def loss(self) -> float:
        if self.tokens == 0:
            msg = "no tokens have been accumulated"
            raise ValueError(msg)
        return self.nats / self.tokens

    @property
    def perplexity(self) -> float:
        return math.exp(self.loss)

    @property
    def bits_per_token(self) -> float:
        return self.loss / math.log(2.0)

    @property
    def bpb(self) -> float:
        if self.bytes == 0:
            msg = "no bytes have been accumulated"
            raise ValueError(msg)
        return self.nats / (math.log(2.0) * self.bytes)


@dataclass(frozen=True)
class McqCase:
    """One multiple-choice example normalized to context/choices/label."""

    context: str
    choices: tuple[str, ...]
    label: int


def benchmark_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in BENCHMARK_SPECS)


def parse_only(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    names = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = names - set(SPEC_BY_NAME)
    if unknown:
        msg = f"unknown benchmark name(s): {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    return names


def selected_specs(skip_large: bool, only: set[str] | None) -> list[BenchmarkSpec]:
    selected = []
    for spec in BENCHMARK_SPECS:
        if skip_large and spec.large:
            continue
        if only is not None and spec.name not in only:
            continue
        selected.append(spec)
    return selected


def _dataset_dir(data_dir: Path, name: str) -> Path | None:
    dest = data_dir / name
    if not dest.is_dir():
        return None
    if not (dest / "manifest.json").is_file():
        return None
    return dest


def _manifest_files(dataset_dir: Path) -> list[str]:
    manifest_path = dataset_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = data.get("files", [])
    if not isinstance(files, list):
        return []
    return [str(item) for item in files if isinstance(item, str)]


def iter_rows(dataset_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield rows from local Parquet or JSONL files without extra caching."""
    for raw_path in _manifest_files(dataset_dir):
        path = dataset_dir / raw_path
        if not path.is_file():
            msg = f"benchmark file not found: {path}"
            raise FileNotFoundError(msg)
        if path.suffix == ".parquet":
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=1024):
                yield from batch.to_pylist()
            continue
        if path.name.endswith((".jsonl", ".json", ".jsonl.gz", ".json.gz")):
            if path.name.endswith(".gz"):
                with gzip.open(path, mode="rt", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if line:
                            row = json.loads(line)
                            if isinstance(row, dict):
                                yield row
            else:
                with path.open("r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if line:
                            row = json.loads(line)
                            if isinstance(row, dict):
                                yield row
            continue
        msg = f"unsupported benchmark file type: {path}"
        raise ValueError(msg)


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _choice_label(labels: object, answer: str) -> int | None:
    if (
        isinstance(labels, list)
        and all(isinstance(item, str) for item in labels)
        and answer in labels
    ):
        return labels.index(answer)
    if len(answer) == 1 and answer.isalpha():
        index = ord(answer.upper()) - ord("A")
        if index >= 0:
            return index
    return None


def _sciq_shuffle_perm(question: str, seed: int) -> list[int]:
    digest = hashlib.sha256(f"{seed}:{question}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    perm = list(range(4))
    rng.shuffle(perm)
    return perm


def mcq_case(name: str, row: dict[str, Any], seed: int = 0) -> McqCase | None:
    """Normalize one raw benchmark row into a multiple-choice case."""
    if name == "hellaswag":
        context = str(row.get("ctx", ""))
        endings = row.get("endings")
        if not isinstance(endings, list) or not endings:
            return None
        choices = tuple(str(item) for item in endings)
        label = _as_int(row.get("label"))
        if label is None or label >= len(choices):
            return None
        return McqCase(context=context, choices=choices, label=label)

    if name == "piqa":
        context = str(row.get("goal", ""))
        sol1 = row.get("sol1")
        sol2 = row.get("sol2")
        label = _as_int(row.get("label"))
        if not isinstance(sol1, str) or not isinstance(sol2, str) or label not in (0, 1):
            return None
        return McqCase(context=context, choices=(sol1, sol2), label=label)

    if name in {"arc_easy", "arc_challenge"}:
        question = row.get("question")
        choices_raw = row.get("choices")
        answer = row.get("answerKey")
        if not isinstance(question, str) or not isinstance(choices_raw, dict):
            return None
        texts = choices_raw.get("text")
        if not isinstance(texts, list) or not texts:
            return None
        label = _choice_label(choices_raw.get("label"), str(answer))
        if label is None or label >= len(texts):
            return None
        return McqCase(
            context=question,
            choices=tuple(str(item) for item in texts),
            label=label,
        )

    if name == "winogrande":
        sentence = row.get("sentence")
        option1 = row.get("option1")
        option2 = row.get("option2")
        answer = str(row.get("answer", ""))
        if not isinstance(sentence, str) or not isinstance(option1, str):
            return None
        if not isinstance(option2, str):
            return None
        label = 0 if answer == "1" else 1
        if "_" in sentence:
            prefix, suffix = sentence.split("_", 1)
            prefix = prefix.rstrip()
            suffix = suffix.lstrip()
            context = f"{prefix} " if prefix else ""
            if suffix:
                choices = (f"{option1} {suffix}", f"{option2} {suffix}")
            else:
                choices = (option1, option2)
        else:
            context = f"{sentence} "
            choices = (option1, option2)
        return McqCase(context=context, choices=choices, label=label)

    if name == "openbookqa":
        question = row.get("question_stem")
        choices_raw = row.get("choices")
        answer = row.get("answerKey")
        if not isinstance(question, str) or not isinstance(choices_raw, dict):
            return None
        texts = choices_raw.get("text")
        if not isinstance(texts, list) or not texts:
            return None
        label = _choice_label(choices_raw.get("label"), str(answer))
        if label is None or label >= len(texts):
            return None
        return McqCase(
            context=question,
            choices=tuple(str(item) for item in texts),
            label=label,
        )

    if name == "boolq":
        passage = row.get("passage")
        question = row.get("question")
        label = _as_int(row.get("label"))
        if not isinstance(passage, str) or not isinstance(question, str):
            return None
        if label not in (0, 1):
            return None
        context = f"{passage}\nQuestion: {question} "
        return McqCase(context=context, choices=("no", "yes"), label=label)

    if name == "sciq":
        question = row.get("question")
        distractor1 = row.get("distractor1")
        distractor2 = row.get("distractor2")
        distractor3 = row.get("distractor3")
        correct = row.get("correct_answer")
        raw_values = [distractor1, distractor2, distractor3, correct]
        if not isinstance(question, str) or not all(isinstance(item, str) for item in raw_values):
            return None
        values = [str(item) for item in raw_values]
        perm = _sciq_shuffle_perm(question, seed)
        choices = tuple(values[index] for index in perm)
        label = perm.index(3)
        return McqCase(context=question, choices=choices, label=label)

    if name == "mmlu":
        question = row.get("question")
        choices_raw = row.get("choices")
        answer = _as_int(row.get("answer"))
        if not isinstance(question, str) or not isinstance(choices_raw, list):
            return None
        if answer is None or answer >= len(choices_raw):
            return None
        return McqCase(
            context=question,
            choices=tuple(str(item) for item in choices_raw),
            label=answer,
        )

    msg = f"no multiple-choice mapping for benchmark: {name}"
    raise ValueError(msg)


def _continuation_logprob(
    model: nn.Module,
    tokenizer: Tokenizer,
    context: str,
    continuation: str,
    max_length: int,
) -> tuple[float, int] | None:
    """Return total logprob and token count for continuation given context."""
    if not continuation:
        return None
    full = context + continuation
    encoding = tokenizer.encode(full, add_special_tokens=False)
    original_ids = encoding.ids
    if not original_ids:
        return None

    drop = max(0, len(original_ids) - max_length)
    ids = original_ids[drop:]
    x = mx.array([ids], dtype=mx.int32)
    logits = model(x)
    log_probs = nn.log_softmax(logits[0])

    context_length = len(context)
    total = 0.0
    count = 0
    for new_index, old_index in enumerate(range(drop, len(original_ids))):
        end = int(encoding.offsets[old_index][1])
        if end <= context_length:
            continue
        if new_index == 0:
            continue
        total += float(log_probs[new_index - 1, ids[new_index]].item())
        count += 1
    if count == 0:
        return None
    return total, count


def _format_progress(acc: BpbAccumulator, examples: int) -> str:
    return f"{acc.tokens:,} tokens, {examples:,} examples, bpb={acc.bpb:.6f}"


def evaluate_language_modeling(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset_dir: Path,
    *,
    name: str,
    text_field: str,
    context_length: int,
    max_tokens: int | None = None,
    max_examples: int | None = None,
    progress_every_tokens: int = 0,
) -> BenchmarkResult:
    """Evaluate BPB over local text rows."""
    acc = BpbAccumulator()
    next_progress = progress_every_tokens

    for row in iter_rows(dataset_dir):
        if max_examples is not None and acc.examples >= max_examples:
            break
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue

        encoding = tokenizer.encode(text, add_special_tokens=False)
        ids = encoding.ids
        if not ids:
            continue

        acc.examples += 1
        start = 0
        while start < len(ids):
            if max_tokens is not None and acc.tokens >= max_tokens:
                break
            end = min(start + context_length, len(ids))
            if max_tokens is not None:
                remaining = max_tokens - acc.tokens
                end = min(end, start + remaining + 1)
            if end - start < 2:
                break

            chunk = ids[start:end]
            chunk_bytes = sum(
                int(encoding.offsets[index][1]) - int(encoding.offsets[index][0])
                for index in range(start, end)
            )
            x = mx.array([chunk], dtype=mx.int32)
            target = mx.array([chunk[1:]], dtype=mx.int32)
            logits = model(x)
            loss_sum = cross_entropy(logits[:, :-1], target, reduction="sum")
            evaluated_tokens = len(chunk) - 1

            acc.nats += cast(float, loss_sum.item())
            acc.tokens += evaluated_tokens
            acc.bytes += chunk_bytes

            if progress_every_tokens > 0 and acc.tokens >= next_progress:
                print(
                    f"[benchmarks] {name}: {_format_progress(acc, acc.examples)}",
                    file=sys.stderr,
                    flush=True,
                )
                next_progress += progress_every_tokens
            start = end

    if acc.tokens == 0 or acc.bytes == 0:
        return BenchmarkResult(
            name=name,
            kind=LM_KIND,
            status="error",
            error="no tokens or bytes evaluated",
        )

    return BenchmarkResult(
        name=name,
        kind=LM_KIND,
        status="ok",
        metrics={
            "loss": acc.loss,
            "perplexity": acc.perplexity,
            "bits_per_token": acc.bits_per_token,
            "bpb": acc.bpb,
            "tokens": acc.tokens,
            "bytes": acc.bytes,
            "examples": acc.examples,
        },
    )


def evaluate_multiple_choice(
    model: nn.Module,
    tokenizer: Tokenizer,
    dataset_dir: Path,
    *,
    name: str,
    context_length: int,
    max_examples: int | None = None,
    seed: int = 0,
    progress_every: int = 0,
) -> BenchmarkResult:
    """Evaluate zero-shot multiple-choice accuracy."""
    correct = 0
    correct_norm = 0
    examples = 0
    tokens = 0
    next_progress = progress_every

    for row in iter_rows(dataset_dir):
        if max_examples is not None and examples >= max_examples:
            break
        case = mcq_case(name, row, seed=seed)
        if case is None:
            continue

        scores: list[float] = []
        norm_scores: list[float] = []
        example_tokens = 0
        for choice in case.choices:
            result = _continuation_logprob(
                model,
                tokenizer,
                case.context,
                choice,
                max_length=context_length,
            )
            if result is None:
                scores.append(float("-inf"))
                norm_scores.append(float("-inf"))
                continue
            logprob, count = result
            scores.append(logprob)
            norm_scores.append(logprob / count)
            example_tokens += count

        if all(math.isinf(score) for score in scores):
            continue

        examples += 1
        tokens += example_tokens
        if max(range(len(scores)), key=lambda index: scores[index]) == case.label:
            correct += 1
        if max(range(len(norm_scores)), key=lambda index: norm_scores[index]) == case.label:
            correct_norm += 1

        if progress_every > 0 and examples >= next_progress:
            print(
                f"[benchmarks] {name}: {examples:,} examples, acc={100.0 * correct / examples:.2f}",
                file=sys.stderr,
                flush=True,
            )
            next_progress += progress_every

    if examples == 0:
        return BenchmarkResult(
            name=name,
            kind=MCQ_KIND,
            status="error",
            error="no examples evaluated",
        )

    return BenchmarkResult(
        name=name,
        kind=MCQ_KIND,
        status="ok",
        metrics={
            "acc": 100.0 * correct / examples,
            "acc_norm": 100.0 * correct_norm / examples,
            "examples": examples,
            "tokens": tokens,
        },
    )


def evaluate_selected_benchmarks(
    *,
    pretrain_config: PretrainConfig,
    checkpoint: str | Path,
    data_dir: str | Path,
    specs: list[BenchmarkSpec],
    max_tokens: int | None = None,
    max_examples: int | None = None,
    seed: int = 0,
    allow_missing: bool = False,
    progress_every_tokens: int = 0,
) -> BenchmarkScorecard:
    """Evaluate selected local benchmarks and return a scorecard."""
    checkpoint_path = Path(checkpoint)
    weights_path = checkpoint_path / "weights.npz"
    if not weights_path.is_file():
        msg = f"checkpoint weights not found: {weights_path}"
        raise FileNotFoundError(msg)

    data_dir_path = Path(data_dir)
    if not data_dir_path.is_dir():
        msg = f"benchmark data directory not found: {data_dir_path}"
        raise FileNotFoundError(msg)

    model_config = load_config(pretrain_config.model, ModelConfig)
    model = load_model(model_config, checkpoint_path)
    tokenizer = Tokenizer.from_file(pretrain_config.tokenizer)
    context_length = pretrain_config.trainer.seq_len

    results: list[BenchmarkResult] = []
    for spec in specs:
        dest = _dataset_dir(data_dir_path, spec.name)
        if dest is None:
            if allow_missing:
                results.append(
                    BenchmarkResult(
                        name=spec.name,
                        kind=spec.kind,
                        status="missing",
                        error="dataset directory or manifest.json not found",
                    )
                )
                continue
            msg = f"benchmark dataset not found: {data_dir_path / spec.name}"
            raise FileNotFoundError(msg)

        try:
            if spec.kind == LM_KIND:
                if spec.text_field is None:
                    msg = f"language-modeling benchmark missing text field: {spec.name}"
                    raise ValueError(msg)
                result = evaluate_language_modeling(
                    model,
                    tokenizer,
                    dest,
                    name=spec.name,
                    text_field=spec.text_field,
                    context_length=context_length,
                    max_tokens=max_tokens,
                    max_examples=max_examples,
                    progress_every_tokens=progress_every_tokens,
                )
            elif spec.kind == MCQ_KIND:
                result = evaluate_multiple_choice(
                    model,
                    tokenizer,
                    dest,
                    name=spec.name,
                    context_length=context_length,
                    max_examples=max_examples,
                    seed=seed,
                    progress_every=progress_every_tokens,
                )
            else:
                msg = f"unknown benchmark kind: {spec.kind}"
                raise ValueError(msg)
        except Exception as exc:
            if not allow_missing:
                raise
            result = BenchmarkResult(
                name=spec.name,
                kind=spec.kind,
                status="error",
                error=str(exc),
            )
        results.append(result)

    return BenchmarkScorecard(
        format_version=1,
        checkpoint=str(checkpoint_path),
        pretrain_config=str(pretrain_config.model),
        data_dir=str(data_dir_path),
        seed=seed,
        max_tokens=max_tokens,
        max_examples=max_examples,
        results=tuple(results),
    )


def write_scorecard(scorecard: BenchmarkScorecard, output_path: str | Path) -> Path:
    """Write a scorecard as pretty-printed JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scorecard.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
