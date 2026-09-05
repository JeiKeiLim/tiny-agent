"""Download Kestrel pretrain evaluation datasets.

Downloads only the raw data files for the standard evaluation split of each
benchmark. It does not download training splits and does not build the full
Hugging Face dataset cache.

Usage:
    uv run python scripts/download_pretrain_eval_datasets.py --data-dir /path/to/datasets
    uv run python scripts/download_pretrain_eval_datasets.py --data-dir /path --skip-large
    uv run python scripts/download_pretrain_eval_datasets.py --data-dir /path --only hellaswag,piqa
    uv run python scripts/download_pretrain_eval_datasets.py --list
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import truststore

truststore.inject_into_ssl()

from datasets import DatasetBuilder, load_dataset_builder  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo: str
    config: str | None
    split: str
    large: bool = False


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("hellaswag", "Rowan/hellaswag", None, "validation"),
    DatasetSpec("piqa", "baber/piqa", None, "validation"),
    DatasetSpec("arc_easy", "allenai/ai2_arc", "ARC-Easy", "test"),
    DatasetSpec("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "test"),
    DatasetSpec("winogrande", "allenai/winogrande", "winogrande_xl", "validation"),
    DatasetSpec("openbookqa", "allenai/openbookqa", "main", "test"),
    DatasetSpec("mmlu", "cais/mmlu", "all", "test"),
    DatasetSpec("boolq", "aps/super_glue", "boolq", "validation"),
    DatasetSpec("sciq", "allenai/sciq", None, "test"),
    DatasetSpec("lambada", "EleutherAI/lambada_openai", "default", "test"),
    DatasetSpec("wikitext2", "EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", "test"),
    DatasetSpec("wikitext103", "EleutherAI/wikitext_document_level", "wikitext-103-raw-v1", "test"),
    DatasetSpec("c4_en_validation", "allenai/c4", "en", "validation", large=True),
    DatasetSpec("pile_test", "EleutherAI/pile_val_test", "default", "test", large=True),
)


def _dataset_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in DATASETS)


def _parse_only(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    names = {item.strip() for item in raw.split(",") if item.strip()}
    known = set(_dataset_names())
    unknown = names - known
    if unknown:
        raise SystemExit(f"error: unknown dataset name(s): {', '.join(sorted(unknown))}")
    return names


def _selected_specs(skip_large: bool, only: set[str] | None) -> list[DatasetSpec]:
    selected = []
    for spec in DATASETS:
        if skip_large and spec.large:
            continue
        if only is not None and spec.name not in only:
            continue
        selected.append(spec)
    return selected


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _parse_hf_dataset_url(url: str) -> tuple[str, str | None, str]:
    parsed = urlparse(url)
    if parsed.scheme != "hf" or parsed.netloc != "datasets":
        raise ValueError(f"unsupported dataset file URL: {url}")
    path = parsed.path.lstrip("/")
    if "@" in path:
        repo_id, _, remainder = path.partition("@")
        revision, sep, filename = remainder.partition("/")
        if not repo_id or not sep or not filename:
            raise ValueError(f"unsupported dataset file URL: {url}")
        return repo_id, revision or None, filename

    org, sep, remainder = path.partition("/")
    repo, sep, filename = remainder.partition("/")
    if not org or not sep or not repo or not filename:
        raise ValueError(f"unsupported dataset file URL: {url}")
    return f"{org}/{repo}", None, filename


def _evaluation_file_urls(builder: DatasetBuilder, split: str) -> list[str]:
    data_files = builder.config.data_files
    if data_files is None:
        raise ValueError("dataset builder has no data_files metadata")
    for key, value in data_files.items():
        if str(key) != split:
            continue
        urls = [value] if isinstance(value, str) else list(value)
        if not urls:
            raise ValueError(f"no data files found for split {split!r}")
        return urls
    raise ValueError(f"split {split!r} not found in dataset metadata")


def _download_spec(spec: DatasetSpec, data_dir: Path, overwrite: bool) -> Path:
    dest = data_dir / spec.name
    if dest.exists():
        if not overwrite:
            print(f"skip {spec.name}: {dest} already exists")
            return dest
        shutil.rmtree(dest)

    config = f" config={spec.config}" if spec.config is not None else ""
    print(f"downloading {spec.name}: {spec.repo}{config} split={spec.split}")

    dest.mkdir(parents=True)
    builder = load_dataset_builder(
        spec.repo,
        name=spec.config,
        cache_dir=str(dest / ".hf-metadata"),
    )
    urls = _evaluation_file_urls(builder, spec.split)

    downloaded: list[str] = []
    for url in urls:
        repo_id, revision, filename = _parse_hf_dataset_url(url)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=str(dest),
        )
        downloaded.append(filename)

    split_info = builder.info.splits.get(spec.split)
    manifest = {
        "name": spec.name,
        "repo": spec.repo,
        "config": spec.config,
        "split": spec.split,
        "files": downloaded,
        "num_examples": split_info.num_examples if split_info is not None else None,
        "num_bytes": split_info.num_bytes if split_info is not None else None,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    size_mb = _directory_size(dest) / (1024 * 1024)
    print(f"saved {spec.name}: {dest} ({size_mb:.1f} MB, {len(downloaded)} file(s))")
    return dest


def _print_specs(specs: list[DatasetSpec]) -> None:
    for spec in specs:
        config = spec.config if spec.config is not None else "-"
        large = " large" if spec.large else ""
        print(f"{spec.name}: {spec.repo} config={config} split={spec.split}{large}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kestrel pretrain evaluation datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="destination directory for downloaded datasets",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated dataset names to download",
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="skip large language-modeling evaluation sets (C4 validation, Pile test)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing dataset directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print selected datasets without downloading",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list all known datasets and exit",
    )
    args = parser.parse_args()

    if args.list:
        _print_specs(list(DATASETS))
        return

    only = _parse_only(args.only)
    specs = _selected_specs(args.skip_large, only)
    if not specs:
        raise SystemExit("error: no datasets selected")

    if args.dry_run:
        _print_specs(specs)
        return

    if args.data_dir is None:
        parser.error("--data-dir is required unless --dry-run or --list is used")

    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        try:
            _download_spec(spec, data_dir, args.overwrite)
        except Exception as exc:
            raise SystemExit(f"error: failed to download {spec.name}: {exc}") from exc

    print(f"done: {len(specs)} dataset(s) in {data_dir}")


if __name__ == "__main__":
    main()
