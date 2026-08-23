"""Interactive tokenizer explorer (dev tool).

Loads the trained tokenizer artifact and lets you inspect it in a REPL: type
any text to see its tokenization (tokens, ids, byte values, kind), or use
:commands to inspect the vocab, special tokens, and individual ids.

Usage:
    uv run python scripts/visualize_tokenizer.py
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from kestrel.common.config import load_config
from kestrel.tokenizer.config import TokenizerConfig

KIND_SPECIAL = "special"
KIND_MERGED = "merged"
KIND_BYTE = "byte"

_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_GRAY = "\033[90m"

_KIND_COLOR = {KIND_SPECIAL: _BOLD + _MAGENTA, KIND_MERGED: _GREEN, KIND_BYTE: _GRAY}

_COLOR = True

_HELP = """commands:
  <text>            tokenize and visualize the text
  :specials         list special tokens with their ids
  :vocab [n]        vocab size + first n tokens (id order = merge frequency)
  :id <n>           show the token for id n
  :token <s>        show the id for token string s
  :file <path> [n]  tokenize the first n chars of a file (default 200)
  :help             this help
  :quit             exit (Ctrl-D also works)
"""


@dataclass(frozen=True)
class TokenInfo:
    token: str
    token_id: int
    kind: str
    byte_values: tuple[int, ...]


def build_byte_map(tokenizer: Tokenizer) -> dict[str, int]:
    """Map each byte-alphabet char (single-char vocab token) to its byte value."""
    single = [(t, i) for t, i in tokenizer.get_vocab().items() if len(t) == 1]
    decoded = tokenizer.decode_batch([[i] for _, i in single], skip_special_tokens=False)
    return {
        t: d.encode("utf-8")[0] for (t, _), d in zip(single, decoded, strict=True) if len(d) == 1
    }


class TokenizerView:
    """Precomputed lookup state for visualizing one tokenizer artifact."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.specials = frozenset(
            t.content for t in tokenizer.get_added_tokens_decoder().values() if t.special
        )
        self._byte_map = build_byte_map(tokenizer)

    def _info(self, token: str, token_id: int) -> TokenInfo:
        if token in self.specials:
            kind = KIND_SPECIAL
        elif len(token) > 1:
            kind = KIND_MERGED
        else:
            kind = KIND_BYTE
        byte_values = tuple(self._byte_map.get(ch, ord(ch)) for ch in token)
        return TokenInfo(token, token_id, kind, byte_values)

    def view(self, text: str) -> list[TokenInfo]:
        enc = self.tokenizer.encode(text)
        return [self._info(t, i) for t, i in zip(enc.tokens, enc.ids, strict=True)]

    def view_id(self, token_id: int) -> TokenInfo | None:
        token = self.tokenizer.id_to_token(token_id)
        if token is None:
            return None
        return self._info(token, token_id)

    def roundtrip_ok(self, text: str) -> bool:
        return self.tokenizer.decode(self.tokenizer.encode(text).ids) == text


def _paint(color: str, s: str) -> str:
    return f"{color}{s}{_RESET}" if _COLOR and color else s


def _disp_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _disp_width(s))


def _short(s: str, limit: int = 60) -> str:
    return s if len(s) <= limit else s[:limit] + "…"


def _table(
    headers: list[str], rows: list[list[str]], cell_colors: list[list[str]] | None = None
) -> str:
    widths = [
        max([_disp_width(h)] + [_disp_width(r[i]) for r in rows]) for i, h in enumerate(headers)
    ]

    def body(cells: list[str], colors: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            color = colors[i] if i < len(colors) else ""
            parts.append(_paint(color, _pad(cell, widths[i])))
        return " │ ".join(parts)

    def rule(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right

    lines = [
        rule("┌", "┬", "┐"),
        _paint(_BOLD, body(headers, [""] * len(headers))),
        rule("├", "┼", "┤"),
    ]
    for row, colors in zip(rows, cell_colors or [[""] * len(headers) for _ in rows], strict=True):
        lines.append(body(row, colors))
    lines.append(rule("└", "┴", "┘"))
    return "\n".join(lines)


def _row(info: TokenInfo) -> list[str]:
    return [
        info.token,
        str(info.token_id),
        " ".join(f"{b:02x}" for b in info.byte_values),
        info.kind,
    ]


def render_view(text: str, infos: list[TokenInfo], ok: bool) -> str:
    rows = [_row(i) for i in infos]
    colors = [[_KIND_COLOR[i.kind], _CYAN, _GRAY, _KIND_COLOR[i.kind]] for i in infos]
    status = _paint(_GREEN, "round-trip: OK") if ok else _paint(_RED, "round-trip: FAIL")
    unit = "token" if len(infos) == 1 else "tokens"
    header = f"{_paint(_BOLD, repr(_short(text)))} → {len(infos)} {unit} | {status}"
    return header + "\n" + _table(["token", "id", "bytes", "kind"], rows, colors)


def _read_prefix(path: Path, n: int) -> str:
    with path.open("rb") as f:
        return f.read(n).decode("utf-8", errors="replace")


def _command(line: str, view: TokenizerView) -> bool:
    parts = line.split()
    cmd = parts[0]
    if cmd in (":q", ":quit"):
        return True
    if cmd == ":help":
        print(_HELP)
    elif cmd == ":specials":
        added = view.tokenizer.get_added_tokens_decoder()
        rows = [[at.content, str(tid)] for tid, at in sorted(added.items())]
        print(_table(["token", "id"], rows, [[_MAGENTA, _CYAN] for _ in rows]))
    elif cmd == ":vocab":
        n = int(parts[1]) if len(parts) > 1 else 15
        vocab = view.tokenizer.get_vocab()
        print(f"vocab size: {len(vocab)} (id order = merge frequency)")
        items = sorted(vocab.items(), key=lambda kv: kv[1])[:n]
        rows = [[t, str(i)] for t, i in items]
        print(_table(["token", "id"], rows, [["", _CYAN] for _ in rows]))
    elif cmd == ":id":
        info = view.view_id(int(parts[1]))
        if info is None:
            print("no token for that id")
        else:
            print(render_view(f"id {info.token_id}", [info], True))
    elif cmd == ":token":
        token = parts[1]
        info = view.view_id(view.tokenizer.token_to_id(token) or -1)
        if info is None:
            print(f"{token!r} is not in the vocab")
        else:
            print(render_view(token, [info], True))
    elif cmd == ":file":
        n = int(parts[2]) if len(parts) > 2 else 200
        text = _read_prefix(Path(parts[1]), n)
        print(render_view(text, view.view(text), view.roundtrip_ok(text)))
    else:
        print(f"unknown command {cmd!r} (:help for the list)")
    return False


def main() -> None:
    global _COLOR
    parser = argparse.ArgumentParser(description="Interactively explore the trained tokenizer.")
    parser.add_argument("--config", default="configs/tokenizer/train.yaml")
    args = parser.parse_args()
    config = load_config(args.config, TokenizerConfig)
    artifact = Path(config.output_dir) / "tokenizer.json"
    if not artifact.exists():
        print(f"artifact not found: {artifact}")
        print("train it first: uv run python -m kestrel.tokenizer.train")
        raise SystemExit(1)

    _COLOR = sys.stdout.isatty()
    view = TokenizerView(Tokenizer.from_file(str(artifact)))
    print(f"{_paint(_BOLD, 'Kestrel tokenizer explorer')}")
    print(f"artifact: {artifact}")
    print(f"vocab: {len(view.tokenizer.get_vocab())} tokens | specials: {len(view.specials)}")
    print("type text to tokenize, :help for commands, :quit to exit")
    while True:
        try:
            line = input("tok> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.startswith(":"):
            try:
                if _command(line, view):
                    return
            except (ValueError, IndexError, OSError) as e:
                print(f"error in {line!r}: {e}")
            continue
        infos = view.view(line)
        print(render_view(line, infos, view.roundtrip_ok(line)))


if __name__ == "__main__":
    main()
