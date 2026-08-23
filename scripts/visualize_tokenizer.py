"""Interactive tokenizer explorer (dev tool).

Loads the trained tokenizer artifact and lets you inspect it in a REPL: type
any text to see its tokenization (tokens, ids, byte values, kind), or use
:commands to inspect the vocab, special tokens, and individual ids.

Usage:
    uv run python scripts/visualize_tokenizer.py [--verbose]

Default output is compact: the input as color-blocked token spans with the
token ids on the line below (same colors). --verbose adds the full
token/id/bytes/kind table.

Line editing is handled in-process (Backspace and Delete both work regardless
of the terminal's erase-char setting): Backspace/Delete remove the last char,
Ctrl-U clears the line, Ctrl-C cancels, Ctrl-D exits.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import tty
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

# (background, foreground) pairs, cycled per token so adjacent spans contrast.
# No black background: it is invisible on dark terminals.
_PALETTE = [
    ("\033[41m", "\033[97m"),
    ("\033[42m", "\033[30m"),
    ("\033[43m", "\033[30m"),
    ("\033[44m", "\033[97m"),
    ("\033[45m", "\033[97m"),
    ("\033[46m", "\033[30m"),
    ("\033[47m", "\033[30m"),
    ("\033[100m", "\033[97m"),
    ("\033[101m", "\033[97m"),
    ("\033[102m", "\033[30m"),
    ("\033[103m", "\033[30m"),
    ("\033[104m", "\033[97m"),
    ("\033[105m", "\033[97m"),
    ("\033[106m", "\033[30m"),
    ("\033[107m", "\033[30m"),
]

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
    start: int = 0
    end: int = 0


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

    def _info(self, token: str, token_id: int, start: int = 0, end: int = 0) -> TokenInfo:
        if token in self.specials:
            kind = KIND_SPECIAL
        elif len(token) > 1:
            kind = KIND_MERGED
        else:
            kind = KIND_BYTE
        byte_values = tuple(self._byte_map.get(ch, ord(ch)) for ch in token)
        return TokenInfo(token, token_id, kind, byte_values, start, end)

    def view(self, text: str) -> list[TokenInfo]:
        enc = self.tokenizer.encode(text)
        return [
            self._info(t, i, s, e)
            for t, i, (s, e) in zip(enc.tokens, enc.ids, enc.offsets, strict=True)
        ]

    def view_id(self, token_id: int) -> TokenInfo | None:
        token = self.tokenizer.id_to_token(token_id)
        if token is None:
            return None
        return self._info(token, token_id, 0, len(token))

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


def _span_color(i: int) -> str:
    bg, fg = _PALETTE[i % len(_PALETTE)]
    return bg + fg


def render_spans(text: str, infos: list[TokenInfo]) -> str:
    parts = []
    for i, info in enumerate(infos):
        parts.append(_paint(_span_color(i), text[info.start : info.end]))
    return "".join(parts)


def render_ids(text: str, infos: list[TokenInfo]) -> str:
    parts = []
    for i, info in enumerate(infos):
        width = _disp_width(text[info.start : info.end])
        parts.append(_paint(_span_color(i), _pad(str(info.token_id), width)))
    return "".join(parts)


def render_view(text: str, infos: list[TokenInfo], ok: bool, verbose: bool = False) -> str:
    status = _paint(_GREEN, "round-trip: OK") if ok else _paint(_RED, "round-trip: FAIL")
    unit = "token" if len(infos) == 1 else "tokens"
    header = f"{_paint(_BOLD, repr(_short(text)))} → {len(infos)} {unit} | {status}"
    out = header + "\n" + render_spans(text, infos) + "\n" + render_ids(text, infos)
    if verbose:
        rows = [_row(i) for i in infos]
        colors = [
            [_span_color(i), _CYAN, _GRAY, _KIND_COLOR[info.kind]] for i, info in enumerate(infos)
        ]
        out += "\n" + _table(["token", "id", "bytes", "kind"], rows, colors)
    return out


def _read_prefix(path: Path, n: int) -> str:
    with path.open("rb") as f:
        return f.read(n).decode("utf-8", errors="replace")


def _read_byte(fd: int, timeout: float | None = None) -> bytes:
    if timeout is None:
        return os.read(fd, 1)
    ready, _, _ = select.select([fd], [], [], timeout)
    return os.read(fd, 1) if ready else b""


def _read_escape_seq(fd: int) -> str:
    """Read a full key sequence after an ESC; return it (e.g. '3~' for Delete)."""
    first = _read_byte(fd, 0.05)
    if first not in (b"[", b"O"):
        return ""
    seq = first.decode("ascii")
    while True:
        b = _read_byte(fd, 0.05)
        if not b:
            break
        seq += chr(b[0])
        if 0x40 <= b[0] <= 0x7E:
            break
    return seq


def _erase_last(chars: list[str]) -> None:
    if not chars:
        return
    ch = chars.pop()
    w = _disp_width(ch)
    sys.stdout.write("\b" * w + " " * w + "\b" * w)
    sys.stdout.flush()


def _readline_cbreak(prompt: str) -> str | None:
    """Read one line with manual editing, immune to the terminal's erase char.

    A mismatched terminal erase char makes the driver echo ^? and let stray
    DEL bytes into the line, so editing is done here instead: both 0x7F
    (Delete) and 0x08 (Backspace) remove the last character. Ctrl-U clears
    the line, Ctrl-C cancels, Ctrl-D exits. Returns None on Ctrl-D/EOF.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    chars: list[str] = []
    try:
        tty.setcbreak(fd)
        sys.stdout.write(prompt)
        sys.stdout.flush()
        while True:
            b = os.read(fd, 1)
            if not b:
                return None
            c = b[0]
            if c in (0x0D, 0x0A):
                break
            if c == 0x03:
                raise KeyboardInterrupt
            if c == 0x04:
                return None
            if c in (0x7F, 0x08):
                _erase_last(chars)
            elif c == 0x15:
                if chars:
                    w = sum(_disp_width(ch) for ch in chars)
                    sys.stdout.write("\b" * w + " " * w + "\b" * w)
                    chars.clear()
                    sys.stdout.flush()
            elif c == 0x1B:
                if _read_escape_seq(fd) == "[3~":
                    _erase_last(chars)
            elif c < 0x20:
                continue
            elif c < 0x80:
                chars.append(chr(c))
                sys.stdout.write(chr(c))
                sys.stdout.flush()
            else:
                n = 2 if c < 0xE0 else 3
                raw = bytes([c]) + b"".join(os.read(fd, 1) for _ in range(n))
                ch = raw.decode("utf-8", "replace")
                chars.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()
        sys.stdout.write("\n")
        return "".join(chars)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_line(prompt: str) -> str | None:
    if sys.stdin.isatty():
        return _readline_cbreak(prompt)
    try:
        return input(prompt)
    except EOFError:
        return None


def _command(line: str, view: TokenizerView, verbose: bool) -> bool:
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
            print(render_view(info.token, [info], True, verbose))
    elif cmd == ":token":
        token = parts[1]
        info = view.view_id(view.tokenizer.token_to_id(token) or -1)
        if info is None:
            print(f"{token!r} is not in the vocab")
        else:
            print(render_view(token, [info], True, verbose))
    elif cmd == ":file":
        n = int(parts[2]) if len(parts) > 2 else 200
        text = _read_prefix(Path(parts[1]), n)
        print(render_view(text, view.view(text), view.roundtrip_ok(text), verbose))
    else:
        print(f"unknown command {cmd!r} (:help for the list)")
    return False


def main() -> None:
    global _COLOR
    parser = argparse.ArgumentParser(description="Interactively explore the trained tokenizer.")
    parser.add_argument("--config", default="configs/tokenizer/train.yaml")
    parser.add_argument("--verbose", action="store_true", help="also print the full token table")
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
            line = _read_line("tok> ")
        except KeyboardInterrupt:
            print()
            return
        if line is None:
            return
        line = line.strip().replace("\x7f", "")
        if not line:
            continue
        if line.startswith(":"):
            try:
                if _command(line, view, args.verbose):
                    return
            except (ValueError, IndexError, OSError) as e:
                print(f"error in {line!r}: {e}")
            continue
        infos = view.view(line)
        print(render_view(line, infos, view.roundtrip_ok(line), args.verbose))


if __name__ == "__main__":
    main()
