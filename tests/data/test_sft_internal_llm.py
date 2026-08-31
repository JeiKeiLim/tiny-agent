from __future__ import annotations

import importlib.util
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from kestrel.data.sft_internal_llm import (
    InternalLLMConfig,
    OpenAICompatibleClient,
    _normalize_math_final_answer,
    create_llm_client,
    generate_internal_llm_rows,
)
from kestrel.data.sft_prepare import SFTDataConfig, SourceManifest
from kestrel.data.sft_schema import SFTRow

_FINAL_ANSWER = re.compile(r"Final answer: -?\d+(?:\.\d+)?\Z")


def _extract_tool_schema(prompt: str) -> dict[str, Any]:
    lines = prompt.splitlines()
    for index, line in enumerate(lines):
        if line == "Tool schema:" and index + 1 < len(lines):
            return json.loads(lines[index + 1])
    msg = "tool schema not found in prompt"
    raise AssertionError(msg)


def _valid_arguments(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    arguments: dict[str, Any] = {}
    for key in required:
        prop = properties.get(key, {})
        if "enum" in prop:
            arguments[key] = prop["enum"][0]
        elif prop.get("type") == "integer":
            arguments[key] = 1
        elif prop.get("type") == "number":
            arguments[key] = 1.5
        elif prop.get("type") == "array":
            arguments[key] = ["value"]
        else:
            arguments[key] = f"value-{key}"
    return arguments


class FakeLLMClient:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        call = self.calls
        self.calls += 1

        if self._responses is not None:
            if call < len(self._responses):
                return self._responses[call]
            return "invalid"

        variation_match = re.search(r"Variation: (\d+)", prompt)
        variation = int(variation_match.group(1)) if variation_match else call
        if "synthetic assistant training example" in prompt:
            return json.dumps(
                {
                    "prompt": f"Assistant request {variation}",
                    "response": f"Assistant response {variation}",
                }
            )
        if "grade-school math word problem" in prompt:
            return json.dumps(
                {
                    "problem": f"Math problem {variation}",
                    "solution": f"Math solution {variation}",
                    "final_answer": str(variation),
                }
            )
        schema = _extract_tool_schema(prompt)
        return json.dumps(
            {
                "user_prompt": f"Tool request {variation}",
                "arguments": _valid_arguments(schema["function"]["parameters"]),
                "tool_result": json.dumps({"ok": True, "variation": variation}),
                "final_answer": f"Tool answer {variation}",
            }
        )


class RaisingLLMClient:
    def complete(self, prompt: str) -> str:
        msg = "network unavailable"
        raise RuntimeError(msg)


class BarrierFakeClient:
    def __init__(self, parties: int) -> None:
        self._barrier = threading.Barrier(parties)
        self._lock = threading.Lock()
        self._active = 0
        self.peak = 0

    def complete(self, prompt: str) -> str:
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
        self._barrier.wait(timeout=2.0)
        with self._lock:
            self._active -= 1
        variation_match = re.search(r"Variation: (\d+)", prompt)
        variation = int(variation_match.group(1)) if variation_match else 0
        return json.dumps(
            {
                "prompt": f"Assistant request {variation}",
                "response": f"Assistant response {variation}",
            }
        )


def _is_math_row(row: SFTRow) -> bool:
    if row.tools or len(row.messages) != 2:
        return False
    assistant = row.messages[1]
    return (
        assistant.role == "assistant"
        and assistant.content is not None
        and bool(_FINAL_ANSWER.search(assistant.content))
    )


def test_internal_llm_config_requires_env_names_when_enabled() -> None:
    with pytest.raises(ValidationError):
        InternalLLMConfig(enabled=True, api_base_env="")
    with pytest.raises(ValidationError):
        InternalLLMConfig(enabled=True, api_key_env="   ")
    with pytest.raises(ValidationError):
        InternalLLMConfig(enabled=True, model_env="")
    InternalLLMConfig(enabled=False, api_base_env="", api_key_env="", model_env="")


def test_create_llm_client_rejects_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("KESTREL_LLM_API_BASE", "KESTREL_LLM_API_KEY", "KESTREL_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    config = InternalLLMConfig(enabled=True)

    with pytest.raises(ValueError, match="KESTREL_LLM_API_BASE"):
        create_llm_client(config)


def test_create_llm_client_reads_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KESTREL_LLM_API_BASE", "https://example.local/v1")
    monkeypatch.setenv("KESTREL_LLM_API_KEY", "test-key")
    monkeypatch.setenv("KESTREL_LLM_MODEL", "test-model")

    client = create_llm_client(InternalLLMConfig(enabled=True))

    assert isinstance(client, OpenAICompatibleClient)
    assert client.api_base == "https://example.local/v1"
    assert client.api_key == "test-key"
    assert client.model == "test-model"


def test_mocked_generation_produces_locked_split() -> None:
    config = InternalLLMConfig(enabled=True)
    client = FakeLLMClient()

    rows, counts = generate_internal_llm_rows(config, client, seed=7)

    assert counts == {"assistant": 2_000, "math": 2_000, "tool": 1_000}
    assert len(rows) == 5_000
    assert client.calls == 5_000

    math_rows = [row for row in rows if _is_math_row(row)]
    tool_rows = [row for row in rows if row.tools]
    assistant_rows = [row for row in rows if not row.tools and not _is_math_row(row)]
    assert len(assistant_rows) == 2_000
    assert len(math_rows) == 2_000
    assert len(tool_rows) == 1_000

    for row in rows:
        assert row.source == "internal_llm"
        SFTRow.model_validate(row.model_dump(mode="json"))

    tool_row = tool_rows[0]
    assert [message.role for message in tool_row.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert tool_row.messages[2].tool_calls is not None
    assert tool_row.messages[2].tool_calls[0].function.name == tool_row.tools[0].function.name


def test_generator_drops_invalid_and_duplicate_responses() -> None:
    config = InternalLLMConfig(enabled=True, assistant_rows=2, math_rows=0, tool_rows=0)
    responses = [
        "not json",
        json.dumps({"prompt": "Same prompt", "response": "one"}),
        json.dumps({"prompt": "Same prompt", "response": "two"}),
        json.dumps({"prompt": "Different prompt", "response": "three"}),
    ]

    rows, counts = generate_internal_llm_rows(config, FakeLLMClient(responses), seed=1)

    assert counts["assistant"] == 2
    assert [row.messages[0].content for row in rows] == ["Same prompt", "Different prompt"]


def test_generator_drops_reserved_token_text() -> None:
    config = InternalLLMConfig(enabled=True, assistant_rows=1, math_rows=0, tool_rows=0)
    responses = [
        json.dumps({"prompt": "hello", "response": "bad im_end"}),
        json.dumps({"prompt": "hello", "response": "good"}),
    ]

    rows, counts = generate_internal_llm_rows(config, FakeLLMClient(responses), seed=1)

    assert counts["assistant"] == 1
    assert rows[0].messages[1].content == "good"


def test_normalize_math_final_answer_accepts_numeric_json_values() -> None:
    assert _normalize_math_final_answer(8) == "8"
    assert _normalize_math_final_answer(8.0) == "8"
    assert _normalize_math_final_answer(2.55) == "2.55"
    assert _normalize_math_final_answer(" 12 ") == "12"
    assert _normalize_math_final_answer("1,000") == "1000"
    assert _normalize_math_final_answer("twelve") is None
    assert _normalize_math_final_answer(True) is None
    assert _normalize_math_final_answer(None) is None


def test_math_generator_accepts_numeric_json_final_answers() -> None:
    config = InternalLLMConfig(
        enabled=True,
        assistant_rows=0,
        math_rows=2,
        tool_rows=0,
        max_attempts_per_row=1,
    )
    responses = [
        json.dumps({"problem": "p1", "solution": "s1", "final_answer": 12}),
        json.dumps({"problem": "p2", "solution": "s2", "final_answer": 3.5}),
    ]

    rows, counts = generate_internal_llm_rows(config, FakeLLMClient(responses), seed=1)

    assert counts["math"] == 2
    assert rows[0].messages[1].content == "s1\n\nFinal answer: 12"
    assert rows[1].messages[1].content == "s2\n\nFinal answer: 3.5"


def test_math_generator_requires_numeric_final_answer() -> None:
    config = InternalLLMConfig(enabled=True, assistant_rows=0, math_rows=1, tool_rows=0)
    responses = [
        json.dumps({"problem": "p", "solution": "s", "final_answer": "twelve"}),
        json.dumps({"problem": "p", "solution": "s", "final_answer": "12"}),
    ]

    rows, counts = generate_internal_llm_rows(config, FakeLLMClient(responses), seed=1)

    assert counts["math"] == 1
    assert rows[0].messages[1].content == "s\n\nFinal answer: 12"


def test_tool_generator_rejects_invalid_arguments() -> None:
    config = InternalLLMConfig(enabled=True, assistant_rows=0, math_rows=0, tool_rows=1)
    invalid = json.dumps(
        {
            "user_prompt": "p",
            "arguments": {},
            "tool_result": "{}",
            "final_answer": "a",
        }
    )

    rows, counts = generate_internal_llm_rows(
        config, FakeLLMClient([invalid, invalid, invalid]), seed=1
    )

    assert counts["tool"] == 0
    assert rows == []


def test_progress_callback_reports_accepted_rows() -> None:
    config = InternalLLMConfig(
        enabled=True,
        assistant_rows=2,
        math_rows=1,
        tool_rows=1,
        progress_every=2,
    )
    updates: list[dict[str, int]] = []

    rows, counts = generate_internal_llm_rows(
        config, FakeLLMClient(), seed=7, progress_callback=updates.append
    )

    assert counts == {"assistant": 2, "math": 1, "tool": 1}
    assert len(rows) == 4
    assert updates[-1] == {"assistant": 2, "math": 1, "tool": 1}
    assert {"assistant": 2, "math": 0, "tool": 0} in updates


def test_progress_callback_can_be_disabled() -> None:
    config = InternalLLMConfig(
        enabled=True,
        assistant_rows=1,
        math_rows=0,
        tool_rows=0,
        progress_every=0,
    )
    updates: list[dict[str, int]] = []

    generate_internal_llm_rows(config, FakeLLMClient(), seed=7, progress_callback=updates.append)

    assert updates == []


def test_generator_uses_concurrent_clients() -> None:
    config = InternalLLMConfig(
        enabled=True,
        assistant_rows=4,
        math_rows=0,
        tool_rows=0,
        max_workers=4,
        progress_every=0,
    )
    client = BarrierFakeClient(4)

    rows, counts = generate_internal_llm_rows(config, client, seed=7)

    assert counts["assistant"] == 4
    assert len(rows) == 4
    assert client.peak > 1


def test_prepare_internal_llm_reports_progress_to_stderr(
    tmp_path: Path,
    tiny_sft_tokenizer: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import kestrel.data.sft_prepare as module

    monkeypatch.setattr(module, "create_llm_client", lambda config: FakeLLMClient())
    config = SFTDataConfig(
        output_dir=str(tmp_path),
        tokenizer_path=str(tiny_sft_tokenizer),
        context_length=2048,
        seed=7,
        internal_llm=InternalLLMConfig(
            enabled=True,
            assistant_rows=1,
            math_rows=1,
            tool_rows=1,
            progress_every=1,
        ),
    )

    module.prepare_internal_llm(config)

    assert "internal_llm: assistant 1/1, math 1/1, tool 1/1" in capsys.readouterr().err


def test_debug_callback_reports_math_drop_reason() -> None:
    config = InternalLLMConfig(
        enabled=True,
        assistant_rows=0,
        math_rows=1,
        tool_rows=0,
        max_attempts_per_row=1,
        debug_drop_limit=5,
    )
    responses = [
        json.dumps({"problem": "p", "solution": "s", "final_answer": "twelve"}),
    ]
    debug: list[tuple[str, str, str]] = []

    generate_internal_llm_rows(
        config,
        FakeLLMClient(responses),
        seed=1,
        debug_callback=lambda category, reason, detail: debug.append((category, reason, detail)),
    )

    assert debug == [("math", "invalid_final_answer", "got 'twelve'")]


def test_debug_callback_reports_client_errors() -> None:
    config = InternalLLMConfig(
        enabled=True,
        assistant_rows=0,
        math_rows=1,
        tool_rows=0,
        max_attempts_per_row=1,
        debug_drop_limit=5,
    )
    debug: list[tuple[str, str, str]] = []

    generate_internal_llm_rows(
        config,
        RaisingLLMClient(),
        seed=1,
        debug_callback=lambda category, reason, detail: debug.append((category, reason, detail)),
    )

    assert debug == [("math", "client_error", "")]


def test_debug_callback_respects_drop_limit() -> None:
    config = InternalLLMConfig(
        enabled=True,
        assistant_rows=2,
        math_rows=0,
        tool_rows=0,
        debug_drop_limit=1,
    )
    responses = ["invalid", "invalid", "invalid", "invalid"]
    debug: list[tuple[str, str, str]] = []

    generate_internal_llm_rows(
        config,
        FakeLLMClient(responses),
        seed=1,
        debug_callback=lambda category, reason, detail: debug.append((category, reason, detail)),
    )

    assert len(debug) == 1
    assert debug[0][0] == "assistant"
    assert debug[0][1] == "invalid_json"


def test_prepare_internal_llm_reports_drops_to_stderr(
    tmp_path: Path,
    tiny_sft_tokenizer: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import kestrel.data.sft_prepare as module

    responses = [
        json.dumps({"problem": "p", "solution": "s", "final_answer": "twelve"}),
    ]
    monkeypatch.setattr(module, "create_llm_client", lambda config: FakeLLMClient(responses))
    config = SFTDataConfig(
        output_dir=str(tmp_path),
        tokenizer_path=str(tiny_sft_tokenizer),
        context_length=2048,
        seed=7,
        internal_llm=InternalLLMConfig(
            enabled=True,
            assistant_rows=0,
            math_rows=1,
            tool_rows=0,
            debug_drops=True,
            debug_drop_limit=5,
        ),
    )

    module.prepare_internal_llm(config)

    assert "internal_llm debug: math dropped reason=invalid_final_answer" in (
        capsys.readouterr().err
    )


def test_client_errors_are_dropped() -> None:
    config = InternalLLMConfig(enabled=True, assistant_rows=1, math_rows=0, tool_rows=0)

    rows, counts = generate_internal_llm_rows(config, RaisingLLMClient(), seed=1)

    assert counts["assistant"] == 0
    assert rows == []


def test_prepare_internal_llm_writes_manifest(
    tmp_path: Path, tiny_sft_tokenizer: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kestrel.data.sft_prepare as module

    monkeypatch.setattr(module, "create_llm_client", lambda config: FakeLLMClient())
    config = SFTDataConfig(
        output_dir=str(tmp_path),
        tokenizer_path=str(tiny_sft_tokenizer),
        context_length=2048,
        seed=7,
        internal_llm=InternalLLMConfig(
            enabled=True,
            assistant_rows=2,
            math_rows=2,
            tool_rows=1,
            prompt_version="test-v1",
        ),
    )

    manifest = module.prepare_internal_llm(config)

    assert manifest is not None
    assert manifest.source == "internal_llm"
    assert manifest.requested_rows == 5
    assert manifest.candidate_rows == 5
    assert manifest.written_rows == 5
    assert manifest.filtered_rows == 0
    assert manifest.dropped_rows == 0
    assert manifest.model_env == "KESTREL_LLM_MODEL"
    assert manifest.prompt_version == "test-v1"
    assert manifest.generated_counts == {"assistant": 2, "math": 2, "tool": 1}
    assert manifest.output_path == str(tmp_path / "internal_llm.jsonl")

    rows = [
        json.loads(line)
        for line in (tmp_path / "internal_llm.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 5
    assert all(row["source"] == "internal_llm" for row in rows)

    manifest_data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["internal_llm"]["model_env"] == "KESTREL_LLM_MODEL"
    assert manifest_data["internal_llm"]["prompt_version"] == "test-v1"
    assert manifest_data["internal_llm"]["generated_counts"] == {
        "assistant": 2,
        "math": 2,
        "tool": 1,
    }


def test_prepare_internal_llm_is_disabled_by_default(
    tmp_path: Path, tiny_sft_tokenizer: Path
) -> None:
    import kestrel.data.sft_prepare as module

    config = SFTDataConfig(
        output_dir=str(tmp_path),
        tokenizer_path=str(tiny_sft_tokenizer),
        context_length=2048,
        seed=7,
    )

    assert module.prepare_internal_llm(config) is None
    assert not (tmp_path / "internal_llm.jsonl").exists()


def test_prepare_internal_llm_filters_long_rows(
    tmp_path: Path, tiny_sft_tokenizer: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kestrel.data.sft_prepare as module

    monkeypatch.setattr(module, "create_llm_client", lambda config: FakeLLMClient())
    config = SFTDataConfig(
        output_dir=str(tmp_path),
        tokenizer_path=str(tiny_sft_tokenizer),
        context_length=32,
        seed=7,
        internal_llm=InternalLLMConfig(enabled=True, assistant_rows=2, math_rows=0, tool_rows=0),
    )
    long_response = json.dumps({"prompt": "p1", "response": "hello " * 100})
    short_response = json.dumps({"prompt": "p2", "response": "short"})
    monkeypatch.setattr(
        module,
        "create_llm_client",
        lambda config: FakeLLMClient([long_response, short_response]),
    )

    manifest = module.prepare_internal_llm(config)

    assert manifest is not None
    assert manifest.written_rows == 1
    assert manifest.filtered_rows == 1


def test_run_prepare_sft_cli_selects_internal_llm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = importlib.util.spec_from_file_location(
        "kestrel_run_prepare_sft", "scripts/run_prepare_sft.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fake_manifest = SourceManifest(
        source="internal_llm",
        dataset_id="internal-llm:KESTREL_LLM_MODEL",
        split="generated",
        seed=0,
        requested_rows=1,
        candidate_rows=1,
        written_rows=1,
        filtered_rows=0,
        output_path="internal_llm.jsonl",
        sha256="abc123",
        model_env="KESTREL_LLM_MODEL",
        prompt_version="v1",
        generated_counts={"assistant": 1, "math": 0, "tool": 0},
    )
    calls: list[str] = []

    monkeypatch.setattr(module, "load_config", lambda path, config_type: SFTDataConfig())
    monkeypatch.setattr(
        module,
        "prepare_internal_llm",
        lambda config: calls.append("internal_llm") or fake_manifest,
    )
    monkeypatch.setattr(sys, "argv", ["run_prepare_sft.py", "--source", "internal_llm"])

    module.main()

    assert calls == ["internal_llm"]
    assert "internal_llm: wrote 1/1 rows" in capsys.readouterr().out


def test_run_prepare_sft_cli_rejects_disabled_internal_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "kestrel_run_prepare_sft", "scripts/run_prepare_sft.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "load_config", lambda path, config_type: SFTDataConfig())
    monkeypatch.setattr(module, "prepare_internal_llm", lambda config: None)
    monkeypatch.setattr(sys, "argv", ["run_prepare_sft.py", "--source", "internal_llm"])

    with pytest.raises(SystemExit, match="disabled"):
        module.main()
