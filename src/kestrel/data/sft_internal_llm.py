"""Optional internal LLM generator for the M2 SFT data slice."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from random import Random
from typing import Any, Protocol, Self

import truststore
from pydantic import Field, model_validator

from kestrel.common.config import BaseConfig
from kestrel.data.sft_chat import validate_arguments
from kestrel.data.sft_schema import SFTRow, ToolDefinition
from kestrel.data.sft_tool_generator import SYSTEM_PROMPT
from kestrel.tokenizer.config import DEFAULT_SPECIAL_TOKENS
from kestrel.tools.schema_sampler import TRAIN_TOOL_FAMILIES

PROMPT_VERSION = "v1"

_RESERVED_TOKENS = tuple(DEFAULT_SPECIAL_TOKENS)
_FINAL_ANSWER = re.compile(r"^-?\d+(?:\.\d+)?$")
_WHITESPACE = re.compile(r"\s+")

_ASSISTANT_TOPICS = (
    "writing help",
    "explaining a technical concept",
    "planning a small project",
    "debugging advice",
    "summarizing information",
    "travel planning",
    "healthy habits",
    "science explanation",
    "history question",
    "communication advice",
)
_ASSISTANT_STYLES = (
    "concise",
    "friendly",
    "formal",
    "step-by-step",
    "practical",
    "encouraging",
)
_MATH_TOPICS = (
    "shopping",
    "distance and speed",
    "time and schedules",
    "rates",
    "ratios",
    "percentages",
    "age comparisons",
    "work and production",
    "money and change",
    "simple geometry",
)


class LLMClient(Protocol):
    """Minimal completion client used by the internal LLM generator."""

    def complete(self, prompt: str) -> str: ...


class InternalLLMConfig(BaseConfig):
    """Strict settings for optional internal LLM SFT generation."""

    enabled: bool = False
    api_base_env: str = "KESTREL_LLM_API_BASE"
    api_key_env: str = "KESTREL_LLM_API_KEY"
    model_env: str = "KESTREL_LLM_MODEL"
    source: str = "internal_llm"
    prompt_version: str = PROMPT_VERSION
    assistant_rows: int = Field(default=2_000, ge=0)
    math_rows: int = Field(default=2_000, ge=0)
    tool_rows: int = Field(default=1_000, ge=0)
    max_attempts_per_row: int = Field(default=3, gt=0)
    max_workers: int = Field(default=1, ge=1)
    progress_every: int = Field(default=10, ge=0)
    debug_drops: bool = False
    debug_drop_limit: int = Field(default=20, ge=0)
    max_list_items: int = Field(default=10, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1_024, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    assistant_prompt_max_chars: int = Field(default=512, gt=0)
    assistant_response_max_chars: int = Field(default=2_048, gt=0)
    math_problem_max_chars: int = Field(default=512, gt=0)
    math_solution_max_chars: int = Field(default=2_048, gt=0)
    tool_prompt_max_chars: int = Field(default=512, gt=0)
    tool_result_max_chars: int = Field(default=1_024, gt=0)
    tool_answer_max_chars: int = Field(default=1_024, gt=0)

    @model_validator(mode="after")
    def _check_enabled(self) -> Self:
        if self.enabled:
            if not all(
                {
                    self.api_base_env.strip(),
                    self.api_key_env.strip(),
                    self.model_env.strip(),
                }
            ):
                msg = "enabled internal LLM config requires non-empty environment variable names"
                raise ValueError(msg)
            if self.assistant_rows + self.math_rows + self.tool_rows == 0:
                msg = "enabled internal LLM config requires at least one generated row"
                raise ValueError(msg)
        return self


@dataclass(frozen=True)
class OpenAICompatibleClient:
    """Small OpenAI-compatible chat client used during offline data prep."""

    api_base: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0
    temperature: float = 0.7
    max_tokens: int = 1_024

    def complete(self, prompt: str) -> str:
        truststore.inject_into_ssl()
        url = self.api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            data: Any = json.load(response)

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            msg = "LLM response is missing choices"
            raise RuntimeError(msg)
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            msg = "LLM response is missing message content"
            raise RuntimeError(msg)
        return content


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        msg = f"missing required environment variable {name}"
        raise ValueError(msg)
    return value


def create_llm_client(config: InternalLLMConfig) -> LLMClient:
    """Build the production LLM client from environment variable values."""
    return OpenAICompatibleClient(
        api_base=_require_env(config.api_base_env),
        api_key=_require_env(config.api_key_env),
        model=_require_env(config.model_env),
        timeout_seconds=config.timeout_seconds,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _is_safe_text(value: str) -> bool:
    return bool(value.strip()) and not any(token in value for token in _RESERVED_TOKENS)


def _normalize_key(value: str) -> str:
    return _WHITESPACE.sub(" ", value.lower()).strip()


def _normalize_math_final_answer(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if _FINAL_ANSWER.fullmatch(cleaned):
            return cleaned
        comma_free = cleaned.replace(",", "")
        if _FINAL_ANSWER.fullmatch(comma_free):
            return comma_free
    return None


def _is_flat_value(value: Any, max_list_items: int) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return len(value) <= max_list_items and all(
            item is None or isinstance(item, str | int | float | bool) for item in value
        )
    return False


def _is_flat_arguments(arguments: dict[str, Any], max_list_items: int) -> bool:
    return all(_is_flat_value(value, max_list_items) for value in arguments.values())


def _user_key(row: SFTRow) -> str:
    for message in row.messages:
        if message.role == "user":
            return _normalize_key(message.content)
    return ""


@dataclass(frozen=True)
class _ConversionOutcome:
    row: SFTRow | None
    reason: str = ""
    detail: str = ""


def _ok(row: SFTRow) -> _ConversionOutcome:
    return _ConversionOutcome(row)


def _drop(reason: str, detail: str = "") -> _ConversionOutcome:
    return _ConversionOutcome(None, reason, detail)


def _debug_detail(value: str, limit: int = 200) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


class InternalLLMGenerator:
    """Generate, validate, and deduplicate internal LLM SFT rows."""

    def __init__(
        self,
        config: InternalLLMConfig,
        client: LLMClient,
        seed: int,
        progress_callback: Callable[[dict[str, int]], None] | None = None,
        debug_callback: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._assistant_rng = Random(seed)
        self._math_rng = Random(seed + 1)
        self._tool_rng = Random(seed + 2)
        self._progress_callback = progress_callback
        self._debug_callback = debug_callback

    def generate(self) -> tuple[list[SFTRow], dict[str, int]]:
        if self._config.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self._config.max_workers) as executor:
                return self._generate_all(executor)
        return self._generate_all(None)

    def _generate_all(
        self, executor: ThreadPoolExecutor | None
    ) -> tuple[list[SFTRow], dict[str, int]]:
        def report(assistant: int, math: int, tool: int, force: bool = False) -> None:
            if self._progress_callback is None or self._config.progress_every == 0:
                return
            total = assistant + math + tool
            if not force and total % self._config.progress_every != 0:
                return
            self._progress_callback({"assistant": assistant, "math": math, "tool": tool})

        debug_counts: dict[str, int] = {}

        def debug(category: str, reason: str, detail: str = "") -> None:
            if self._debug_callback is None or self._config.debug_drop_limit == 0:
                return
            count = debug_counts.get(category, 0)
            if count >= self._config.debug_drop_limit:
                return
            debug_counts[category] = count + 1
            self._debug_callback(category, reason, detail)

        assistant_rows = self._generate_assistant(
            executor, lambda count: report(count, 0, 0), debug
        )
        report(len(assistant_rows), 0, 0, force=True)
        math_rows = self._generate_math(
            executor, lambda count: report(len(assistant_rows), count, 0), debug
        )
        report(len(assistant_rows), len(math_rows), 0, force=True)
        tool_rows = self._generate_tool(
            executor, lambda count: report(len(assistant_rows), len(math_rows), count), debug
        )
        report(len(assistant_rows), len(math_rows), len(tool_rows), force=True)
        rows = [*assistant_rows, *math_rows, *tool_rows]
        counts = {
            "assistant": len(assistant_rows),
            "math": len(math_rows),
            "tool": len(tool_rows),
        }
        return rows, counts

    def _generate(
        self,
        executor: ThreadPoolExecutor | None,
        category: str,
        target: int,
        attempts: Iterator[tuple[str, Any]],
        convert: Callable[[str, Any], _ConversionOutcome],
        on_progress: Callable[[int], None],
        debug: Callable[[str, str, str], None],
    ) -> list[SFTRow]:
        rows: list[SFTRow] = []
        seen: set[str] = set()
        while len(rows) < target:
            batch_size = min(self._config.max_workers, target - len(rows))
            batch = list(islice(attempts, batch_size))
            if not batch:
                break
            responses = self._complete_batch(executor, batch)
            for (_prompt, context), response in zip(batch, responses, strict=True):
                if len(rows) == target:
                    break
                if response is None:
                    debug(category, "client_error", "")
                    continue
                outcome = convert(response, context)
                if outcome.row is None:
                    debug(category, outcome.reason or "conversion_failed", outcome.detail)
                    continue
                key = _user_key(outcome.row)
                if key in seen:
                    debug(category, "duplicate", key)
                    continue
                seen.add(key)
                rows.append(outcome.row)
                on_progress(len(rows))
        return rows

    def _complete_batch(
        self, executor: ThreadPoolExecutor | None, batch: list[tuple[str, Any]]
    ) -> list[str | None]:
        prompts = [prompt for prompt, _ in batch]
        if executor is None:
            return [self._safe_complete(prompt) for prompt in prompts]
        futures = [executor.submit(self._safe_complete, prompt) for prompt in prompts]
        return [future.result() for future in futures]

    def _safe_complete(self, prompt: str) -> str | None:
        try:
            return self._client.complete(prompt)
        except Exception:
            return None

    def _generate_assistant(
        self,
        executor: ThreadPoolExecutor | None,
        on_progress: Callable[[int], None],
        debug: Callable[[str, str, str], None],
    ) -> list[SFTRow]:
        return self._generate(
            executor,
            "assistant",
            self._config.assistant_rows,
            self._assistant_attempts(),
            self._convert_assistant_response,
            on_progress,
            debug,
        )

    def _assistant_attempts(self) -> Iterator[tuple[str, None]]:
        config = self._config
        max_attempts = config.assistant_rows * config.max_attempts_per_row
        for attempt in range(1, max_attempts + 1):
            topic = self._assistant_rng.choice(_ASSISTANT_TOPICS)
            style = self._assistant_rng.choice(_ASSISTANT_STYLES)
            prompt = (
                "Generate one synthetic assistant training example.\n"
                f"Topic: {topic}\n"
                f"Style: {style}\n"
                f"Variation: {attempt}\n"
                'Return only a JSON object with fields "prompt" and "response".\n'
                "The prompt must be a realistic user request.\n"
                "The response must be helpful and self-contained."
            )
            yield prompt, None

    def _convert_assistant_response(self, response: str, _context: Any) -> _ConversionOutcome:
        config = self._config
        data = _parse_json_object(response)
        if data is None:
            return _drop("invalid_json", _debug_detail(response))
        user_content = data.get("prompt")
        assistant_content = data.get("response")
        if not isinstance(user_content, str) or not isinstance(assistant_content, str):
            return _drop("missing_fields", _debug_detail(response))
        if len(user_content) > config.assistant_prompt_max_chars:
            return _drop("prompt_too_long", str(len(user_content)))
        if len(assistant_content) > config.assistant_response_max_chars:
            return _drop("response_too_long", str(len(assistant_content)))
        if not _is_safe_text(user_content) or not _is_safe_text(assistant_content):
            return _drop("reserved_token", _debug_detail(response))
        try:
            return _ok(
                SFTRow.model_validate(
                    {
                        "source": config.source,
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content},
                        ],
                    }
                )
            )
        except Exception as exc:
            return _drop("schema_validation_failed", str(exc))

    def _generate_math(
        self,
        executor: ThreadPoolExecutor | None,
        on_progress: Callable[[int], None],
        debug: Callable[[str, str, str], None],
    ) -> list[SFTRow]:
        return self._generate(
            executor,
            "math",
            self._config.math_rows,
            self._math_attempts(),
            self._convert_math_response,
            on_progress,
            debug,
        )

    def _math_attempts(self) -> Iterator[tuple[str, None]]:
        config = self._config
        max_attempts = config.math_rows * config.max_attempts_per_row
        for attempt in range(1, max_attempts + 1):
            topic = self._math_rng.choice(_MATH_TOPICS)
            prompt = (
                "Generate one grade-school math word problem.\n"
                f"Topic: {topic}\n"
                f"Variation: {attempt}\n"
                'Return only a JSON object with fields "problem", "solution", and "final_answer".\n'
                "The solution must explain the steps briefly.\n"
                "The final_answer must be a plain integer or decimal number."
            )
            yield prompt, None

    def _convert_math_response(self, response: str, _context: Any) -> _ConversionOutcome:
        config = self._config
        data = _parse_json_object(response)
        if data is None:
            return _drop("invalid_json", _debug_detail(response))
        problem = data.get("problem")
        solution = data.get("solution")
        final_answer_raw = data.get("final_answer")
        final_answer = _normalize_math_final_answer(final_answer_raw)
        if not isinstance(problem, str) or not isinstance(solution, str):
            return _drop("missing_fields", _debug_detail(response))
        if final_answer is None:
            return _drop("invalid_final_answer", f"got {final_answer_raw!r}")
        if len(problem) > config.math_problem_max_chars:
            return _drop("problem_too_long", str(len(problem)))
        if len(solution) > config.math_solution_max_chars:
            return _drop("solution_too_long", str(len(solution)))
        if (
            not _is_safe_text(problem)
            or not _is_safe_text(solution)
            or not _is_safe_text(final_answer)
        ):
            return _drop("reserved_token", _debug_detail(response))
        assistant_content = f"{solution}\n\nFinal answer: {final_answer}"
        try:
            return _ok(
                SFTRow.model_validate(
                    {
                        "source": config.source,
                        "messages": [
                            {"role": "user", "content": problem},
                            {"role": "assistant", "content": assistant_content},
                        ],
                    }
                )
            )
        except Exception as exc:
            return _drop("schema_validation_failed", str(exc))

    def _generate_tool(
        self,
        executor: ThreadPoolExecutor | None,
        on_progress: Callable[[int], None],
        debug: Callable[[str, str, str], None],
    ) -> list[SFTRow]:
        return self._generate(
            executor,
            "tool",
            self._config.tool_rows,
            self._tool_attempts(),
            self._convert_tool_response,
            on_progress,
            debug,
        )

    def _tool_attempts(self) -> Iterator[tuple[str, ToolDefinition]]:
        config = self._config
        max_attempts = config.tool_rows * config.max_attempts_per_row
        for attempt in range(1, max_attempts + 1):
            family = self._tool_rng.choice(TRAIN_TOOL_FAMILIES)
            definition = family.sample_definition(self._tool_rng)
            schema_json = json.dumps(definition.model_dump(mode="json"), ensure_ascii=False)
            prompt = (
                "Generate one tool-calling training example for the exact tool schema below.\n"
                "Tool schema:\n"
                f"{schema_json}\n"
                f"Variation: {attempt}\n"
                'Return only a JSON object with fields "user_prompt", "arguments", '
                '"tool_result", and "final_answer".\n'
                "arguments must match the schema and use flat values only.\n"
                "tool_result must be a JSON-encoded string.\n"
                "final_answer must be a short assistant answer using the tool_result."
            )
            yield prompt, definition

    def _convert_tool_response(self, response: str, context: Any) -> _ConversionOutcome:
        config = self._config
        definition: ToolDefinition = context
        data = _parse_json_object(response)
        if data is None:
            return _drop("invalid_json", _debug_detail(response))
        user_prompt = data.get("user_prompt")
        arguments = data.get("arguments")
        tool_result = data.get("tool_result")
        final_answer = data.get("final_answer")
        if (
            not isinstance(user_prompt, str)
            or not isinstance(arguments, dict)
            or not isinstance(tool_result, str)
            or not isinstance(final_answer, str)
        ):
            return _drop("missing_fields", _debug_detail(response))
        if len(user_prompt) > config.tool_prompt_max_chars:
            return _drop("prompt_too_long", str(len(user_prompt)))
        if len(tool_result) > config.tool_result_max_chars:
            return _drop("tool_result_too_long", str(len(tool_result)))
        if len(final_answer) > config.tool_answer_max_chars:
            return _drop("final_answer_too_long", str(len(final_answer)))
        if not (
            _is_safe_text(user_prompt)
            and _is_safe_text(tool_result)
            and _is_safe_text(final_answer)
        ):
            return _drop("reserved_token", _debug_detail(response))
        if not _is_flat_arguments(arguments, config.max_list_items):
            return _drop("non_flat_arguments", _debug_detail(json.dumps(arguments)))
        schema_errors = validate_arguments(definition.function.parameters, arguments)
        if schema_errors:
            return _drop("schema_invalid_arguments", "; ".join(schema_errors))
        try:
            json.loads(tool_result)
        except json.JSONDecodeError:
            return _drop("invalid_tool_result_json", _debug_detail(tool_result))
        name = definition.function.name
        try:
            return _ok(
                SFTRow.model_validate(
                    {
                        "source": config.source,
                        "tools": [definition.model_dump(mode="json")],
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {"name": name, "arguments": arguments},
                                    }
                                ],
                            },
                            {"role": "tool", "name": name, "content": tool_result},
                            {"role": "assistant", "content": final_answer},
                        ],
                    }
                )
            )
        except Exception as exc:
            return _drop("schema_validation_failed", str(exc))


def generate_internal_llm_rows(
    config: InternalLLMConfig,
    client: LLMClient,
    seed: int,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    debug_callback: Callable[[str, str, str], None] | None = None,
) -> tuple[list[SFTRow], dict[str, int]]:
    """Generate the internal LLM assistant, math, and tool rows."""
    return InternalLLMGenerator(config, client, seed, progress_callback, debug_callback).generate()
