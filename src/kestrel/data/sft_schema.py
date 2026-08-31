"""Strict Pydantic schema for Kestrel M2 SFT rows."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from kestrel.common.config import BaseConfig


class FunctionSpec(BaseConfig):
    """JSON Schema function definition used in tool rows."""

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})

    @model_validator(mode="after")
    def _check_parameters(self) -> Self:
        param_type = self.parameters.get("type")
        if param_type is not None and param_type != "object":
            msg = "tool parameters must be an object schema"
            raise ValueError(msg)
        return self


class ToolDefinition(BaseConfig):
    """OpenAI-style tool definition: ``type=function`` plus function metadata."""

    type: Literal["function"]
    function: FunctionSpec


class ToolCallFunction(BaseConfig):
    """Logical assistant tool-call payload."""

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseConfig):
    """OpenAI-style tool call wrapper."""

    type: Literal["function"]
    function: ToolCallFunction


class SystemMessage(BaseConfig):
    role: Literal["system"]
    content: str


class UserMessage(BaseConfig):
    role: Literal["user"]
    content: str


class AssistantMessage(BaseConfig):
    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def _check_payload(self) -> Self:
        if self.content is None and self.tool_calls is None:
            msg = "assistant message requires content or tool_calls"
            raise ValueError(msg)
        if self.content is not None and self.tool_calls is not None:
            msg = "assistant message cannot have both content and tool_calls"
            raise ValueError(msg)
        if self.tool_calls is not None and len(self.tool_calls) != 1:
            msg = "M2 assistant messages support exactly one tool call"
            raise ValueError(msg)
        return self


class ToolMessage(BaseConfig):
    role: Literal["tool"]
    name: str = Field(min_length=1)
    content: str


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]


class SFTRow(BaseConfig):
    """One logical SFT example: source tag, optional tools, and messages."""

    source: str = Field(min_length=1)
    tools: list[ToolDefinition] = Field(default_factory=list)
    messages: list[Message] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_conversation(self) -> Self:
        if self.messages[0].role not in {"system", "user"}:
            msg = "first message must be system or user"
            raise ValueError(msg)

        tool_names = [tool.function.name for tool in self.tools]
        if len(set(tool_names)) != len(tool_names):
            msg = "duplicate tool names in tools"
            raise ValueError(msg)
        tool_name_set = set(tool_names)

        previous: Message | None = None
        for message in self.messages:
            if message.role == "tool" and (
                not isinstance(previous, AssistantMessage)
                or previous.tool_calls is None
                or not previous.tool_calls
                or previous.tool_calls[0].function.name != message.name
            ):
                msg = "tool message must follow an assistant tool call with the same name"
                raise ValueError(msg)

            if isinstance(message, AssistantMessage) and message.tool_calls:
                call_name = message.tool_calls[0].function.name
                if not self.tools:
                    msg = "assistant tool call requires a non-empty tools list"
                    raise ValueError(msg)
                if call_name not in tool_name_set:
                    msg = f"assistant tool call {call_name!r} is not defined in tools"
                    raise ValueError(msg)

            previous = message

        return self
