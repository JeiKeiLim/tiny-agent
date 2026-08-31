---
id: doc-007
title: M2 SFT Tool-Calling Format Decision
type: specification
created_date: '2026-08-31 01:19'
updated_date: '2026-08-31 01:19'
---
# M2 SFT Tool-Calling Format Decision

_Status: Accepted for M2 planning._

## Decision

Kestrel M2 uses one standard logical tool-calling schema and one simple rendered assistant tool-call payload.

The dataset schema is the common OpenAI-style / transformers / TRL-style structure:

- `tools`: JSON Schema tool definitions
- `messages`: explicit roles
- assistant tool calls are represented as `tool_calls`
- tool results are represented as `tool` role messages
- final answers are normal assistant messages

## Logical row shape

```json
{
  "source": "tool_local",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "some_tool",
        "description": "Some description",
        "parameters": {
          "type": "object",
          "properties": {
            "arg": {"type": "string", "description": "Some argument"}
          },
          "required": ["arg"]
        }
      }
    }
  ],
  "messages": [
    {"role": "system", "content": "System instructions and tool policy"},
    {"role": "user", "content": "User task"},
    {
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "some_tool",
            "arguments": {"arg": "value"}
          }
        }
      ]
    },
    {"role": "tool", "name": "some_tool", "content": "Tool result"},
    {"role": "assistant", "content": "Final answer"}
  ]
}
```

Non-tool rows may omit `tools` or use an empty list.

## Rendered M2 format

The chat renderer maps roles to the tokenizer reserved role markers.

For M2, assistant tool calls are rendered as a simple JSON object:

```json
{"name":"some_tool","arguments":{"arg":"value"}}
```

This rendered payload is intentionally simple for a 50M model.

The dataset still stores the standard logical `tool_calls` structure.

## M2 scope

- Single tool call per assistant turn
- Parallel or multi-tool calls are deferred
- Public tool data is normalized into this same schema
- Alternate rendered conventions are not trained in M2
- Tool names and schemas are runtime variables, not hardcoded constants
- Evaluation must include unseen tool names and schemas

## Public tool data

The selected public tool source, `argilla/apigen-function-calling`, is call-only.

Its rows are normalized to:

- `tools`
- system message
- user message
- assistant message with `tool_calls`

They do not require:

- `tool` role messages
- final assistant answer

The local tool slice covers tool-result/final-answer behavior.

## Parser requirements

The M2 parser should:

- extract the assistant rendered JSON payload
- parse `name` and `arguments`
- validate `arguments` against the supplied tool schema
- treat missing JSON, invalid JSON, unknown tool names, and schema-invalid arguments as failures
- support eval metrics for valid-call rate, tool-selection accuracy, and argument accuracy
