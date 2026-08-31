---
id: TASK-007.02
title: Document locked standard tool-calling format for Kestrel SFT
status: Done
assignee: []
created_date: '2026-08-31 00:27'
updated_date: '2026-08-31 01:19'
labels:
  - sft
  - research
  - decision
  - tool-calling
dependencies: []
references:
  - 'https://huggingface.co/docs/trl/dataset_formats#tool-calling'
  - 'https://docs.vllm.ai/en/latest/features/tool_calling/'
parent_task_id: TASK-007
priority: high
ordinal: 39000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Locked target tool-calling format for Kestrel M2 SFT.

Decision:
- Use the common logical message schema used by OpenAI-style / transformers / TRL-style tool-calling data.
- Each SFT row has a tools field containing JSON Schema tool definitions and a messages list with explicit roles.
- Assistant tool calls are represented logically as assistant messages containing tool_calls.
- Tool results are represented as tool role messages.
- Final answers are normal assistant content messages.

Logical row shape:
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "some_tool",
        "description": "...",
        "parameters": {"type": "object", "properties": {}, "required": []}
      }
    }
  ],
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
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
    {"role": "tool", "name": "some_tool", "content": "..."},
    {"role": "assistant", "content": "final answer"}
  ]
}

Rendered M2 format:
- The chat renderer maps roles to the tokenizer reserved role markers.
- For M2, assistant tool calls are rendered as a simple JSON object:
  {"name": "some_tool", "arguments": {"arg": "value"}}
- This rendered payload is intentionally simple for a 50M model.
- The dataset still stores the standard logical tool_calls structure.

M2 scope:
- Single tool call per assistant turn.
- Parallel or multi-tool calls are deferred.
- Public tool data is normalized into this same schema.
- Alternate rendered conventions are not trained in M2.
- Tool names and schemas are runtime variables, not hardcoded constants.
- Evaluation must include unseen tool names and schemas.

Implementation implication:
- The SFT chat template should render from the logical row shape.
- Parsers for eval and later serving should parse the rendered simple JSON payload.
- Public tool normalizers should output the logical row shape, not raw source-specific formats.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document records the locked logical messages/tools/tool_calls schema with explicit roles
- [x] #2 Document records the rendered M2 assistant tool-call payload as simple JSON with name and arguments
- [x] #3 Document states that M2 uses single tool calls per assistant turn and defers parallel/multi-call behavior
- [x] #4 Document states that public tool data is normalized to the Kestrel schema and alternate rendered conventions are excluded from M2
- [x] #5 Document requires unseen tool name/schema evaluation
- [x] #6 Document records that the selected public tool source is call-only and normalizes to assistant tool_calls without requiring tool-role results
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The simple rendered JSON payload is close to Llama-3-style JSON tool calls, but the dataset schema is the standard messages/tools/tool_calls structure. This separation avoids confusing dataset interop with rendered model output.

Public tool normalizer target: argilla/apigen-function-calling. Source rows are call-only and must be normalized into the Kestrel logical schema. The normalizer should parse tools JSON strings, handle both OpenAI-style JSON Schema tools and xLAM-style Python-type parameter maps, parse answers JSON strings, keep only single-call rows, and emit assistant tool_calls with function name and arguments. Rendered assistant tool-call payload remains simple JSON with name and arguments. Public tool rows do not require tool-role messages or final assistant answers in M2.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and filled doc-007 with the accepted M2 tool-calling format: standard logical messages/tools/tool_calls schema, simple JSON rendered payload, single-call M2 scope, and call-only public tool normalization.
<!-- SECTION:FINAL_SUMMARY:END -->
