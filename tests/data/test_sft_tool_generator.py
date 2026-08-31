from __future__ import annotations

from kestrel.data.sft_chat import validate_arguments
from kestrel.data.sft_schema import SFTRow
from kestrel.data.sft_tool_generator import (
    ToolEvalBreakdown,
    ToolGeneratorConfig,
    ToolTrainBreakdown,
    generate_tool_eval,
    generate_tool_train,
    row_tool_names,
)


def _small_config() -> ToolGeneratorConfig:
    return ToolGeneratorConfig(
        seed=7,
        train=ToolTrainBreakdown(
            direct=2,
            no_tool=2,
            distractor_heavy=2,
            missing_info=2,
            hard_variation=2,
        ),
        eval=ToolEvalBreakdown(seen=2, unseen=2, no_call=2, missing_info=2),
    )


def _all_rows(config: ToolGeneratorConfig) -> list[SFTRow]:
    train = generate_tool_train(config)
    eval_rows = generate_tool_eval(config)
    return [*train.all_rows, *eval_rows.all_rows]


def test_default_train_breakdown_is_locked() -> None:
    config = ToolGeneratorConfig()
    assert config.train.direct == 6_000
    assert config.train.no_tool == 1_500
    assert config.train.distractor_heavy == 1_000
    assert config.train.missing_info == 750
    assert config.train.hard_variation == 750
    total = (
        config.train.direct
        + config.train.no_tool
        + config.train.distractor_heavy
        + config.train.missing_info
        + config.train.hard_variation
    )
    assert total == 10_000


def test_default_train_generation_produces_locked_breakdown() -> None:
    rows = generate_tool_train(ToolGeneratorConfig())
    assert len(rows.direct) == 6_000
    assert len(rows.no_tool) == 1_500
    assert len(rows.distractor_heavy) == 1_000
    assert len(rows.missing_info) == 750
    assert len(rows.hard_variation) == 750
    assert len(rows.all_rows) == 10_000


def test_generation_is_seeded_and_deterministic() -> None:
    config = _small_config()
    first = [row.model_dump(mode="json") for row in _all_rows(config)]
    second = [row.model_dump(mode="json") for row in _all_rows(config)]
    assert first == second


def test_every_row_has_three_to_five_tool_definitions() -> None:
    for row in _all_rows(_small_config()):
        assert 3 <= len(row.tools) <= 5


def test_direct_rows_contain_tool_call_tool_result_and_final_answer() -> None:
    train = generate_tool_train(_small_config())
    for row in [*train.direct, *train.distractor_heavy, *train.hard_variation]:
        assistant_messages = [message for message in row.messages if message.role == "assistant"]
        tool_messages = [message for message in row.messages if message.role == "tool"]
        tool_call_messages = [message for message in assistant_messages if message.tool_calls]
        final_messages = [message for message in assistant_messages if message.content is not None]
        assert len(tool_call_messages) == 1
        assert len(tool_messages) == 1
        assert len(final_messages) == 1
        assert tool_messages[0].name == tool_call_messages[0].tool_calls[0].function.name


def test_no_tool_and_missing_info_rows_do_not_call_tools() -> None:
    train = generate_tool_train(_small_config())
    eval_rows = generate_tool_eval(_small_config())
    for row in [*train.no_tool, *train.missing_info, *eval_rows.no_call, *eval_rows.missing_info]:
        assert all(
            message.tool_calls is None for message in row.messages if message.role == "assistant"
        )
        assert not any(message.role == "tool" for message in row.messages)


def test_unseen_eval_tool_names_are_absent_from_train_rows() -> None:
    config = _small_config()
    train = generate_tool_train(config)
    eval_rows = generate_tool_eval(config)
    assert row_tool_names(train.all_rows).isdisjoint(row_tool_names(eval_rows.unseen))


def test_generated_tool_call_arguments_validate_against_row_schemas() -> None:
    for row in _all_rows(_small_config()):
        tool_by_name = {tool.function.name: tool for tool in row.tools}
        for message in row.messages:
            if message.role != "assistant" or message.tool_calls is None:
                continue
            call = message.tool_calls[0]
            tool = tool_by_name[call.function.name]
            errors = validate_arguments(tool.function.parameters, call.function.arguments)
            assert errors == []


def test_every_generated_row_revalidates_against_sft_schema() -> None:
    for row in _all_rows(_small_config()):
        SFTRow.model_validate(row.model_dump(mode="json"))
