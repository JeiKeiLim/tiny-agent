from __future__ import annotations

from random import Random

from kestrel.data.sft_chat import validate_arguments
from kestrel.tools.schema_sampler import (
    TRAIN_TOOL_FAMILIES,
    UNSEEN_TOOL_FAMILIES,
    ToolFamilySpec,
)

ALLOWED_PRIMITIVE_TYPES = {"string", "integer", "number", "boolean"}


def _assert_flat_schema(family: ToolFamilySpec) -> None:
    for property_name, schema in family.properties_dict().items():
        schema_type = schema.get("type")
        if schema_type == "array":
            items = schema["items"]
            assert items["type"] in ALLOWED_PRIMITIVE_TYPES, property_name
            assert schema.get("maxItems", 10) <= 5, property_name
        else:
            assert schema_type in ALLOWED_PRIMITIVE_TYPES, property_name
        assert "object" not in str(schema_type), property_name


def test_train_and_unseen_families_cover_locked_domains() -> None:
    assert len(TRAIN_TOOL_FAMILIES) == 8
    assert len(UNSEEN_TOOL_FAMILIES) == 8


def test_train_and_unseen_tool_names_are_disjoint() -> None:
    train_names = {name for family in TRAIN_TOOL_FAMILIES for name in family.tool_names}
    unseen_names = {name for family in UNSEEN_TOOL_FAMILIES for name in family.tool_names}
    assert train_names.isdisjoint(unseen_names)


def test_all_sampled_schemas_are_flat() -> None:
    for family in [*TRAIN_TOOL_FAMILIES, *UNSEEN_TOOL_FAMILIES]:
        _assert_flat_schema(family)


def test_sample_definition_is_seeded_and_deterministic() -> None:
    family = TRAIN_TOOL_FAMILIES[0]
    first = family.sample_definition(Random(12))
    second = family.sample_definition(Random(12))
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_sampled_task_arguments_validate_against_sampled_schema() -> None:
    for family in [*TRAIN_TOOL_FAMILIES, *UNSEEN_TOOL_FAMILIES]:
        rng = Random(family.family)
        sampled = family.sample(rng)
        errors = validate_arguments(sampled.definition.function.parameters, sampled.task.arguments)
        assert errors == []
        assert sampled.task.tool_result
        assert sampled.task.final_answer
