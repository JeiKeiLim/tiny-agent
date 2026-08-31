"""Sampled tool schema families for local M2 SFT generation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from random import Random
from typing import Any

from kestrel.data.sft_schema import FunctionSpec, ToolDefinition


@dataclass(frozen=True)
class ToolTask:
    """A generated user prompt plus the expected local tool interaction."""

    user_prompt: str
    arguments: dict[str, Any]
    tool_result: str
    final_answer: str


@dataclass(frozen=True)
class SampledTool:
    """A sampled tool definition and a task that uses that tool."""

    family: str
    definition: ToolDefinition
    task: ToolTask


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _integer(description: str) -> dict[str, Any]:
    return {"type": "integer", "description": description}


def _number(description: str) -> dict[str, Any]:
    return {"type": "number", "description": description}


def _enum(values: tuple[str, ...], description: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values), "description": description}


def _string_list(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": 3,
        "description": description,
    }


def _result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _number_value(value: float) -> int | float:
    rounded = round(value, 6)
    return int(rounded) if float(rounded).is_integer() else rounded


CITIES = ("Seoul", "Tokyo", "Berlin", "Austin", "Oslo", "Lisbon")
CONDITIONS = ("sunny", "cloudy", "rainy", "snowy", "windy")
TERMS = {
    "api": "An interface for programs to request data or actions from software.",
    "cache": "Storage that keeps recently used data available for fast reuse.",
    "token": "A small unit of text or an authentication credential.",
    "vector": "A list of numbers that represents data in a mathematical space.",
}
DOCUMENT_TOPICS = ("authentication", "billing", "deployment", "search")
RECORD_TABLES = ("customer", "order", "product")
RECORD_STATUSES = ("active", "pending", "closed")
WAREHOUSES = ("east", "west", "central")
REGIONS = ("apac", "amer", "emea")
SITES = ("north", "south", "central")


def _weather_task(rng: Random) -> ToolTask:
    city = rng.choice(CITIES)
    unit = rng.choice(("celsius", "fahrenheit"))
    temperature_c = rng.randint(-10, 35)
    condition = rng.choice(CONDITIONS)
    if unit == "fahrenheit":
        temperature = _number_value(temperature_c * 9 / 5 + 32)
        label = "Fahrenheit"
    else:
        temperature = temperature_c
        label = "Celsius"
    arguments = {"city": city, "unit": unit}
    tool_result = _result(
        {"city": city, "condition": condition, "temperature": temperature, "unit": unit}
    )
    user_prompt = f"What is the current weather in {city}? Report it in {unit}."
    final_answer = (
        f"The current weather in {city} is {condition}, "
        f"with a temperature of {temperature} degrees {label}."
    )
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> int | float:
    celsius = value if from_unit == "c" else (value - 32) * 5 / 9
    if to_unit == "c":
        return _number_value(celsius)
    return _number_value(celsius * 9 / 5 + 32)


def _convert_factor(category: str, unit: str) -> float:
    if category == "length":
        return {"m": 1.0, "km": 1000.0, "mi": 1609.344}[unit]
    if category == "weight":
        return {"kg": 1.0, "lb": 0.45359237}[unit]
    msg = "temperature uses a dedicated conversion path"
    raise ValueError(msg)


def _unit_conversion_task(rng: Random) -> ToolTask:
    category = rng.choice(("length", "weight", "temperature"))
    units: tuple[str, ...]
    if category == "length":
        units = ("m", "km", "mi")
    elif category == "weight":
        units = ("kg", "lb")
    else:
        units = ("c", "f")
    from_unit, to_unit = rng.sample(units, 2)
    value = rng.randint(1, 100)
    if category == "temperature":
        converted = _convert_temperature(value, from_unit, to_unit)
    else:
        converted = _number_value(
            value * _convert_factor(category, from_unit) / _convert_factor(category, to_unit)
        )
    arguments = {
        "value": value,
        "category": category,
        "from_unit": from_unit,
        "to_unit": to_unit,
    }
    tool_result = _result({"value": converted, "from_unit": from_unit, "to_unit": to_unit})
    user_prompt = f"Convert {value} {from_unit} to {to_unit}."
    final_answer = f"{value} {from_unit} equals {converted} {to_unit}."
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _calculator_task(rng: Random) -> ToolTask:
    a = rng.randint(-50, 100)
    b = rng.randint(1, 20)
    operator = rng.choice(("add", "subtract", "multiply", "divide"))
    result: int | float
    if operator == "add":
        symbol = "+"
        result = a + b
    elif operator == "subtract":
        symbol = "-"
        result = a - b
    elif operator == "multiply":
        symbol = "*"
        result = a * b
    else:
        symbol = "/"
        result = _number_value(a / b)
    arguments = {"a": a, "operator": operator, "b": b}
    tool_result = _result({"result": result})
    user_prompt = f"Calculate {a} {symbol} {b}."
    final_answer = f"{a} {symbol} {b} equals {result}."
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _date_math_task(rng: Random) -> ToolTask:
    base = date(2026, rng.randint(1, 12), rng.randint(1, 28))
    days = rng.randint(-30, 30)
    shifted = base + timedelta(days=days)
    arguments = {"date": base.isoformat(), "days": days}
    tool_result = _result(
        {"original": base.isoformat(), "days": days, "result": shifted.isoformat()}
    )
    direction = "after" if days >= 0 else "before"
    user_prompt = f"What date is {abs(days)} days {direction} {base.isoformat()}?"
    final_answer = f"{abs(days)} days {direction} {base.isoformat()} is {shifted.isoformat()}."
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _lookup_task(rng: Random) -> ToolTask:
    term = rng.choice(tuple(TERMS))
    definition = TERMS[term]
    arguments = {"term": term}
    tool_result = _result({"term": term, "definition": definition})
    user_prompt = f"What does the term {term!r} mean?"
    final_answer = f"{term.capitalize()} means: {definition}"
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _document_search_task(rng: Random) -> ToolTask:
    query = rng.choice(DOCUMENT_TOPICS)
    limit = rng.randint(1, 3)
    tags = list(rng.sample(("guide", "api", "release"), k=rng.randint(0, 2)))
    arguments: dict[str, Any] = {"query": query, "limit": limit}
    if tags:
        arguments["tags"] = tags
    results = [
        {"id": f"doc-{query}-{index}", "title": f"{query} document {index}"}
        for index in range(1, limit + 1)
    ]
    tool_result = _result({"query": query, "results": results})
    user_prompt = f"Search documents for {query} and return up to {limit} results."
    if tags:
        user_prompt += f" Prefer tags: {', '.join(tags)}."
    final_answer = f"Found {limit} documents matching {query}."
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _record_lookup_task(rng: Random) -> ToolTask:
    table = rng.choice(RECORD_TABLES)
    record_id = f"{table[:3].upper()}-{rng.randint(100, 999)}"
    status = rng.choice(RECORD_STATUSES)
    arguments = {"table": table, "record_id": record_id}
    tool_result = _result({"table": table, "record_id": record_id, "status": status})
    user_prompt = f"Look up the {table} record {record_id}."
    final_answer = f"The {table} record {record_id} is {status}."
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _inventory_task(rng: Random) -> ToolTask:
    sku = f"SKU-{rng.randint(1000, 9999)}"
    warehouse = rng.choice(WAREHOUSES)
    stock = rng.randint(0, 100)
    arguments = {"sku": sku, "warehouse": warehouse}
    tool_result = _result({"sku": sku, "warehouse": warehouse, "stock": stock})
    user_prompt = f"How many {sku} units are in the {warehouse} warehouse?"
    final_answer = f"There are {stock} {sku} units in the {warehouse} warehouse."
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _forecast_task(rng: Random) -> ToolTask:
    location = rng.choice(CITIES)
    base = date(2026, 8, 31)
    offset_days = rng.randint(0, 3)
    forecast_date = base + timedelta(days=offset_days)
    temperature_c = rng.randint(-5, 30)
    condition = rng.choice(CONDITIONS)
    arguments = {"location": location, "forecast_date": forecast_date.isoformat()}
    tool_result = _result(
        {
            "location": location,
            "forecast_date": forecast_date.isoformat(),
            "temperature_c": temperature_c,
            "condition": condition,
        }
    )
    user_prompt = f"What is the forecast for {location} on {forecast_date.isoformat()}?"
    final_answer = (
        f"The forecast for {location} on {forecast_date.isoformat()} "
        f"is {condition}, around {temperature_c} degrees Celsius."
    )
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _amount_conversion_task(rng: Random) -> ToolTask:
    category = rng.choice(("length", "weight", "temperature"))
    scales: tuple[str, ...]
    if category == "length":
        scales = ("m", "km", "mi")
    elif category == "weight":
        scales = ("kg", "lb")
    else:
        scales = ("c", "f")
    source_scale, target_scale = rng.sample(scales, 2)
    amount = rng.randint(1, 100)
    if category == "temperature":
        converted = _convert_temperature(amount, source_scale, target_scale)
    else:
        converted = _number_value(
            amount
            * _convert_factor(category, source_scale)
            / _convert_factor(category, target_scale)
        )
    arguments = {
        "amount": amount,
        "category": category,
        "source_scale": source_scale,
        "target_scale": target_scale,
    }
    tool_result = _result(
        {"value": converted, "source_scale": source_scale, "target_scale": target_scale}
    )
    user_prompt = f"Convert an amount of {amount} {source_scale} into {target_scale}."
    final_answer = f"An amount of {amount} {source_scale} equals {converted} {target_scale}."
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _expression_task(rng: Random) -> ToolTask:
    a = rng.randint(-20, 80)
    b = rng.randint(1, 12)
    operator = rng.choice(("add", "subtract", "multiply", "divide"))
    result: int | float
    if operator == "add":
        symbol = "+"
        result = a + b
    elif operator == "subtract":
        symbol = "-"
        result = a - b
    elif operator == "multiply":
        symbol = "*"
        result = a * b
    else:
        symbol = "/"
        result = _number_value(a / b)
    expression = f"{a} {symbol} {b}"
    arguments = {"expression": expression}
    tool_result = _result({"expression": expression, "result": result})
    user_prompt = f"Evaluate this expression: {expression}."
    final_answer = f"The expression {expression} equals {result}."
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _date_offset_task(rng: Random) -> ToolTask:
    base = date(2026, rng.randint(1, 12), rng.randint(1, 28))
    offset_days = rng.randint(-14, 14)
    shifted = base + timedelta(days=offset_days)
    arguments = {"start_date": base.isoformat(), "offset_days": offset_days}
    tool_result = _result(
        {"start_date": base.isoformat(), "offset_days": offset_days, "result": shifted.isoformat()}
    )
    direction = "after" if offset_days >= 0 else "before"
    user_prompt = f"Move {base.isoformat()} by {abs(offset_days)} days {direction}."
    final_answer = f"Moving {base.isoformat()} by {offset_days} days gives {shifted.isoformat()}."
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _glossary_task(rng: Random) -> ToolTask:
    word = rng.choice(tuple(TERMS))
    definition = TERMS[word]
    arguments = {"word": word}
    tool_result = _result({"word": word, "definition": definition})
    user_prompt = f"Look up the glossary entry for {word!r}."
    final_answer = f"The glossary entry for {word} is: {definition}"
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _knowledge_search_task(rng: Random) -> ToolTask:
    text = rng.choice(DOCUMENT_TOPICS)
    max_results = rng.randint(1, 3)
    labels = list(rng.sample(("manual", "faq", "postmortem"), k=rng.randint(0, 2)))
    arguments: dict[str, Any] = {"text": text, "max_results": max_results}
    if labels:
        arguments["labels"] = labels
    results = [
        {"id": f"kb-{text}-{index}", "title": f"{text} knowledge note {index}"}
        for index in range(1, max_results + 1)
    ]
    tool_result = _result({"text": text, "results": results})
    user_prompt = f"Search the knowledge base for {text} with up to {max_results} results."
    if labels:
        user_prompt += f" Prefer labels: {', '.join(labels)}."
    final_answer = f"Found {max_results} knowledge base results for {text}."
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _account_record_task(rng: Random) -> ToolTask:
    account_id = f"ACC-{rng.randint(10000, 99999)}"
    region = rng.choice(REGIONS)
    status = rng.choice(RECORD_STATUSES)
    arguments = {"account_id": account_id, "region": region}
    tool_result = _result({"account_id": account_id, "region": region, "status": status})
    user_prompt = f"Fetch the {region} account record for {account_id}."
    final_answer = f"The {region} account record {account_id} is {status}."
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


def _stock_level_task(rng: Random) -> ToolTask:
    item_code = f"ITEM-{rng.randint(10000, 99999)}"
    site = rng.choice(SITES)
    quantity = rng.randint(0, 200)
    arguments = {"item_code": item_code, "site": site}
    tool_result = _result({"item_code": item_code, "site": site, "quantity": quantity})
    user_prompt = f"Check the stock level for {item_code} at the {site} site."
    final_answer = f"The {site} site has {quantity} units of {item_code}."
    return ToolTask(user_prompt, arguments, tool_result, final_answer)


@dataclass(frozen=True)
class ToolFamilySpec:
    """A sampled family of flat, read-only M2 tool schemas."""

    family: str
    tool_names: tuple[str, ...]
    descriptions: tuple[str, ...]
    properties: tuple[tuple[str, dict[str, Any]], ...]
    required: tuple[str, ...]
    task: Callable[[Random], ToolTask]
    missing_info: tuple[tuple[str, str], ...] = ()

    def properties_dict(self) -> dict[str, Any]:
        return dict(self.properties)

    def _definition_for(self, rng: Random, name: str) -> ToolDefinition:
        return ToolDefinition(
            type="function",
            function=FunctionSpec(
                name=name,
                description=rng.choice(self.descriptions),
                parameters=_object_schema(self.properties_dict(), list(self.required)),
            ),
        )

    def sample_definition(self, rng: Random) -> ToolDefinition:
        return self._definition_for(rng, rng.choice(self.tool_names))

    def sample(self, rng: Random) -> SampledTool:
        name = rng.choice(self.tool_names)
        return SampledTool(
            family=self.family,
            definition=self._definition_for(rng, name),
            task=self.task(rng),
        )


TRAIN_TOOL_FAMILIES: tuple[ToolFamilySpec, ...] = (
    ToolFamilySpec(
        family="weather",
        tool_names=("get_current_weather", "lookup_current_weather"),
        descriptions=("Look up current weather for a city.", "Get current city weather."),
        properties=(
            ("city", _string("City name")),
            ("unit", _enum(("celsius", "fahrenheit"), "Temperature unit")),
        ),
        required=("city", "unit"),
        task=_weather_task,
        missing_info=(
            ("What is the weather right now?", "Which city would you like the weather for?"),
            ("I need the current conditions.", "Which city would you like the weather for?"),
        ),
    ),
    ToolFamilySpec(
        family="unit_conversion",
        tool_names=("convert_units", "convert_measurement"),
        descriptions=("Convert a measurement between units.", "Convert values across units."),
        properties=(
            ("value", _number("Value to convert")),
            ("category", _enum(("length", "weight", "temperature"), "Measurement category")),
            ("from_unit", _enum(("m", "km", "mi", "kg", "lb", "c", "f"), "Source unit")),
            ("to_unit", _enum(("m", "km", "mi", "kg", "lb", "c", "f"), "Target unit")),
        ),
        required=("value", "category", "from_unit", "to_unit"),
        task=_unit_conversion_task,
        missing_info=(
            (
                "Can you convert a measurement for me?",
                "What value, category, and units should I convert?",
            ),
            ("I need a unit conversion.", "What value, category, and units should I convert?"),
        ),
    ),
    ToolFamilySpec(
        family="calculator",
        tool_names=("calculate", "evaluate_arithmetic"),
        descriptions=(
            "Calculate a basic arithmetic expression.",
            "Evaluate two-number arithmetic.",
        ),
        properties=(
            ("a", _number("First number")),
            ("operator", _enum(("add", "subtract", "multiply", "divide"), "Arithmetic operator")),
            ("b", _number("Second number")),
        ),
        required=("a", "operator", "b"),
        task=_calculator_task,
        missing_info=(
            (
                "Can you calculate something for me?",
                "Which two numbers and operation should I calculate?",
            ),
            (
                "I need an arithmetic calculation.",
                "Which two numbers and operation should I calculate?",
            ),
        ),
    ),
    ToolFamilySpec(
        family="date_math",
        tool_names=("add_days_to_date", "shift_date_by_days"),
        descriptions=("Shift a date by a number of days.", "Calculate a date offset."),
        properties=(
            ("date", _string("Start date in YYYY-MM-DD format")),
            ("days", _integer("Number of days to add; negative values move backward")),
        ),
        required=("date", "days"),
        task=_date_math_task,
        missing_info=(
            ("What date is 10 days from now?", "Which start date should I use?"),
            ("I need a date offset.", "Which start date and day offset should I use?"),
        ),
    ),
    ToolFamilySpec(
        family="simple_lookup",
        tool_names=("lookup_term", "lookup_definition"),
        descriptions=("Look up a short term definition.", "Get a term definition."),
        properties=(("term", _string("Term to look up")),),
        required=("term",),
        task=_lookup_task,
        missing_info=(
            ("What does that term mean?", "Which term would you like me to look up?"),
            ("I need a definition.", "Which term would you like me to look up?"),
        ),
    ),
    ToolFamilySpec(
        family="document_search",
        tool_names=("search_documents", "find_documents"),
        descriptions=("Search documents by query.", "Find documents matching a query."),
        properties=(
            ("query", _string("Search query")),
            ("limit", _integer("Maximum number of results")),
            ("tags", _string_list("Optional document tags")),
        ),
        required=("query", "limit"),
        task=_document_search_task,
        missing_info=(
            ("Search the docs for something useful.", "What query should I search for?"),
            ("I need a document search.", "What query should I search for?"),
        ),
    ),
    ToolFamilySpec(
        family="record_lookup",
        tool_names=("lookup_record", "fetch_record"),
        descriptions=(
            "Look up a simple record by table and ID.",
            "Fetch a record by table and ID.",
        ),
        properties=(
            ("table", _enum(RECORD_TABLES, "Record table")),
            ("record_id", _string("Record identifier")),
        ),
        required=("table", "record_id"),
        task=_record_lookup_task,
        missing_info=(
            ("Look up the record.", "Which table and record ID should I look up?"),
            ("I need a record lookup.", "Which table and record ID should I look up?"),
        ),
    ),
    ToolFamilySpec(
        family="inventory",
        tool_names=("lookup_inventory", "check_inventory"),
        descriptions=("Check inventory for a SKU and warehouse.", "Look up inventory stock."),
        properties=(
            ("sku", _string("Stock keeping unit")),
            ("warehouse", _enum(WAREHOUSES, "Warehouse name")),
        ),
        required=("sku", "warehouse"),
        task=_inventory_task,
        missing_info=(
            ("Check the stock.", "Which SKU and warehouse should I check?"),
            ("I need an inventory check.", "Which SKU and warehouse should I check?"),
        ),
    ),
)

UNSEEN_TOOL_FAMILIES: tuple[ToolFamilySpec, ...] = (
    ToolFamilySpec(
        family="weather_forecast",
        tool_names=("get_weather_forecast", "forecast_weather"),
        descriptions=(
            "Get a weather forecast for a location and date.",
            "Look up a dated forecast.",
        ),
        properties=(
            ("location", _string("Location name")),
            ("forecast_date", _string("Forecast date in YYYY-MM-DD format")),
        ),
        required=("location", "forecast_date"),
        task=_forecast_task,
        missing_info=(
            ("What is the forecast?", "Which location and date should I forecast?"),
            ("I need a weather forecast.", "Which location and date should I forecast?"),
        ),
    ),
    ToolFamilySpec(
        family="amount_conversion",
        tool_names=("convert_amount", "convert_value"),
        descriptions=("Convert an amount between scales.", "Convert a measured amount."),
        properties=(
            ("amount", _number("Amount to convert")),
            ("category", _enum(("length", "weight", "temperature"), "Measurement category")),
            ("source_scale", _enum(("m", "km", "mi", "kg", "lb", "c", "f"), "Source scale")),
            ("target_scale", _enum(("m", "km", "mi", "kg", "lb", "c", "f"), "Target scale")),
        ),
        required=("amount", "category", "source_scale", "target_scale"),
        task=_amount_conversion_task,
        missing_info=(
            ("Can you convert an amount?", "What amount, category, and scales should I convert?"),
            ("I need an amount conversion.", "What amount, category, and scales should I convert?"),
        ),
    ),
    ToolFamilySpec(
        family="expression_eval",
        tool_names=("evaluate_expression", "compute_expression"),
        descriptions=("Evaluate a simple arithmetic expression.", "Compute an expression string."),
        properties=(("expression", _string("Arithmetic expression to evaluate")),),
        required=("expression",),
        task=_expression_task,
        missing_info=(
            ("Can you evaluate an expression?", "Which expression should I evaluate?"),
            ("I need an expression computed.", "Which expression should I evaluate?"),
        ),
    ),
    ToolFamilySpec(
        family="date_offset",
        tool_names=("offset_date", "move_date"),
        descriptions=("Move a start date by an offset.", "Apply a day offset to a date."),
        properties=(
            ("start_date", _string("Start date in YYYY-MM-DD format")),
            ("offset_days", _integer("Day offset to apply")),
        ),
        required=("start_date", "offset_days"),
        task=_date_offset_task,
        missing_info=(
            ("Move a date for me.", "Which start date and offset should I use?"),
            ("I need a date moved.", "Which start date and offset should I use?"),
        ),
    ),
    ToolFamilySpec(
        family="glossary_lookup",
        tool_names=("lookup_glossary_entry", "get_glossary_entry"),
        descriptions=("Look up a glossary entry.", "Get a glossary definition."),
        properties=(("word", _string("Glossary word to look up")),),
        required=("word",),
        task=_glossary_task,
        missing_info=(
            ("Look up the glossary entry.", "Which word should I look up?"),
            ("I need a glossary lookup.", "Which word should I look up?"),
        ),
    ),
    ToolFamilySpec(
        family="knowledge_search",
        tool_names=("search_knowledge_base", "query_knowledge_base"),
        descriptions=("Search the knowledge base.", "Query knowledge base notes."),
        properties=(
            ("text", _string("Search text")),
            ("max_results", _integer("Maximum number of results")),
            ("labels", _string_list("Optional result labels")),
        ),
        required=("text", "max_results"),
        task=_knowledge_search_task,
        missing_info=(
            ("Search the knowledge base.", "What text should I search for?"),
            ("I need a knowledge base search.", "What text should I search for?"),
        ),
    ),
    ToolFamilySpec(
        family="account_record",
        tool_names=("fetch_account_record", "get_account_record"),
        descriptions=("Fetch an account record by ID and region.", "Look up an account record."),
        properties=(
            ("account_id", _string("Account identifier")),
            ("region", _enum(REGIONS, "Account region")),
        ),
        required=("account_id", "region"),
        task=_account_record_task,
        missing_info=(
            ("Fetch the account record.", "Which account ID and region should I use?"),
            ("I need an account record.", "Which account ID and region should I use?"),
        ),
    ),
    ToolFamilySpec(
        family="stock_level",
        tool_names=("check_stock_level", "get_stock_level"),
        descriptions=("Check stock level for an item and site.", "Get item stock at a site."),
        properties=(
            ("item_code", _string("Item code")),
            ("site", _enum(SITES, "Inventory site")),
        ),
        required=("item_code", "site"),
        task=_stock_level_task,
        missing_info=(
            ("Check the stock level.", "Which item code and site should I check?"),
            ("I need a stock level check.", "Which item code and site should I check?"),
        ),
    ),
)


def m2_eval_tool_names() -> frozenset[str]:
    """Return every local tool name used by the M2 seen and unseen eval sets."""
    names: set[str] = set()
    for family in (*TRAIN_TOOL_FAMILIES, *UNSEEN_TOOL_FAMILIES):
        names.update(family.tool_names)
    return frozenset(names)
