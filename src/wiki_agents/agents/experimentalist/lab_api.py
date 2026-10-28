"""Discover, validate, and serialize data from the configured lab API."""

from __future__ import annotations

import csv
import io
import math
from typing import cast

import httpx

_JSON_TYPES = frozenset({"string", "integer", "number", "boolean"})


def discover_lab(
    client: httpx.Client,
) -> tuple[dict[str, object], list[dict[str, object]], bytes]:
    """Load the lab schema, material catalog, and serialized catalog."""
    schema = _get_json(client, "/v1/schema", "schema")
    _validate_schema(schema)
    records = _fetch_all_materials(client)
    catalog = materials_to_csv(
        cast("list[dict[str, str]]", schema["materials"]),
        records,
    )
    return schema, records, catalog


def materials_to_csv(
    definitions: list[dict[str, str]],
    records: list[dict[str, object]],
) -> bytes:
    """Serialize typed material records with agent-facing column names."""
    names = [definition["name"] for definition in definitions]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=names, lineterminator="\n")
    writer.writeheader()
    expected = set(names)
    types = {definition["name"]: definition["type"] for definition in definitions}
    for index, record in enumerate(records):
        if set(record) != expected:
            raise RuntimeError(
                f"Lab material item {index} does not match schema fields",
            )
        for name in names:
            if not _matches_type(record[name], types[name]):
                raise RuntimeError(
                    f"Lab material item {index}.{name} does not match type {types[name]}",
                )
        writer.writerow({name: _csv_scalar(record[name]) for name in names})
    return output.getvalue().encode("utf-8")


def _get_json(client: httpx.Client, path: str, label: str) -> dict[str, object]:
    """Fetch one required JSON object from the lab."""
    try:
        response = client.get(path)
        response.raise_for_status()
        payload = cast("object", response.json())
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"Could not fetch lab {label} from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Lab {label} response must be a JSON object")
    return cast("dict[str, object]", payload)


def _validate_schema(schema: dict[str, object]) -> None:
    """Validate the fields that define the model-facing tool."""
    for section in ("inputs", "outputs", "materials"):
        _validate_definitions(section, schema.get(section))

    input_names = [
        item["name"] for item in cast("list[dict[str, str]]", schema["inputs"])
    ]
    if schema.get("required_inputs") != input_names:
        raise RuntimeError("Lab schema required_inputs must list all inputs in order")
    max_batch_size = schema.get("max_batch_size")
    if (
        not isinstance(max_batch_size, int)
        or isinstance(max_batch_size, bool)
        or max_batch_size < 1
    ):
        raise RuntimeError("Lab schema max_batch_size must be a positive integer")


def _validate_definitions(section: str, value: object) -> None:
    """Validate one list of named and typed schema definitions."""
    if not isinstance(value, list):
        raise TypeError(f"Lab schema field {section} must be an array")
    if not value:
        raise RuntimeError(f"Lab schema field {section} cannot be empty")
    names: set[str] = set()
    for index, definition in enumerate(value):
        if not isinstance(definition, dict):
            raise TypeError(f"Lab schema {section}[{index}] must be an object")
        name = definition.get("name")
        type_name = definition.get("type")
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"Lab schema {section}[{index}].name is invalid")
        if type_name not in _JSON_TYPES:
            raise RuntimeError(f"Lab schema {section}[{index}].type is invalid")
        if name in names:
            raise RuntimeError(f"Lab schema {section} contains duplicate name {name!r}")
        names.add(name)


def _fetch_all_materials(client: httpx.Client) -> list[dict[str, object]]:
    """Fetch every page of material records before sandbox startup."""
    records: list[dict[str, object]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        payload = _get_json(
            client,
            f"/v1/materials?offset={offset}&limit=1000",
            "materials",
        )
        items = payload.get("items")
        page_total = payload.get("total")
        page_offset = payload.get("offset")
        if (
            not isinstance(items, list)
            or not isinstance(page_total, int)
            or isinstance(page_total, bool)
            or page_total < 0
            or page_offset != offset
        ):
            raise RuntimeError("Lab materials response has invalid pagination fields")
        if total is not None and page_total != total:
            raise RuntimeError("Lab materials total changed during pagination")
        total = page_total
        for item in items:
            if not isinstance(item, dict):
                raise TypeError("Lab material items must be JSON objects")
            records.append(cast("dict[str, object]", item))
        if not items and offset < total:
            raise RuntimeError(
                "Lab materials pagination stopped before the reported total",
            )
        offset += len(items)
        if offset > total:
            raise RuntimeError(
                "Lab materials response contains more items than reported",
            )
    return records


def _csv_scalar(value: object) -> str | int | float:
    """Normalize one JSON scalar for the mounted CSV."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return value
    raise RuntimeError(
        f"Lab material value has unsupported type: {type(value).__name__}",
    )


def _matches_type(value: object, type_name: str) -> bool:
    """Check a material value against its declared JSON type."""
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
