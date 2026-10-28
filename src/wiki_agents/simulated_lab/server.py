"""Starlette HTTP interface for a simulated lab."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from starlette.requests import Request

    from wiki_agents.simulated_lab.lab import ColumnDefinition, JsonScalar, SimulatedLab


def create_app(lab: SimulatedLab) -> Starlette:
    """Create a read-only HTTP application for a loaded lab."""

    async def health(_: Request) -> JSONResponse:  # ruff: ignore[unused-async]
        return JSONResponse({"status": "ok", "dataset_rows": lab.dataset_rows})

    async def schema(_: Request) -> JSONResponse:  # ruff: ignore[unused-async]
        return JSONResponse(lab.schema())

    async def materials(request: Request) -> JSONResponse:  # ruff: ignore[unused-async]
        try:
            offset = _query_integer(request, "offset", default=0, minimum=0)
            limit = _query_integer(request, "limit", default=1000, minimum=1)
        except ValueError as exc:
            return _error(str(exc))
        items = lab.material_records[offset : offset + limit]
        return JSONResponse(
            {
                "items": items,
                "total": len(lab.material_records),
                "offset": offset,
                "limit": limit,
            },
        )

    async def data(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:  # ruff: ignore[blind-except] - Starlette exposes parser-specific errors.
            return _error("Request body must be valid JSON")
        try:
            request_items = _validate_envelope(payload, lab)
        except (TypeError, ValueError) as exc:
            return _error(str(exc))

        results: list[dict[str, object]] = []
        for request_item, key in request_items:
            values = lab.experiments.get(key)
            result: dict[str, object] = {
                "request": request_item,
                "found": values is not None,
            }
            if values is not None:
                result["values"] = values
            results.append(result)
        return JSONResponse({"results": results})

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/schema", schema, methods=["GET"]),
            Route("/v1/materials", materials, methods=["GET"]),
            Route("/v1/data", data, methods=["POST"]),
        ],
    )


def _query_integer(
    request: Request,
    name: str,
    *,
    default: int,
    minimum: int,
) -> int:
    """Read and validate an integer query parameter."""
    raw = request.query_params.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Query parameter {name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"Query parameter {name} must be at least {minimum}")
    return value


def _validate_envelope(  # ruff: ignore[complex-structure]
    payload: object,
    lab: SimulatedLab,
) -> list[tuple[dict[str, JsonScalar], tuple[JsonScalar, ...]]]:
    """Validate a complete data-request envelope."""
    if not isinstance(payload, dict) or set(payload) != {"requests"}:
        raise ValueError("Body must be an object containing only a requests field")
    envelope = cast("dict[str, object]", payload)
    requests = envelope["requests"]
    if not isinstance(requests, list):
        raise TypeError("requests must be an array")
    if not requests:
        raise ValueError("requests must contain at least one item")
    if len(requests) > lab.max_batch_size:
        raise ValueError(
            f"requests exceeds the maximum batch size of {lab.max_batch_size}",
        )

    required = [definition.name for definition in lab.inputs]
    required_set = set(required)
    validated: list[tuple[dict[str, JsonScalar], tuple[JsonScalar, ...]]] = []
    for index, item in enumerate(requests):
        if not isinstance(item, dict):
            raise TypeError(f"requests[{index}] must be an object")
        request_item = cast("dict[str, object]", item)
        fields = set(request_item)
        missing = required_set - fields
        unknown = fields - required_set
        if missing:
            raise ValueError(
                f"requests[{index}] is missing input fields: {', '.join(sorted(missing))}",
            )
        if unknown:
            raise ValueError(
                f"requests[{index}] has unknown input fields: {', '.join(sorted(unknown))}",
            )
        typed_item: dict[str, JsonScalar] = {}
        for definition in lab.inputs:
            value = request_item[definition.name]
            if not _has_type(value, definition):
                raise ValueError(
                    f"requests[{index}].{definition.name} must be {definition.type}",
                )
            typed_item[definition.name] = cast("JsonScalar", value)
        validated.append((typed_item, tuple(typed_item[name] for name in required)))
    return validated


def _has_type(value: object, definition: ColumnDefinition) -> bool:
    """Check one JSON value against a lab column type."""
    if definition.type == "string":
        return isinstance(value, str)
    if definition.type == "boolean":
        return isinstance(value, bool)
    if definition.type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _error(message: str, status_code: int = 400) -> JSONResponse:
    """Return a stable whole-request error response."""
    return JSONResponse({"error": message}, status_code=status_code)
