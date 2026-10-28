"""Host-side lab tools for a sandboxed experimentalist."""

from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import (
    Callable,  # ruff: ignore[typing-only-standard-library-import] - Pydantic resolves this annotation.
)
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, ClassVar, cast, override

import httpx
from agents import FunctionTool, Tool
from agents.sandbox.capabilities import Capability
from agents.sandbox.entries import File
from pydantic import Field

from wiki_agents.agents.experimentalist.lab_api import discover_lab, materials_to_csv

if TYPE_CHECKING:
    from agents.sandbox import Manifest
    from agents.tool_context import ToolContext

_DEFAULT_API_URL = "http://127.0.0.1:8000"
_MATERIALS_PATH = PurePosixPath("inputs/available_materials.csv")
_MATERIALS_DESCRIPTION = "Available lab material combinations."
_MAX_REASONING_LENGTH = 1000
logger = logging.getLogger("wiki_agents.agents.experimentalist.lab")


class _InvalidRequestError(ValueError):
    """Signal that a tool submission does not meet the lab contract."""


class LabCapability(Capability):
    """Expose a configured lab API to an agent without sandbox network access."""

    type: str = "lab"
    api_url: str
    schema_data: dict[str, object]
    materials_csv: bytes
    material_records: list[dict[str, object]]
    batch_size: int
    remaining_budget: int
    remaining_budget_state: list[int] = Field(exclude=True, repr=False)
    observations: list[dict[str, object]]
    on_accepted: (
        Callable[[list[dict[str, object]], str, dict[str, object]], None] | None
    ) = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    client: httpx.Client = Field(exclude=True, repr=False)

    _request_tool_name: ClassVar[str] = "request_data"

    @override
    def clone(self) -> LabCapability:
        """Copy per-run metadata and keep the host HTTP client shared."""
        cloned = self.model_copy(deep=False)
        cloned.schema_data = copy.deepcopy(self.schema_data)
        return cloned

    @override
    def tools(self) -> list[Tool]:
        """Build the model-facing lab request tool."""
        params_schema = self._params_schema()

        async def invoke(  # ruff: ignore[unused-async]
            _: ToolContext[object],
            arguments: str,
        ) -> str:
            return self._invoke(arguments)

        return [
            FunctionTool(
                name=self._request_tool_name,
                description="Request exact experiment datapoints from the configured lab.",
                params_json_schema=params_schema,
                on_invoke_tool=invoke,
                strict_json_schema=True,
            ),
        ]

    def _params_schema(self) -> dict[str, object]:
        """Build the strict model-facing request schema."""
        properties = {
            item["name"]: {"type": item["type"]}
            for item in cast("list[dict[str, str]]", self.schema_data["inputs"])
        }
        required = cast("list[str]", self.schema_data["required_inputs"])
        item_schema: dict[str, object] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        params_schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "requests": {
                    "type": "array",
                    "items": item_schema,
                    "minItems": 1,
                    "maxItems": self._request_limit,
                },
                "reasoning": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_REASONING_LENGTH,
                    "description": "Concise evidence-based rationale for the complete batch.",
                },
            },
            "required": ["requests", "reasoning"],
            "additionalProperties": False,
        }

        return params_schema

    @property
    def _request_limit(self) -> int:
        """Maximum number of requests in the next batch."""
        return min(
            self.batch_size,
            self.remaining_budget_state[0],
            cast("int", self.schema_data["max_batch_size"]),
        )

    def _invoke(self, arguments: str) -> str:
        """Validate one submission and contact the lab only when it is accepted."""
        try:
            requests, reasoning = self._parse_submission(arguments)
        except _InvalidRequestError as exc:
            return _tool_error(str(exc))

        logger.info(
            "Lab data requested: requests=%s reasoning=%s",
            json.dumps(requests, sort_keys=True),
            reasoning,
        )
        rejection = self._request_rejection(requests)
        if rejection is not None:
            return rejection

        accepted = self._request_data(requests)
        if self.on_accepted is not None:
            self.on_accepted(requests, reasoning, accepted)
        status = self._record_not_found(requests, reasoning, accepted)
        return json.dumps(
            {"status": status, "reasoning": reasoning, **accepted},
            sort_keys=True,
        )

    def _parse_submission(
        self,
        arguments: str,
    ) -> tuple[list[dict[str, object]], str]:
        """Parse and validate model-supplied request arguments."""
        try:
            payload = cast("object", json.loads(arguments))
        except json.JSONDecodeError as exc:
            raise RuntimeError("request_data received invalid JSON arguments") from exc
        if not isinstance(payload, dict):
            raise _InvalidRequestError("Arguments must be a JSON object.")
        raw_reasoning = payload.get("reasoning")
        if not isinstance(raw_reasoning, str) or not raw_reasoning.strip():
            raise _InvalidRequestError("Provide one concise, non-empty rationale.")
        reasoning = raw_reasoning.strip()
        if len(reasoning) > _MAX_REASONING_LENGTH:
            raise _InvalidRequestError(
                f"The rationale must contain at most {_MAX_REASONING_LENGTH} characters.",
            )
        raw_requests = payload.get("requests")
        if not isinstance(raw_requests, list) or not raw_requests:
            raise _InvalidRequestError("requests must contain at least one item.")
        requests = raw_requests[: self._request_limit]
        if not requests:
            raise _InvalidRequestError("The campaign request budget is exhausted.")
        if not all(isinstance(item, dict) for item in requests):
            raise _InvalidRequestError("Every request must be an object.")
        return cast("list[dict[str, object]]", requests), reasoning

    def _request_rejection(self, requests: list[dict[str, object]]) -> str | None:
        """Return a rejection response for a duplicate or unavailable request."""
        if any(request in requests[:index] for index, request in enumerate(requests)):
            return _tool_error(
                "A batch cannot contain the same experiment more than once.",
            )
        cached = self._cached_results(requests)
        if cached:
            return json.dumps(
                {
                    "status": "duplicate",
                    "message": (
                        "This batch contains an experiment that was already measured. "
                        "Its cached result is included. Select another available material."
                    ),
                    "cached_results": cached,
                },
                sort_keys=True,
            )
        unavailable = [
            request
            for request in requests
            if not any(request == material for material in self.material_records)
        ]
        if unavailable:
            return json.dumps(
                {
                    "status": "unavailable",
                    "message": (
                        "The complete batch was rejected. Select only rows from "
                        "inputs/available_materials.csv."
                    ),
                    "requests": unavailable,
                },
                sort_keys=True,
            )
        return None

    def _request_data(
        self,
        requests: list[dict[str, object]],
    ) -> dict[str, object]:
        """Submit an accepted batch to the lab and validate its response."""
        try:
            response = self.client.post("/v1/data", json={"requests": requests})
            response.raise_for_status()
            response_payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"Lab data request failed: {exc}") from exc
        if not isinstance(response_payload, dict):
            raise TypeError("Lab data response must be a JSON object")
        return cast("dict[str, object]", response_payload)

    def _record_not_found(
        self,
        requests: list[dict[str, object]],
        reasoning: str,
        accepted: dict[str, object],
    ) -> str:
        """Record a fully missing batch and return its tool status."""
        raw_results = accepted.get("results")
        if not (
            isinstance(raw_results, list)
            and len(raw_results) == len(requests)
            and all(
                isinstance(result, dict) and result.get("found") is False
                for result in raw_results
            )
        ):
            return "accepted"

        for request, result in zip(requests, raw_results, strict=True):
            self.observations.append(
                {"request": request, "reasoning": reasoning, "result": result},
            )
        self.material_records[:] = [
            material for material in self.material_records if material not in requests
        ]
        self.materials_csv = materials_to_csv(
            cast("list[dict[str, str]]", self.schema_data["materials"]),
            self.material_records,
        )
        return "not_found"

    def _cached_results(
        self,
        requests: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Return cached records for requests that occur in the batch."""
        return [
            observation
            for request in requests
            for observation in self.observations
            if observation.get("request") == request
        ]

    def set_campaign_state(
        self,
        *,
        materials: list[dict[str, object]],
        remaining_budget: int,
        observations: list[dict[str, object]],
    ) -> None:
        """Update mutable state before a researcher session."""
        self.material_records[:] = copy.deepcopy(materials)
        self.materials_csv = materials_to_csv(
            cast("list[dict[str, str]]", self.schema_data["materials"]),
            materials,
        )
        self.remaining_budget = remaining_budget
        self.remaining_budget_state[0] = remaining_budget
        self.observations[:] = copy.deepcopy(observations)

    @override
    def process_manifest(self, manifest: Manifest) -> Manifest:
        updated = manifest.model_copy(deep=True)
        existing_path = _manifest_materials_path(updated)
        if existing_path is not None:
            existing = updated.entries[existing_path]
            if (
                not isinstance(existing, File)
                or existing.description != _MATERIALS_DESCRIPTION
            ):
                raise ValueError(
                    f"Manifest path collision at {_MATERIALS_PATH.as_posix()}",
                )
        target = (
            Path(_MATERIALS_PATH.as_posix()) if existing_path is None else existing_path
        )
        updated.entries[target] = File(
            content=self.materials_csv,
            description=_MATERIALS_DESCRIPTION,
        )
        return updated

    @override
    async def instructions(self, manifest: Manifest) -> str:
        _ = manifest
        required = ", ".join(cast("list[str]", self.schema_data["required_inputs"]))
        returned = ", ".join(
            item["name"]
            for item in cast("list[dict[str, str]]", self.schema_data["outputs"])
        )
        return (
            "Available material combinations are in "
            f"`{_MATERIALS_PATH.as_posix()}`. Use `request_data` with every required "
            f"input field: {required}. Found results return these fields: {returned}. "
            "Lab requests use exact matches. A valid combination that is absent from the "
            "experiment dataset returns `found: false`. Include one concise evidence-based "
            "`reasoning` rationale with each submission."
        )


def create_lab_capability(  # ruff: ignore[too-many-arguments] - Campaign limits are explicit.
    api_url: str | None = None,
    *,
    timeout: float = 10.0,
    transport: httpx.BaseTransport | None = None,
    batch_size: int | None = None,
    remaining_budget: int = 100,
    observations: list[dict[str, object]] | None = None,
    on_accepted: Callable[[list[dict[str, object]], str, dict[str, object]], None]
    | None = None,
) -> LabCapability:
    """Discover a lab API and create its sandbox capability."""
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    if remaining_budget < 1:
        raise ValueError("remaining_budget must be positive")
    configured_url = (
        api_url
        if api_url is not None
        else os.environ.get("LAB_API_URL", _DEFAULT_API_URL)
    )
    resolved_url = configured_url.rstrip("/")
    client = httpx.Client(base_url=resolved_url, timeout=timeout, transport=transport)
    try:
        schema, material_records, materials_csv = discover_lab(client)
    except Exception:
        client.close()
        raise
    return LabCapability(
        api_url=resolved_url,
        schema_data=schema,
        materials_csv=materials_csv,
        material_records=material_records,
        batch_size=(
            cast("int", schema["max_batch_size"]) if batch_size is None else batch_size
        ),
        remaining_budget=remaining_budget,
        remaining_budget_state=[remaining_budget],
        observations=[] if observations is None else observations,
        on_accepted=on_accepted,
        client=client,
    )


def _tool_error(message: str) -> str:
    """Return a model-facing rejected-tool result."""
    return json.dumps({"status": "rejected", "message": message}, sort_keys=True)


def _manifest_materials_path(manifest: Manifest) -> str | Path | None:
    """Return the existing manifest key for the mounted materials path."""
    root = PurePosixPath(manifest.root)
    target_absolute = root / _MATERIALS_PATH
    for raw_path in manifest.entries:
        path = PurePosixPath(Path(raw_path).as_posix())
        if path in {_MATERIALS_PATH, target_absolute}:
            return raw_path
    return None
