"""Tests for the sandbox lab capability."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx
import pytest
from agents.sandbox import Manifest
from agents.sandbox.entries import File

from wiki_agents.agents.experimentalist.lab_tools import create_lab_capability

if TYPE_CHECKING:
    from typing import Any

    from agents import FunctionTool, ToolContext


async def _invoke_tool(tool: FunctionTool, arguments: str) -> str:
    """Invoke a function tool through its awaitable callback."""
    result = await tool.on_invoke_tool(cast("ToolContext[Any]", None), arguments)
    return cast("str", result)


def _transport(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/schema":
        return httpx.Response(
            200,
            json={
                "inputs": [
                    {"name": "sequence", "type": "string"},
                    {"name": "ph", "type": "number"},
                ],
                "outputs": [{"name": "surface_tension", "type": "number"}],
                "materials": [
                    {"name": "sequence", "type": "string"},
                    {"name": "ph", "type": "number"},
                ],
                "required_inputs": ["sequence", "ph"],
                "max_batch_size": 2,
            },
        )
    if request.url.path == "/v1/materials":
        offset = int(request.url.params["offset"])
        items = (
            [{"sequence": "A", "ph": 3.0}]
            if offset == 0
            else [{"sequence": "B", "ph": 7.0}]
        )
        return httpx.Response(
            200,
            json={"items": items, "total": 2, "offset": offset, "limit": 1000},
        )
    if request.url.path == "/v1/data":
        payload = json.loads(request.content)
        return httpx.Response(200, json={"results": [{"forwarded": payload}]})
    return httpx.Response(404)


@pytest.fixture
def capability():
    capability = create_lab_capability(
        "http://lab.test/",
        transport=httpx.MockTransport(_transport),
    )
    yield capability
    capability.client.close()


def test_capability_builds_runtime_schema_and_forwards_requests(capability) -> None:
    tool = cast("FunctionTool", capability.tools()[0])

    assert tool.name == "request_data"
    request_schema = tool.params_json_schema["properties"]["requests"]
    assert request_schema["maxItems"] == 2
    assert tool.params_json_schema["required"] == ["requests", "reasoning"]
    assert request_schema["items"] == {
        "type": "object",
        "properties": {"sequence": {"type": "string"}, "ph": {"type": "number"}},
        "required": ["sequence", "ph"],
        "additionalProperties": False,
    }


def test_capability_tool_returns_lab_response(capability) -> None:
    tool = cast("FunctionTool", capability.tools()[0])
    arguments = (
        '{"requests":[{"sequence":"A","ph":3.0}],'
        '"reasoning":"This point tests the current trend."}'
    )

    async def invoke() -> str:
        result = await tool.on_invoke_tool(cast("ToolContext[Any]", None), arguments)
        return cast("str", result)

    result = asyncio.run(invoke())

    assert json.loads(result) == {
        "status": "accepted",
        "reasoning": "This point tests the current trend.",
        "results": [
            {"forwarded": {"requests": [{"sequence": "A", "ph": 3.0}]}},
        ],
    }


def test_capability_allows_another_request_after_no_data_is_found() -> None:
    calls = 0

    def transport(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path != "/v1/data":
            return _transport(request)
        calls += 1
        payload = cast("dict[str, object]", json.loads(request.content))
        requests = cast("list[dict[str, object]]", payload["requests"])
        return httpx.Response(
            200,
            json={
                "results": [
                    {"request": request_item, "found": False}
                    for request_item in requests
                ],
            },
        )

    capability = create_lab_capability(
        "http://lab.test",
        transport=httpx.MockTransport(transport),
    )
    try:
        tool = cast("FunctionTool", capability.tools()[0])
        arguments = json.dumps(
            {
                "requests": [{"sequence": "A", "ph": 3.0}],
                "reasoning": "Test an unmeasured condition.",
            },
        )

        first_result = json.loads(asyncio.run(_invoke_tool(tool, arguments)))
        repeated_result = json.loads(asyncio.run(_invoke_tool(tool, arguments)))

        assert first_result["status"] == "not_found"
        assert repeated_result["status"] == "duplicate"
        assert calls == 1
        assert capability.material_records == [{"sequence": "B", "ph": 7.0}]
    finally:
        capability.client.close()


def test_capability_accepts_a_batch_when_any_result_is_found() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/data":
            return _transport(request)
        payload = cast("dict[str, object]", json.loads(request.content))
        requests = cast("list[dict[str, object]]", payload["requests"])
        return httpx.Response(
            200,
            json={
                "results": [
                    {"request": requests[0], "found": False},
                    {
                        "request": requests[1],
                        "found": True,
                        "values": {"surface_tension": 50.0},
                    },
                ],
            },
        )

    capability = create_lab_capability(
        "http://lab.test",
        transport=httpx.MockTransport(transport),
    )
    try:
        tool = cast("FunctionTool", capability.tools()[0])
        result = asyncio.run(
            _invoke_tool(
                tool,
                json.dumps(
                    {
                        "requests": [
                            {"sequence": "A", "ph": 3.0},
                            {"sequence": "B", "ph": 7.0},
                        ],
                        "reasoning": "Compare two conditions.",
                    },
                ),
            ),
        )

        assert json.loads(result)["status"] == "accepted"
    finally:
        capability.client.close()


def test_capability_logs_each_lab_request(capability, caplog) -> None:
    tool = cast("FunctionTool", capability.tools()[0])
    caplog.set_level(logging.INFO, logger="wiki_agents.agents.experimentalist.lab")

    asyncio.run(
        _invoke_tool(
            tool,
            json.dumps(
                {
                    "requests": [{"sequence": "A", "ph": 3.0}],
                    "reasoning": "This point tests the current trend.",
                },
            ),
        ),
    )

    assert 'requests=[{"ph": 3.0, "sequence": "A"}]' in caplog.text
    assert "reasoning=This point tests the current trend." in caplog.text


def test_capability_mounts_all_paginated_materials(capability) -> None:
    manifest = capability.process_manifest(Manifest())
    entry = cast("File", manifest.entries[Path("inputs/available_materials.csv")])

    assert entry.content == b"sequence,ph\nA,3.0\nB,7.0\n"


def test_capability_refreshes_its_owned_material_manifest_entry(capability) -> None:
    manifest = capability.process_manifest(Manifest())
    capability.set_campaign_state(
        materials=[{"sequence": "B", "ph": 7.0}],
        remaining_budget=1,
        observations=[],
    )

    refreshed = capability.process_manifest(manifest)
    entry = cast("File", refreshed.entries[Path("inputs/available_materials.csv")])

    assert entry.content == b"sequence,ph\nB,7.0\n"


def test_capability_instructions_describe_contract(capability) -> None:
    instructions = asyncio.run(capability.instructions(Manifest()))

    assert "inputs/available_materials.csv" in instructions
    assert "sequence, ph" in instructions
    assert "surface_tension" in instructions
    assert "exact matches" in instructions
    assert "reasoning" in instructions


def test_capability_rejects_manifest_path_collision(capability) -> None:
    manifest = Manifest(
        entries={"inputs/available_materials.csv": File(content=b"owned")},
    )

    with pytest.raises(ValueError, match="collision"):
        capability.process_manifest(manifest)


def test_capability_reports_connection_failure() -> None:
    def fail(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    with pytest.raises(RuntimeError, match="Could not fetch lab schema"):
        create_lab_capability(transport=httpx.MockTransport(fail))


@pytest.mark.parametrize("reasoning", ["", "   "])
def test_capability_requires_a_nonempty_rationale(capability, reasoning: str) -> None:
    tool = cast("FunctionTool", capability.tools()[0])
    arguments = json.dumps(
        {"requests": [{"sequence": "A", "ph": 3.0}], "reasoning": reasoning},
    )

    result = asyncio.run(_invoke_tool(tool, arguments))

    assert json.loads(result)["status"] == "rejected"


def test_capability_returns_cached_duplicate_without_forwarding() -> None:
    calls = 0

    def transport(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/v1/data":
            calls += 1
        return _transport(request)

    observation: dict[str, object] = {
        "request": {"sequence": "A", "ph": 3.0},
        "reasoning": "Earlier rationale.",
        "result": {"found": True, "values": {"surface_tension": 50.0}},
    }
    capability = create_lab_capability(
        "http://lab.test",
        observations=[observation],
        transport=httpx.MockTransport(transport),
    )
    try:
        tool = cast("FunctionTool", capability.tools()[0])
        arguments = json.dumps(
            {
                "requests": [{"sequence": "A", "ph": 3.0}],
                "reasoning": "Repeat the earlier point.",
            },
        )
        result = asyncio.run(_invoke_tool(tool, arguments))

        payload = json.loads(result)
        assert payload["status"] == "duplicate"
        assert payload["cached_results"] == [observation]
        assert calls == 0
    finally:
        capability.client.close()


def test_capability_rejects_an_unavailable_batch_atomically() -> None:
    calls = 0

    def transport(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/v1/data":
            calls += 1
        return _transport(request)

    capability = create_lab_capability(
        "http://lab.test",
        transport=httpx.MockTransport(transport),
    )
    try:
        tool = cast("FunctionTool", capability.tools()[0])
        arguments = json.dumps(
            {
                "requests": [
                    {"sequence": "A", "ph": 3.0},
                    {"sequence": "missing", "ph": 9.0},
                ],
                "reasoning": "Compare an available and unavailable point.",
            },
        )
        result = asyncio.run(_invoke_tool(tool, arguments))

        assert json.loads(result)["status"] == "unavailable"
        assert calls == 0
    finally:
        capability.client.close()


def test_capability_clamps_to_campaign_batch_size() -> None:
    forwarded: list[dict[str, object]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/data":
            forwarded.append(cast("dict[str, object]", json.loads(request.content)))
        return _transport(request)

    capability = create_lab_capability(
        "http://lab.test",
        batch_size=1,
        transport=httpx.MockTransport(transport),
    )
    try:
        tool = cast("FunctionTool", capability.tools()[0])
        arguments = json.dumps(
            {
                "requests": [
                    {"sequence": "A", "ph": 3.0},
                    {"sequence": "B", "ph": 7.0},
                ],
                "reasoning": "Submit a ranked pair.",
            },
        )
        result = asyncio.run(_invoke_tool(tool, arguments))

        assert json.loads(result)["status"] == "accepted"
        assert forwarded == [{"requests": [{"sequence": "A", "ph": 3.0}]}]
    finally:
        capability.client.close()
