"""Tests for CSV loading and the simulated lab HTTP API."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import pytest

from wiki_agents.simulated_lab import create_app, load_lab

if TYPE_CHECKING:
    from pathlib import Path

    from starlette.applications import Starlette


def _dataset(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "experiments.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_lab_infers_types_aliases_outputs_and_material_order(
    tmp_path: Path,
) -> None:
    path = _dataset(
        tmp_path,
        "sequence,pH,active,result\nA,3,true,1.5\nB,7,false,\n",
    )

    lab = load_lab(
        path,
        inputs=["sequence", "pH", "active"],
        outputs=["result"],
        aliases={"ph": "pH", "measurement": "result"},
    )

    assert lab.schema() == {
        "inputs": [
            {"name": "sequence", "type": "string"},
            {"name": "ph", "type": "integer"},
            {"name": "active", "type": "boolean"},
        ],
        "outputs": [{"name": "measurement", "type": "number"}],
        "materials": [
            {"name": "sequence", "type": "string"},
            {"name": "ph", "type": "integer"},
            {"name": "active", "type": "boolean"},
        ],
        "required_inputs": ["sequence", "ph", "active"],
        "max_batch_size": 100,
    }
    assert lab.material_records == (
        {"sequence": "A", "ph": 3, "active": True},
        {"sequence": "B", "ph": 7, "active": False},
    )
    assert lab.experiments["B", 7, False] == {"measurement": None}


def test_explicit_type_override_wins(tmp_path: Path) -> None:
    path = _dataset(tmp_path, "code,value\n001,4\n002,5\n")
    lab = load_lab(
        path,
        inputs=["code"],
        outputs=["value"],
        column_types={"code": "string", "value": "number"},
    )

    assert lab.inputs[0].type == "string"
    assert lab.outputs[0].type == "number"
    assert ("001",) in lab.experiments
    assert lab.experiments["001",]["value"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("content", "inputs", "outputs", "message"),
    [
        ("x,y\n1,2\n", ["missing"], ["y"], "Unknown dataset columns"),
        ("x,y\n,2\n", ["x"], ["y"], "missing input values"),
        ("x,y\n1,2\n1,3\n", ["x"], ["y"], "duplicate input key"),
    ],
)
def test_load_lab_rejects_invalid_datasets(
    tmp_path: Path,
    content: str,
    inputs: list[str],
    outputs: list[str],
    message: str,
) -> None:
    path = _dataset(tmp_path, content)

    with pytest.raises(ValueError, match=message):
        load_lab(path, inputs=inputs, outputs=outputs)


def test_load_lab_rejects_invalid_mappings(tmp_path: Path) -> None:
    path = _dataset(tmp_path, "x,y\n1,2\n")

    with pytest.raises(ValueError, match="unselected"):
        load_lab(path, inputs=["x"], outputs=["y"], aliases={"other": "z"})
    with pytest.raises(ValueError, match="Invalid type"):
        load_lab(
            path,
            inputs=["x"],
            outputs=["y"],
            column_types={"x": "decimal"},
        )
    with pytest.raises(ValueError, match="unknown CSV column"):
        load_lab(
            path,
            inputs=["x"],
            outputs=["y"],
            column_types={"z": "string"},
        )


def test_external_materials_are_typed_aliased_and_deduplicated(tmp_path: Path) -> None:
    path = _dataset(tmp_path, "sequence,pH,result\nA,3,1\nB,7,2\n")
    materials = tmp_path / "materials.csv"
    materials.write_text("sequence,pH\nB,7\nA,3\nB,7\n", encoding="utf-8")

    lab = load_lab(
        path,
        inputs=["sequence", "pH"],
        outputs=["result"],
        aliases={"ph": "pH"},
        materials_path=materials,
        material_fields=["sequence", "pH"],
    )

    assert lab.material_records == (
        {"sequence": "B", "ph": 7},
        {"sequence": "A", "ph": 3},
    )


@pytest.fixture
def api_client(tmp_path: Path) -> Starlette:
    path = _dataset(
        tmp_path,
        "sequence,pH,tension,concentration\nA,3,62.3,0.5\nB,7,58.1,1.0\n",
    )
    lab = load_lab(
        path,
        inputs=["sequence", "pH"],
        outputs=["tension", "concentration"],
        aliases={"ph": "pH"},
        max_batch_size=2,
    )
    return create_app(lab)


def _request(
    app: Starlette,
    method: str,
    path: str,
    *,
    payload: object | None = None,
) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test",
        ) as client:
            return await client.request(method, path, json=payload)

    return asyncio.run(send())


def test_api_schema_health_and_material_pagination(api_client: Starlette) -> None:
    assert _request(api_client, "GET", "/health").json() == {
        "status": "ok",
        "dataset_rows": 2,
    }
    assert _request(api_client, "GET", "/v1/schema").json()["required_inputs"] == [
        "sequence",
        "ph",
    ]
    assert _request(api_client, "GET", "/v1/materials?offset=1&limit=1").json() == {
        "items": [{"sequence": "B", "ph": 7}],
        "total": 2,
        "offset": 1,
        "limit": 1,
    }


def test_api_preserves_mixed_batch_order_and_multiple_outputs(
    api_client: Starlette,
) -> None:
    response = _request(
        api_client,
        "POST",
        "/v1/data",
        payload={
            "requests": [{"sequence": "missing", "ph": 4}, {"sequence": "A", "ph": 3}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {"request": {"sequence": "missing", "ph": 4}, "found": False},
            {
                "request": {"sequence": "A", "ph": 3},
                "found": True,
                "values": {"tension": 62.3, "concentration": 0.5},
            },
        ],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"requests": "bad"},
        {"requests": []},
        {"requests": [{"sequence": "A"}]},
        {"requests": [{"sequence": "A", "ph": 3, "extra": 1}]},
        {"requests": [{"sequence": "A", "ph": "3"}]},
        {
            "requests": [
                {"sequence": "A", "ph": 3},
                {"sequence": "B", "ph": 7},
                {"sequence": "C", "ph": 9},
            ],
        },
    ],
)
def test_api_rejects_malformed_batches_as_whole_requests(
    api_client: Starlette,
    payload: object,
) -> None:
    response = _request(api_client, "POST", "/v1/data", payload=payload)

    assert response.status_code == 400
    assert "error" in response.json()
