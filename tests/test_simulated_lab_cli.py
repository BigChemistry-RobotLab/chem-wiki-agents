"""Tests for the start-lab command-line entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import httpx
import pytest
from starlette.applications import Starlette

from wiki_agents.simulated_lab import cli


def test_cli_loads_configuration_and_starts_uvicorn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("sequence,pH,result\nA,3,1.5\n", encoding="utf-8")
    started: dict[str, object] = {}

    def fake_run(app: Starlette, *, host: str, port: int) -> None:
        started.update(app=app, host=host, port=port)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    cli.main(
        [
            str(dataset),
            "--input",
            "sequence",
            "--input",
            "pH",
            "--output",
            "result",
            "--alias",
            "ph=pH",
            "--host",
            "127.0.0.2",
            "--port",
            "9000",
        ],
    )

    assert isinstance(started["app"], Starlette)
    assert started["host"] == "127.0.0.2"
    assert started["port"] == 9000


def test_cli_rejects_invalid_mapping_syntax() -> None:
    with pytest.raises(SystemExit):
        cli.main(["data.csv", "--alias", "invalid"])


def test_cli_defaults_configure_the_sequence_ph_campaign_lab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: dict[str, Starlette] = {}

    def fake_run(app: Starlette, **_: object) -> None:
        started["app"] = app

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    dataset = Path(__file__).parents[1] / "data" / "dataset.csv"

    cli.main([str(dataset)])

    async def schema() -> dict[str, object]:
        transport = httpx.ASGITransport(app=started["app"])
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test",
        ) as client:
            response = await client.get("/v1/schema")
            payload = response.json()
            assert isinstance(payload, dict)
            return cast("dict[str, object]", payload)

    payload = asyncio.run(schema())
    assert payload["required_inputs"] == ["sequence", "ph"]
    assert payload["outputs"] == [
        {"name": "surface_tension", "type": "number"},
    ]
    assert payload["materials"] == [
        {"name": "sequence", "type": "string"},
        {"name": "ph", "type": "number"},
    ]
