"""Run report-centered experimental campaigns in an isolated sandbox."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import shutil
import tarfile
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass
from itertools import batched
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

from agents import RunConfig, Runner
from agents.sandbox import Manifest, SandboxRunConfig
from agents.sandbox.config import DEFAULT_PYTHON_SANDBOX_IMAGE
from agents.sandbox.entries import BaseEntry, Dir, File
from agents.sandbox.sandboxes.docker import (
    DockerSandboxClient,
    DockerSandboxClientOptions,
)
from agents.sandbox.session import manifest_ops
from agents.sandbox.types import FileMode, Permissions, User
from docker import from_env as docker_from_env

from wiki_agents.agents.experimentalist.lab_api import materials_to_csv
from wiki_agents.agents.experimentalist.lab_tools import (
    LabCapability,
    create_lab_capability,
)
from wiki_agents.agents.experimentalist.report import (
    GoalAssessment,
    ReporterResult,
    create_experimentalist_reporter_agent,
    validate_report_markdown,
)
from wiki_agents.agents.experimentalist.research import create_experimentalist_agent
from wiki_agents.tracing import get_run_id, init_weave_tracing

if TYPE_CHECKING:
    from agents.result import RunResult
    from agents.sandbox import SandboxAgent
    from agents.sandbox.session.base_sandbox_session import BaseSandboxSession

_DEFAULT_MODEL = "openai/gpt-5.6-luna"
_REPORTER_TURNS = 20
_WIKI_ARCHIVE_CHUNK_SIZE = 250
logger = logging.getLogger("wiki_agents.orchestrator.experimentalist")


@dataclass(frozen=True, slots=True)
class ExperimentalistOrchestratorConfig:
    """Configure one persistent experimental campaign."""

    wiki_path: Path | None
    campaign_path: str | Path
    task: str
    lab_url: str | None = None
    request_budget: int = 100
    batch_size: int = 1
    researcher_turns: int = 30
    researcher_model: str = _DEFAULT_MODEL
    curator_model: str = _DEFAULT_MODEL

    def __post_init__(self) -> None:
        """Validate campaign settings."""
        if not self.task.strip():
            raise ValueError("task cannot be empty")
        for name, value in {
            "request_budget": self.request_budget,
            "batch_size": self.batch_size,
            "researcher_turns": self.researcher_turns,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")


class ExperimentalistOrchestrator:
    """Coordinate researcher sessions, lab requests, and report updates."""

    def __init__(
        self,
        config: ExperimentalistOrchestratorConfig,
        *,
        capability: LabCapability | None = None,
    ) -> None:
        """Prepare an orchestrator without accessing Docker."""
        self.config = config
        self.wiki_path = (
            None if config.wiki_path is None else Path(config.wiki_path).resolve()
        )
        self.campaign_path = Path(config.campaign_path).resolve()
        self.capability = capability
        self._accepted: list[dict[str, object]] = []
        self._active_curation_group_id: str | None = None

    def run(self) -> None:
        """Run or resume the campaign until budget or materials are exhausted."""
        init_weave_tracing()
        logger.info(
            "Starting experimental campaign: campaign=%s request_budget=%d",
            self.campaign_path,
            self.config.request_budget,
        )
        asyncio.run(self._run())

    async def _run(self) -> None:
        capability = self.capability or create_lab_capability(
            self.config.lab_url,
            batch_size=self.config.batch_size,
            remaining_budget=self.config.request_budget,
            on_accepted=self._record_accepted,
        )
        capability.on_accepted = self._record_accepted
        self.capability = capability
        self._load_or_initialize_campaign(capability)

        client = DockerSandboxClient(docker_from_env())
        session = await client.create(
            manifest=self._build_session_manifest(),
            options=DockerSandboxClientOptions(
                image=DEFAULT_PYTHON_SANDBOX_IMAGE,
                network_mode="none",
            ),
        )
        try:
            async with session:
                observations = self._read_observations()
                remaining = self._remaining_materials(capability, observations)
                manifest = capability.process_manifest(
                    self._build_manifest(observations, remaining),
                )
                await self._materialize_workspace(session, manifest)

                include_wiki = self.wiki_path is not None
                researcher = create_experimentalist_agent(
                    capability,
                    model=self.config.researcher_model,
                    include_wiki=include_wiki,
                )
                reporter = create_experimentalist_reporter_agent(
                    model=self.config.curator_model,
                    include_wiki=include_wiki,
                )
                await self._process_pending_collation(
                    reporter=reporter,
                    session=session,
                )

                while self._campaign_status() == "running":
                    observations = self._read_observations()
                    valid_requests = self._valid_request_count(observations)
                    remaining = self._remaining_materials(capability, observations)
                    if not remaining:
                        self._set_campaign_outcome("materials_exhausted", valid_requests)
                        break
                    budget = max(self.config.request_budget - valid_requests, 0)
                    if budget == 0:
                        self._set_campaign_outcome("budget_exhausted", valid_requests)
                        break

                    capability.set_campaign_state(
                        materials=remaining,
                        remaining_budget=budget,
                        observations=observations,
                    )
                    await self._refresh_session_inputs(session, capability, observations)
                    self._accepted.clear()
                    self._start_curation_group()
                    logger.info(
                        "Starting researcher agent: completed=%d remaining=%d",
                        valid_requests,
                        budget,
                    )
                    result = await Runner.run(
                        researcher,
                        self._researcher_input(valid_requests, budget),
                        max_turns=self.config.researcher_turns,
                        run_config=RunConfig(
                            sandbox=SandboxRunConfig(session=session),
                            workflow_name="Experimental campaign research",
                            group_id=get_run_id(),
                            trace_metadata={
                                "campaign_path": str(self.campaign_path),
                                "completed_requests": valid_requests,
                                "remaining_budget": budget,
                            },
                        ),
                    )
                    if not self._accepted:
                        self._clear_pending_collation()
                        raise RuntimeError(
                            "Researcher session ended without an accepted lab request",
                        )
                    transcript_path = self._write_transcript(result)
                    pending = self._require_pending_collation()
                    pending["transcript_path"] = transcript_path.relative_to(
                        self.campaign_path,
                    ).as_posix()
                    pending["phase"] = "transcript_persisted"
                    self._write_pending_collation(pending)
                    await self._mirror_work(session)
                    await self._process_pending_collation(
                        reporter=reporter,
                        session=session,
                    )
                    self._active_curation_group_id = None
        finally:
            await client.delete(session)
            capability.client.close()

    def _load_or_initialize_campaign(
        self,
        capability: LabCapability,
    ) -> dict[str, object]:
        """Create campaign artifacts or validate immutable resume fields."""
        if self.wiki_path is not None and not self.wiki_path.is_dir():
            raise ValueError(f"Wiki path is not a directory: {self.wiki_path}")
        strategy_path = self.campaign_path / "strategy.json"
        if strategy_path.exists():
            raise ValueError(
                "This directory contains a planner-era campaign. Start a new campaign "
                "directory; planner-format campaigns cannot be resumed.",
            )

        fingerprint = hashlib.sha256(
            json.dumps(capability.schema_data, sort_keys=True).encode("utf-8"),
        ).hexdigest()
        immutable = {
            "task": self.config.task,
            "lab_schema_fingerprint": fingerprint,
            "configuration": self._serialized_config(),
        }
        path = self.campaign_path / "campaign.json"
        if path.exists():
            campaign = self._read_json_object(path)
            configuration = campaign.get("configuration")
            if isinstance(configuration, dict) and any(
                key in configuration
                for key in (
                    "planner_model",
                    "planner_turns",
                    "strategy_review_patience",
                )
            ):
                raise ValueError(
                    "This directory contains planner-era configuration. Start a new "
                    "campaign directory; planner-format campaigns cannot be resumed.",
                )
            for key, expected in immutable.items():
                if campaign.get(key) != expected:
                    raise ValueError(f"Campaign resume mismatch for {key}")
            validate_report_markdown(
                (self.campaign_path / "report.md").read_text(encoding="utf-8"),
            )
            return campaign

        self.campaign_path.mkdir(parents=True, exist_ok=True)
        (self.campaign_path / "sessions").mkdir(exist_ok=True)
        (self.campaign_path / "work").mkdir(exist_ok=True)
        self._write_json(self.campaign_path / "requests.json", [])
        self._write_text(self.campaign_path / "observations.jsonl", "")
        self._write_text(
            self.campaign_path / "report.md",
            self._initial_report(self.config.task),
        )
        campaign = {
            **immutable,
            "progress": {
                "accepted_requests": 0,
                "request_budget": self.config.request_budget,
            },
            "status": "running",
        }
        self._write_json(path, campaign)
        return campaign

    def _serialized_config(self) -> dict[str, object]:
        """Return stable campaign configuration for resume validation."""
        data = asdict(self.config)
        data["campaign_path"] = str(self.campaign_path)
        data.pop("task")
        if self.wiki_path is None:
            data.pop("wiki_path")
        else:
            data["wiki_path"] = str(self.wiki_path)
        return cast("dict[str, object]", data)

    def _build_manifest(
        self,
        observations: list[dict[str, object]],
        remaining: list[dict[str, object]],
    ) -> Manifest:
        """Build researcher and reporter workspaces with persistent work files."""
        capability = cast("LabCapability", self.capability)
        entries: dict[str, File | Dir] = {
            "inputs": Dir(permissions=_read_only_directory()),
            "work": Dir(
                permissions=_writable_directory(),
                group=User(name="researcher"),
            ),
            "inputs/planning_context.json": File(
                content=self._planning_context_bytes(capability=capability),
                permissions=_read_only_file(),
            ),
            "inputs/task.md": File(
                content=self.config.task.encode("utf-8"),
                permissions=_read_only_file(),
            ),
            "inputs/report.md": File(
                content=(self.campaign_path / "report.md").read_bytes(),
                permissions=_read_only_file(),
            ),
            "inputs/observations.jsonl": File(
                content=_jsonl_bytes(observations),
                permissions=_read_only_file(),
            ),
        }
        if self.wiki_path is not None:
            entries["wiki"] = Dir(permissions=_read_only_directory())
            for path in sorted(self.wiki_path.rglob("*.md")):
                relative = path.relative_to(self.wiki_path).as_posix()
                entries[f"wiki/{relative}"] = File(
                    content=path.read_bytes(),
                    permissions=_read_only_file(),
                )
        self._add_work_entries(entries)
        capability.materials_csv = materials_to_csv(
            cast("list[dict[str, str]]", capability.schema_data["materials"]),
            remaining,
        )
        return Manifest(
            entries=entries,
            users=[User(name="researcher"), User(name="reporter")],
        )

    def _add_work_entries(self, entries: dict[str, File | Dir]) -> None:
        """Add persistent work files with researcher-only write access."""
        work_path = self.campaign_path / "work"
        work_directories: set[Path] = set()
        for path in sorted(item for item in work_path.rglob("*") if item.is_file()):
            relative = path.relative_to(work_path).as_posix()
            parent = Path(relative).parent
            while parent != Path():
                work_directories.add(parent)
                parent = parent.parent
            entries[f"work/{relative}"] = File(
                content=path.read_bytes(),
                permissions=_writable_file(),
                group=User(name="researcher"),
            )
        for directory in sorted(work_directories):
            entries[f"work/{directory.as_posix()}"] = Dir(
                permissions=_writable_directory(),
                group=User(name="researcher"),
            )

    def _planning_context_bytes(self, *, capability: LabCapability) -> bytes:
        """Serialize stable public planning context."""
        return (
            json.dumps(
                {
                    "request_budget": self.config.request_budget,
                    "request_batch_size": self.config.batch_size,
                    "lab_schema": capability.schema_data,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def _build_session_manifest() -> Manifest:
        """Build the model-facing manifest without a file inventory."""
        return Manifest(users=[User(name="researcher"), User(name="reporter")])

    @staticmethod
    async def _materialize_workspace(
        session: BaseSandboxSession,
        manifest: Manifest,
        *,
        materialize_wiki: bool = True,
    ) -> None:
        """Add workspace files without changing the model-facing manifest."""
        started = time.monotonic()
        base_manifest, wiki_files = _partition_wiki_files(manifest)
        wiki_bytes = sum(len(entry.content) for _, entry in wiki_files)
        logger.info(
            "Materializing workspace: entries=%d wiki_files=%d wiki_bytes=%d",
            len(base_manifest.entries),
            len(wiki_files),
            wiki_bytes,
        )
        await manifest_ops.apply_manifest(
            session,
            manifest=base_manifest,
            provision_accounts=False,
        )
        if materialize_wiki:
            chunks = tuple(batched(wiki_files, _WIKI_ARCHIVE_CHUNK_SIZE))
            completed = 0
            for index, chunk in enumerate(chunks, start=1):
                archive = _build_wiki_archive(chunk)
                archive_path = Path(f".wiki-materialization-{index:04d}.tar")
                await session.write(archive_path, archive)
                try:
                    result = await session.exec(
                        "tar",
                        "-xf",
                        archive_path,
                        "-C",
                        ".",
                        shell=False,
                    )
                    if not result.ok():
                        stderr = result.stderr.decode("utf-8", errors="replace")
                        raise RuntimeError(
                            f"Could not extract Wiki archive chunk {index}: {stderr}",
                        )
                finally:
                    await session.rm(archive_path)
                completed += len(chunk)
                logger.info(
                    "Materialized Wiki archive: chunk=%d/%d completed=%d/%d elapsed=%.1fs",
                    index,
                    len(chunks),
                    completed,
                    len(wiki_files),
                    time.monotonic() - started,
                )
        logger.info(
            "Workspace materialization complete: wiki_files=%d elapsed=%.1fs",
            len(wiki_files),
            time.monotonic() - started,
        )

    async def _refresh_session_inputs(
        self,
        session: BaseSandboxSession,
        capability: LabCapability,
        observations: list[dict[str, object]],
    ) -> None:
        """Refresh campaign evidence before a researcher session."""
        await _session_write(
            session,
            "inputs/available_materials.csv",
            capability.materials_csv,
        )
        await _session_write(
            session,
            "inputs/observations.jsonl",
            _jsonl_bytes(observations),
        )
        await _session_write(
            session,
            "inputs/report.md",
            (self.campaign_path / "report.md").read_bytes(),
        )
        transcript = Path("inputs/latest_researcher_session.json")
        with suppress(Exception):
            await session.rm(transcript)

    async def _refresh_reporter_inputs(
        self,
        session: BaseSandboxSession,
        observations: list[dict[str, object]],
        pending: dict[str, object],
    ) -> None:
        """Expose current read-only evidence for one reporter pass."""
        capability = cast("LabCapability", self.capability)
        remaining = self._remaining_materials(capability, observations)
        materials = materials_to_csv(
            cast("list[dict[str, str]]", capability.schema_data["materials"]),
            remaining,
        )
        await _session_write(session, "inputs/available_materials.csv", materials)
        await _session_write(
            session,
            "inputs/observations.jsonl",
            _jsonl_bytes(observations),
        )
        await _session_write(
            session,
            "inputs/report.md",
            (self.campaign_path / "report.md").read_bytes(),
        )
        transcript_path = pending.get("transcript_path")
        if isinstance(transcript_path, str):
            await _session_write(
                session,
                "inputs/latest_researcher_session.json",
                (self.campaign_path / transcript_path).read_bytes(),
            )

    def _record_accepted(
        self,
        requests: list[dict[str, object]],
        reasoning: str,
        response: dict[str, object],
    ) -> None:
        """Persist accepted measurements before model execution can continue."""
        pending = self._require_pending_collation()
        group_id = cast("str", pending["group_id"])
        raw_results = response.get("results")
        if not isinstance(raw_results, list) or len(raw_results) != len(requests):
            raise RuntimeError("Lab response does not match the accepted request batch")

        audit = self._read_requests()
        observations = self._read_observations()
        accepted_records: list[dict[str, object]] = []
        for request, raw_result in zip(requests, raw_results, strict=True):
            if not isinstance(raw_result, dict):
                raise TypeError("Lab result must be a JSON object")
            result = cast("dict[str, object]", raw_result)
            audit.append({"request": request, "reasoning": reasoning})
            record: dict[str, object] = {
                "request": request,
                "reasoning": reasoning,
                "result": result,
                "curation_group_id": group_id,
            }
            observations.append(record)
            accepted_records.append(record)
            logger.info(
                "Lab data recorded: request=%s result=%s",
                json.dumps(request, sort_keys=True),
                json.dumps(result, sort_keys=True, default=str),
            )
        self._write_json(self.campaign_path / "requests.json", audit)
        self._write_observations(observations)
        pending["observation_indices"] = [
            index
            for index, observation in enumerate(observations)
            if observation.get("curation_group_id") == group_id
        ]
        pending["phase"] = "measurement_recorded"
        self._write_pending_collation(pending)
        self._accepted.append(
            {"reasoning": reasoning, "records": accepted_records, "response": response},
        )

    async def _process_pending_collation(  # ruff: ignore[complex-structure,too-many-branches,too-many-statements] - Recovery phases are explicit.
        self,
        *,
        reporter: SandboxAgent[None],
        session: BaseSandboxSession,
    ) -> None:
        """Complete the per-observation report journal without duplicate calls."""
        pending = self._pending_collation()
        if pending is None:
            return
        observations = self._read_observations()
        group_id = cast("str", pending["group_id"])
        indices = [
            index
            for index, observation in enumerate(observations)
            if observation.get("curation_group_id") == group_id
        ]
        pending["observation_indices"] = indices
        if not indices:
            if pending.get("phase") == "recording":
                self._clear_pending_collation()
                return
            raise RuntimeError("A pending curation group has no recorded observations")

        if not isinstance(pending.get("transcript_path"), str):
            recovered_transcript = self._find_group_transcript(group_id)
            if recovered_transcript is not None:
                pending["transcript_path"] = recovered_transcript
                pending["phase"] = "transcript_persisted"
                self._write_pending_collation(pending)
        if not isinstance(pending.get("transcript_path"), str):
            pending["transcript_unavailable"] = True
            if pending.get("phase") == "measurement_recorded":
                pending["phase"] = "transcript_unavailable"
            self._write_pending_collation(pending)

        offset = cast("int", pending.get("observation_offset", 0))
        while offset < len(indices):
            index = indices[offset]
            record = observations[index]
            phase = cast("str", pending.get("phase", "transcript_persisted"))
            raw_draft = pending.get("reporter_draft")
            if phase in {"report_published", "transcript_persisted", "transcript_unavailable"}:
                if phase == "report_published":
                    offset += 1
                    pending["observation_offset"] = offset
                    pending["phase"] = (
                        "transcript_persisted"
                        if isinstance(pending.get("transcript_path"), str)
                        else "transcript_unavailable"
                    )
                    pending.pop("reporter_draft", None)
                    self._write_pending_collation(pending)
                    continue
                await self._refresh_reporter_inputs(session, observations, pending)
                output = await self._run_reporter(
                    reporter,
                    self._reporter_payload(record, observations, index, pending),
                    record,
                    self._valid_request_count(observations),
                    session,
                )
                output = self._normalized_reporter_result(record, output)
                pending["reporter_draft"] = output.model_dump(mode="json")
                pending["phase"] = "reporter_draft_persisted"
                self._write_pending_collation(pending)
                raw_draft = pending["reporter_draft"]

            if not isinstance(raw_draft, dict):
                raise TypeError("pending reporter_draft must be an object")
            draft = ReporterResult.model_validate(raw_draft)
            if pending.get("phase") == "reporter_draft_persisted":
                self._attach_curation(observations, index, draft)
                self._write_observations(observations)
                pending["phase"] = "assessment_attached"
                self._write_pending_collation(pending)
            if pending.get("phase") == "assessment_attached":
                self._publish_report(draft.report_markdown)
                pending["phase"] = "report_published"
                self._write_pending_collation(pending)

        valid_requests = self._valid_request_count(observations)
        if valid_requests >= self.config.request_budget:
            self._set_campaign_outcome("budget_exhausted", valid_requests)
        elif not self._remaining_materials(
            cast("LabCapability", self.capability),
            observations,
        ):
            self._set_campaign_outcome("materials_exhausted", valid_requests)
        else:
            self._set_campaign_outcome("running", valid_requests)
        self._finish_pending_collation(pending)

    async def _run_reporter(
        self,
        reporter: SandboxAgent[None],
        payload: dict[str, object],
        record: dict[str, object],
        valid_requests: int,
        session: BaseSandboxSession,
    ) -> ReporterResult:
        """Run and validate one reporter result."""
        result = await Runner.run(
            reporter,
            json.dumps(payload, sort_keys=True),
            max_turns=_REPORTER_TURNS,
            run_config=RunConfig(
                sandbox=SandboxRunConfig(session=session),
                workflow_name="Experimental campaign reporting",
                group_id=get_run_id(),
                trace_metadata={
                    "campaign_path": str(self.campaign_path),
                    "accepted_requests": valid_requests,
                    "request": record["request"],
                },
            ),
        )
        return result.final_output_as(ReporterResult, raise_if_incorrect_type=True)

    def _reporter_payload(
        self,
        record: dict[str, object],
        observations: list[dict[str, object]],
        record_index: int,
        pending: dict[str, object],
    ) -> dict[str, object]:
        """Build reporter context and point to inspectable evidence files."""
        return {
            "workspace": "/workspace",
            "current_observation_index": record_index,
            "latest_measurement": {
                "request": record["request"],
                "result": record["result"],
                "selection_rationale": record.get("reasoning"),
            },
            "researcher_transcript": (
                "inputs/latest_researcher_session.json"
                if isinstance(pending.get("transcript_path"), str)
                else (
                    "Unavailable: the process stopped after measurement acceptance "
                    "and before the visible researcher session could be persisted. "
                    "Use the stored rationale, ledger, report, and work files."
                )
            ),
            "evidence_paths": {
                "task": "inputs/task.md",
                "planning_context": "inputs/planning_context.json",
                "current_report": "inputs/report.md",
                "complete_observation_ledger": "inputs/observations.jsonl",
                "remaining_catalog": "inputs/available_materials.csv",
                "persistent_analysis": "work/",
                **({"wiki": "wiki/"} if self.wiki_path is not None else {}),
            },
            "campaign_progress": {
                "recorded_observations": len(observations),
                "valid_requests": self._valid_request_count(observations),
                "request_budget": self.config.request_budget,
            },
        }

    @staticmethod
    def _normalized_reporter_result(
        record: dict[str, object],
        result: ReporterResult,
    ) -> ReporterResult:
        """Force every missing lab result to be inconclusive."""
        measurement = record.get("result")
        if isinstance(measurement, dict) and measurement.get("found") is False:
            return result.model_copy(
                update={
                    "assessment": GoalAssessment.INCONCLUSIVE,
                    "assessment_rationale": (
                        "The lab returned found: false, so the measurement is "
                        "inconclusive against the immutable task."
                    ),
                },
            )
        return result

    @staticmethod
    def _attach_curation(
        observations: list[dict[str, object]],
        index: int,
        result: ReporterResult,
    ) -> None:
        """Attach only the reporter assessment to one observation."""
        observations[index]["curation"] = {
            "assessment": result.assessment.value,
            "assessment_rationale": result.assessment_rationale,
        }

    def _write_transcript(self, result: RunResult) -> Path:
        """Persist all visible user, shell, tool, and output items."""
        sessions = self.campaign_path / "sessions"
        path = sessions / f"session-{len(list(sessions.glob('session-*.json'))) + 1:04d}.json"
        visible_items = [
            item
            for item in result.to_input_list()
            if not isinstance(item, dict) or item.get("type") != "reasoning"
        ]
        self._write_json(
            path,
            {
                "curation_group_id": self._active_curation_group_id,
                "items": visible_items,
            },
        )
        return path

    def _find_group_transcript(self, group_id: str) -> str | None:
        """Find a persisted transcript after a journal-update crash."""
        sessions = self.campaign_path / "sessions"
        for path in sorted(sessions.glob("session-*.json"), reverse=True):
            payload = self._read_json_object(path)
            if payload.get("curation_group_id") == group_id:
                return path.relative_to(self.campaign_path).as_posix()
        return None

    async def _mirror_work(self, session: BaseSandboxSession) -> None:
        """Replace the host work mirror with files from the live sandbox."""
        result = await session.exec("find", "work", "-type", "f", "-print", shell=False)
        if not result.ok():
            raise RuntimeError("Could not inventory sandbox work files")
        source_paths = [
            line for line in result.stdout.decode("utf-8").splitlines() if line
        ]
        target_root = self.campaign_path / "work"
        shutil.rmtree(target_root)
        target_root.mkdir()
        for source in source_paths:
            relative = PurePosixPath(source).relative_to("work")
            target = target_root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = await session.read(Path(source))
            try:
                target.write_bytes(cast("bytes", handle.read()))
            finally:
                handle.close()

    @staticmethod
    def _remaining_materials(
        capability: LabCapability,
        observations: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Remove every accepted experiment from the material catalog."""
        accepted = [item.get("request") for item in observations]
        return [item for item in capability.material_records if item not in accepted]

    @staticmethod
    def _valid_request_count(observations: list[dict[str, object]]) -> int:
        """Count recorded requests for which the lab found data."""
        return sum(
            isinstance(result := observation.get("result"), dict)
            and result.get("found") is True
            for observation in observations
        )

    def _researcher_input(self, completed: int, remaining_budget: int) -> str:
        """Build the short per-session campaign instruction."""
        if completed == 0:
            strategy_sources = (
                "the Wiki, task, and available materials"
                if self.wiki_path is not None
                else "the task and available materials"
            )
            return (
                "Begin the immutable objective in inputs/task.md. No valid experiments "
                f"are recorded and {remaining_budget} budget units remain. First, plan "
                f"the initial global predictive policy. Use {strategy_sources} to evaluate "
                "the remaining catalog before selecting the highest-total-value batch. "
                "End with one accepted request_data call. Verify that your request exists in the "
                "available materials catalog and that it is not a duplicate of any "
                "previously recorded experiment."
            )
        return (
            f"Continue the immutable objective in inputs/task.md. {completed} valid "
            f"experiments are recorded and {remaining_budget} budget units remain. "
            "Update the global predictive policy from all evidence. Select the highest-total-value "
            "batch, use the report guidance as advisory, and "
            "end with one accepted request_data call. Verify that your request exists in the "
            "available materials catalog and that it is not a duplicate of any "
            "previously recorded experiment."
        )

    def _campaign_status(self) -> str:
        """Return the persisted campaign status."""
        status = cast(
            "str",
            self._read_json_object(self.campaign_path / "campaign.json")["status"],
        )
        if status not in {"running", "budget_exhausted", "materials_exhausted"}:
            raise ValueError(f"Unknown campaign status: {status}")
        return status

    def _set_campaign_outcome(self, status: str, valid_requests: int) -> None:
        """Persist progress without changing a terminal status."""
        allowed = {"running", "budget_exhausted", "materials_exhausted"}
        if status not in allowed:
            raise ValueError(f"Unknown campaign status: {status}")
        path = self.campaign_path / "campaign.json"
        campaign = self._read_json_object(path)
        current = cast("str", campaign["status"])
        if current not in {"running", status}:
            return
        campaign["progress"] = {
            "accepted_requests": valid_requests,
            "request_budget": self.config.request_budget,
        }
        campaign["status"] = status
        self._write_json(path, campaign)

    def _pending_collation(self) -> dict[str, object] | None:
        """Return the pending observation journal when it exists."""
        campaign = self._read_json_object(self.campaign_path / "campaign.json")
        value = campaign.get("pending_collation")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("pending_collation must be an object")
        return cast("dict[str, object]", value)

    def _start_curation_group(self, group_id: str | None = None) -> str:
        """Persist a new researcher-session curation group."""
        if self._pending_collation() is not None:
            raise RuntimeError("A curation group is already pending")
        selected_group_id = uuid.uuid4().hex if group_id is None else group_id
        self._active_curation_group_id = selected_group_id
        self._write_pending_collation(
            {
                "group_id": selected_group_id,
                "observation_indices": [],
                "observation_offset": 0,
                "phase": "recording",
            },
        )
        return selected_group_id

    def _require_pending_collation(self) -> dict[str, object]:
        """Return the pending journal or fail on invalid lifecycle use."""
        pending = self._pending_collation()
        if pending is None:
            raise RuntimeError("No curation group is pending")
        return pending

    def _write_pending_collation(self, pending: dict[str, object]) -> None:
        """Persist the current observation journal in campaign state."""
        path = self.campaign_path / "campaign.json"
        campaign = self._read_json_object(path)
        campaign["pending_collation"] = pending
        self._write_json(path, campaign)

    def _clear_pending_collation(self) -> None:
        """Remove the completed or empty observation journal."""
        path = self.campaign_path / "campaign.json"
        campaign = self._read_json_object(path)
        campaign.pop("pending_collation", None)
        self._write_json(path, campaign)

    def _finish_pending_collation(self, pending: dict[str, object]) -> None:
        """Clear the journal only after each observation is published."""
        indices = cast("list[int]", pending["observation_indices"])
        if pending.get("observation_offset") != len(indices):
            raise RuntimeError("Cannot clear an incomplete observation journal")
        self._clear_pending_collation()

    def _update_campaign(self, valid_requests: int) -> None:
        """Update campaign status from host-owned stopping conditions."""
        observations = self._read_observations()
        if valid_requests >= self.config.request_budget:
            self._set_campaign_outcome("budget_exhausted", valid_requests)
        elif not self._remaining_materials(
            cast("LabCapability", self.capability),
            observations,
        ):
            self._set_campaign_outcome("materials_exhausted", valid_requests)
        else:
            self._set_campaign_outcome("running", valid_requests)

    def _read_requests(self) -> list[dict[str, object]]:
        """Read the audit request list."""
        value = self._read_json_value(self.campaign_path / "requests.json")
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise TypeError("requests.json must contain a list of objects")
        return cast("list[dict[str, object]]", value)

    def _read_observations(self) -> list[dict[str, object]]:
        """Read and validate the observation ledger."""
        observations: list[dict[str, object]] = []
        path = self.campaign_path / "observations.jsonl"
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(
                    f"observations.jsonl line {line_number} must be an object",
                )
            observations.append(cast("dict[str, object]", value))
        return observations

    def _write_observations(self, observations: list[dict[str, object]]) -> None:
        """Write the complete observation ledger atomically."""
        self._write_bytes(
            self.campaign_path / "observations.jsonl",
            _jsonl_bytes(observations),
        )

    @staticmethod
    def _initial_report(task: str = "No task is loaded.") -> str:
        """Return the deterministic report before the first measurement."""
        return (
            "# Experimental campaign report\n\n"
            "## Objective and progress\n\n"
            "Immutable task:\n\n"
            f"{task}\n\n"
            "The researcher owns experimental strategy and requests measurements. "
            "The reporter assesses results and gives non-binding guidance.\n\n"
            "## Predictive evidence and current policy\n\n"
            "No predictive evidence or policy yet.\n\n"
            "## Scientific hypotheses and limitations\n\n"
            "No scientific hypotheses or limitations yet.\n\n"
            "## General Observations\n\n"
            "No observations yet.\n\n"
            "## Next ranked options and open questions\n\n"
            "No ranked options or open questions yet.\n"
        )

    def _publish_report(self, report: str) -> None:
        """Validate and publish reporter Markdown without rewriting it."""
        validate_report_markdown(report)
        self._write_text(self.campaign_path / "report.md", report)

    @staticmethod
    def _read_json_value(path: Path) -> object:
        """Read any JSON value from a file."""
        return cast("object", json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def _read_json_object(cls, path: Path) -> dict[str, object]:
        """Read one JSON object from a file."""
        value = cls._read_json_value(path)
        if not isinstance(value, dict):
            raise TypeError(f"Expected a JSON object in {path}")
        return cast("dict[str, object]", value)

    @classmethod
    def _write_json(cls, path: Path, value: object) -> None:
        """Write stable JSON atomically."""
        cls._write_text(
            path,
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        )

    @classmethod
    def _write_text(cls, path: Path, value: str) -> None:
        """Write UTF-8 text atomically."""
        cls._write_bytes(path, value.encode("utf-8"))

    @staticmethod
    def _write_bytes(path: Path, value: bytes) -> None:
        """Write bytes through a sibling temporary file."""
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(value)
        temporary.replace(path)


def _jsonl_bytes(records: list[dict[str, object]]) -> bytes:
    """Serialize records as stable JSON Lines bytes."""
    return "".join(f"{json.dumps(item, sort_keys=True)}\n" for item in records).encode()


def _partition_wiki_files(
    manifest: Manifest,
) -> tuple[Manifest, list[tuple[PurePosixPath, File]]]:
    """Separate Wiki files from entries that need standard materialization."""
    base_entries: dict[str | Path, BaseEntry] = {}
    wiki_files: list[tuple[PurePosixPath, File]] = []
    for raw_path, entry in manifest.entries.items():
        path = PurePosixPath(raw_path.as_posix() if isinstance(raw_path, Path) else raw_path)
        if path.parts[:1] == ("wiki",) and isinstance(entry, File):
            wiki_files.append((path, entry))
        else:
            base_entries[raw_path] = entry
    return manifest.model_copy(update={"entries": base_entries}), wiki_files


def _build_wiki_archive(
    files: tuple[tuple[PurePosixPath, File], ...],
) -> io.BytesIO:
    """Build one deterministic tar archive from trusted Wiki files."""
    archive = io.BytesIO()
    directories = {
        parent
        for path, _ in files
        for parent in path.parents
        if parent != PurePosixPath() and parent != PurePosixPath("wiki")
    }
    with tarfile.open(fileobj=archive, mode="w") as output:
        for directory in sorted(directories):
            info = tarfile.TarInfo(directory.as_posix())
            info.type = tarfile.DIRTYPE
            info.mode = _permissions_mode(_read_only_directory())
            info.mtime = 0
            output.addfile(info)
        for path, entry in files:
            info = tarfile.TarInfo(path.as_posix())
            info.size = len(entry.content)
            info.mode = _permissions_mode(entry.permissions)
            info.mtime = 0
            output.addfile(info, io.BytesIO(entry.content))
    archive.seek(0)
    return archive


def _permissions_mode(permissions: Permissions) -> int:
    """Convert structured permissions to one POSIX mode integer."""
    return (
        int(permissions.owner) << 6
        | int(permissions.group) << 3
        | int(permissions.other)
    )


async def _session_write(
    session: BaseSandboxSession,
    path: str,
    content: bytes,
) -> None:
    """Replace one sandbox input as the host user."""
    await session.write(Path(path), io.BytesIO(content))


def _read_only_file() -> Permissions:
    """Return permissions for model-readable input files."""
    return Permissions(owner=FileMode.READ, group=FileMode.READ, other=FileMode.READ)


def _read_only_directory() -> Permissions:
    """Return permissions for model-readable input directories."""
    mode = FileMode.READ | FileMode.EXEC
    return Permissions(owner=mode, group=mode, other=mode, directory=True)


def _writable_file() -> Permissions:
    """Return permissions for persistent researcher work files."""
    mode = FileMode.READ | FileMode.WRITE
    return Permissions(owner=mode, group=mode, other=FileMode.READ)


def _writable_directory() -> Permissions:
    """Return permissions for the persistent researcher work directory."""
    return Permissions(
        owner=FileMode.ALL,
        group=FileMode.ALL,
        other=FileMode.READ | FileMode.EXEC,
        directory=True,
    )


__all__ = ["ExperimentalistOrchestrator", "ExperimentalistOrchestratorConfig"]
