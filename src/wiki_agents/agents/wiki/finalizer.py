"""Run one corpus owner over the complete proposed Wiki."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from agents import Model, RunConfig, Runner
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell
from agents.sandbox.config import DEFAULT_PYTHON_SANDBOX_IMAGE
from agents.sandbox.entries import File
from agents.sandbox.sandboxes.docker import (
    DockerSandboxClientOptions,
)
from agents.sandbox.session import manifest_ops

from wiki_agents.agents.wiki.workspace import (
    WIKI_PATH,
    directory_entry,
    read_sandbox_wiki,
)
from wiki_agents.docker_sandbox import managed_docker_sandbox_client

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from agents.sandbox.session import BaseSandboxSession

    from wiki_agents.agents.wiki.models import WikiPageManifest
    from wiki_agents.agents.wiki.validation import WikiValidationIssue


def create_wiki_finalizer_agent(
    *,
    model: str | Model | None = None,
) -> SandboxAgent[None]:
    """Create the single agent that corrects cross-page corpus defects."""
    return SandboxAgent(
        name="OKF Wiki corpus finalizer",
        model=model,
        base_instructions="",
        instructions="""Finalize the complete proposed OKF Wiki as one corpus owner.

Inspect all concept pages in the work package, its references, the fixed page
manifest, provenance manifest, and deterministic findings. Correct link, source-footnote,
evidence, frontmatter, and metadata defects. Do not add or delete a concept page.
The workspace contains one bounded work package, not the complete Wiki. Edit only
the concept pages listed in `editable_paths` in the finalization input. Links to
concept pages outside the workspace are valid when the fixed manifest lists them.
Do not edit references, PDFs, indexes, or grouping. Do not add unsupported facts.
Each manifest page contains all accepted relations for that page. Links marked
`selected_for_page` are the only required and allowed internal concept links. You can add a missing
selected link, integrate it into explanatory prose, remove duplicate occurrences,
or repair its exact path or anchor text. State the relationship at the point where
the link is relevant. Remove `Related concepts`, `See also`, link-directory, and
bare-title lists. Do not use generic relationship wording when the manifest gives a
specific connection. Treat concept links as relationships, not provenance. Do not
promote an unselected accepted relation. Do not link a NONE relation, change a MERGE
group, or add or remove a page.
Edit files in the workspace and give a short summary when finished.
""",
        default_manifest=Manifest(),
        capabilities=[Shell()],
        output_type=str,
    )


def validate_finalizer_scope(
    original_files: Mapping[str, bytes],
    resulting_files: Mapping[str, bytes],
    editable_paths: Collection[str],
) -> None:
    """Reject finalizer changes outside the current work package."""
    if original_files.keys() != resulting_files.keys():
        added = sorted(resulting_files.keys() - original_files.keys())
        deleted = sorted(original_files.keys() - resulting_files.keys())
        raise ValueError(
            f"finalizer changed the file set; added={added}, deleted={deleted}",
        )
    editable = {path.removeprefix("wiki/") for path in editable_paths}
    changed = {
        path for path in original_files if original_files[path] != resulting_files[path]
    }
    invalid = sorted(changed - editable)
    if invalid:
        raise ValueError(f"finalizer edited files outside work package: {invalid}")
    for path in changed:
        pure_path = PurePosixPath(path)
        if (
            pure_path.name.lower() in {"index.md", "log.md"}
            or pure_path.suffix == ".pdf"
        ):
            raise ValueError(f"finalizer edited a protected file: {path}")


def _finalizer_workspace(
    task: Mapping[str, object],
    wiki_files: Mapping[str, bytes],
) -> Manifest:
    """Create all files for one finalizer work package."""
    return Manifest(
        entries={
            WIKI_PATH: directory_entry(wiki_files),
            "inputs/finalization.json": File(
                content=(json.dumps(task, indent=2, sort_keys=True) + "\n").encode(),
            ),
        },
    )


def _finalizer_session_manifest() -> Manifest:
    """Build the model-facing manifest without a Wiki file inventory."""
    return Manifest()


async def _materialize_finalizer_workspace(
    session: BaseSandboxSession,
    manifest: Manifest,
) -> None:
    """Add workspace files without changing the model-facing manifest."""
    await manifest_ops.apply_manifest(
        session,
        manifest=manifest,
        provision_accounts=False,
    )


async def _run_finalizer(  # ruff: ignore[too-many-arguments] - Finalization requires the full corpus and correction state.
    agent: SandboxAgent[None],
    wiki_files: Mapping[str, bytes],
    manifest: WikiPageManifest,
    provenance_manifest: Mapping[str, object],
    findings: Sequence[WikiValidationIssue],
    *,
    editable_paths: Collection[str],
    max_turns: int,
) -> dict[str, bytes]:
    """Run one finalizer correction and return its complete Wiki mapping."""
    task = {
        "page_manifest": manifest.model_dump(mode="json"),
        "provenance_manifest": provenance_manifest,
        "validation_findings": [finding.as_payload() for finding in findings],
        "editable_paths": sorted(editable_paths),
    }
    workspace = _finalizer_workspace(task, wiki_files)
    with managed_docker_sandbox_client() as client:
        session = await client.create(
            manifest=_finalizer_session_manifest(),
            options=DockerSandboxClientOptions(image=DEFAULT_PYTHON_SANDBOX_IMAGE),
        )
        try:
            async with session:
                await _materialize_finalizer_workspace(session, workspace)
                await Runner.run(
                    agent,
                    "Finalize the proposed Wiki. Read inputs/finalization.json first.",
                    max_turns=max_turns,
                    run_config=RunConfig(
                        sandbox=SandboxRunConfig(session=session),
                        workflow_name="OKF Wiki corpus finalization",
                    ),
                )
                resulting = await read_sandbox_wiki(session)
                validate_finalizer_scope(wiki_files, resulting, editable_paths)
                return resulting
        finally:
            await client.delete(session)


def run_wiki_finalizer_agent(  # ruff: ignore[too-many-arguments] - Finalization requires the full corpus and correction state.
    agent: SandboxAgent[None],
    wiki_files: Mapping[str, bytes],
    manifest: WikiPageManifest,
    provenance_manifest: Mapping[str, object],
    findings: Sequence[WikiValidationIssue],
    *,
    editable_paths: Collection[str],
    max_turns: int = 30,
    timeout_seconds: float,
) -> dict[str, bytes]:
    """Run one bounded corpus finalizer work package."""
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")

    async def run_with_timeout() -> dict[str, bytes]:
        """Apply the wall-clock limit to one complete package run."""
        try:
            return await asyncio.wait_for(
                _run_finalizer(
                    agent,
                    wiki_files,
                    manifest,
                    provenance_manifest,
                    findings,
                    editable_paths=editable_paths,
                    max_turns=max_turns,
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError as error:
            raise TimeoutError(
                f"Wiki finalizer work package exceeded {timeout_seconds:g} seconds",
            ) from error

    return asyncio.run(run_with_timeout())
