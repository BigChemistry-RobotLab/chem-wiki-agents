"""Define source-document and provenance models."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

LineId = Annotated[
    str,
    Field(
        pattern=r"^p\d{4,}-l\d{4,}$",
        description="The anchor ID of a source line. The format is `p####-l####`.",
    ),
]

FigureAnchorId = Annotated[
    str,
    Field(
        pattern=r"^p\d{4,}-figure-\d{4,}$",
        description="The anchor ID of a source figure.",
    ),
]
TableAnchorId = Annotated[
    str,
    Field(
        pattern=r"^p\d{4,}-table-\d{4,}$",
        description="The anchor ID of a source table.",
    ),
]
RegionAnchorId = Annotated[
    str,
    Field(
        pattern=r"^p\d{4,}-region-\d{4,}$",
        description="The anchor ID of a source-page region.",
    ),
]
VisualAnchorId = FigureAnchorId | TableAnchorId | RegionAnchorId

NormalizedCoordinate = Annotated[float, Field(ge=0.0, le=1.0)]


def _line_position(line_id: str) -> tuple[int, int]:
    page, line = line_id.removeprefix("p").split("-l", maxsplit=1)
    return int(page), int(line)


class EvidenceSpan(BaseModel):
    """A continuous source passage that directly supports a claim."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["text"] = "text"
    start_line_id: LineId
    end_line_id: LineId

    @model_validator(mode="after")
    def validate_line_order(self) -> Self:
        """Require an inclusive source range ordered by page and line."""
        if _line_position(self.end_line_id) < _line_position(self.start_line_id):
            raise ValueError("end_line_id must not precede start_line_id")
        return self


class FigureEvidence(BaseModel):
    """Identify a source figure that directly supports a claim."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["figure"] = "figure"
    anchor_id: FigureAnchorId
    label: str = Field(min_length=1)


class TableEvidence(BaseModel):
    """Identify a source table that directly supports a claim."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["table"] = "table"
    anchor_id: TableAnchorId
    label: str = Field(min_length=1)


class RegionEvidence(BaseModel):
    """Identify a normalized source-page region that supports a claim."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["region"] = "region"
    anchor_id: RegionAnchorId
    label: str = Field(min_length=1)
    bbox: tuple[
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
    ]

    @model_validator(mode="after")
    def validate_bbox_order(self) -> Self:
        """Require a non-empty box in left, top, right, bottom order."""
        left, top, right, bottom = self.bbox
        if right <= left or bottom <= top:
            raise ValueError("bbox must define a non-empty normalized region")
        return self


Evidence = EvidenceSpan | FigureEvidence | TableEvidence | RegionEvidence


class SourceRef(BaseModel):
    """Identify one source document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resource: str = Field(pattern=r"^/")


class AnchoredPage(BaseModel):
    """Keep one PDF page with its page number."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1)
    text: str
