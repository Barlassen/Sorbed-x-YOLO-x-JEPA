"""Quantitative wound metrics — geometry, tissue composition, and healing scores.

Every field here is either computed from real pixels or explicitly ``None`` when
it cannot be derived from a 2D image (depth, undermining) or when the image is
uncalibrated (physical sizes). ``None`` never means "zero".
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sorbed.domain.enums import TissueClass


class GeometryMetrics(BaseModel):
    """Shape and size of the segmented wound region.

    Length and width follow the clinical convention: length is the greatest
    head-to-toe extent, width the greatest extent perpendicular to it. When the
    head direction is unknown they fall back to the mask's major/minor axes and
    ``axis_convention`` records which was used.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    area_px: float = Field(ge=0)
    perimeter_px: float = Field(ge=0)
    length_px: float = Field(ge=0)
    width_px: float = Field(ge=0)
    major_axis_px: float = Field(ge=0)
    minor_axis_px: float = Field(ge=0)
    circularity: float = Field(ge=0, le=1)
    solidity: float = Field(ge=0, le=1)
    centroid_px: tuple[float, float]
    bbox_px: tuple[int, int, int, int]
    axis_convention: str = Field(default="major_minor")
    wound_fraction_of_image: float = Field(ge=0, le=1)

    # Physical measurements — present only when the image is calibrated.
    area_mm2: float | None = None
    length_mm: float | None = None
    width_mm: float | None = None
    area_cm2: float | None = None


class TissueComposition(BaseModel):
    """Per-tissue area fractions over the wound bed.

    Fractions are taken over the wound region only (background excluded) and sum
    to 1. ``mean_confidence`` records how certain the classifier was per class.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fractions: dict[TissueClass, float]
    areas_px: dict[TissueClass, float]
    areas_mm2: dict[TissueClass, float] | None = None
    dominant: TissueClass
    mean_confidence: dict[TissueClass, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fractions_valid(self) -> TissueComposition:
        for cls, frac in self.fractions.items():
            if not 0.0 <= frac <= 1.0:
                raise ValueError(f"fraction for {cls} out of range: {frac}")
        total = sum(self.fractions.values())
        if self.fractions and abs(total - 1.0) > 1e-3:
            raise ValueError(f"tissue fractions must sum to 1.0, got {total:.4f}")
        return self

    def fraction_of(self, cls: TissueClass) -> float:
        return self.fractions.get(cls, 0.0)

    @property
    def obscured_fraction(self) -> float:
        """Share of the bed covered by slough or eschar (drives Unstageable)."""
        return self.fraction_of(TissueClass.SLOUGH) + self.fraction_of(TissueClass.ESCHAR)

    @property
    def has_deep_structure(self) -> bool:
        return any(
            self.fraction_of(c) > 0.0
            for c in (TissueClass.ADIPOSE, TissueClass.MUSCLE, TissueClass.TENDON_BONE)
        )


class DepthProxy(BaseModel):
    """A shading-derived *relative* depth cue — explicitly not a measurement.

    True depth cannot be recovered from a single 2D photograph. This index is a
    monotone shading heuristic in [0, 1] used only as weak, clearly-flagged
    evidence; ``is_physical_measurement`` is always ``False``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_depth_index: float = Field(ge=0, le=1)
    method: str
    is_physical_measurement: bool = False


class HealingScores(BaseModel):
    """Computable sub-scores of validated healing/monitoring instruments.

    Only the image-derivable sub-scores are filled. Items requiring palpation or
    probing (depth, undermining, induration, exudate) are left ``None`` and must
    be entered by a clinician; totals are therefore reported as partial.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    push_size_subscore: int | None = Field(default=None, ge=0, le=10)
    push_tissue_subscore: int | None = Field(default=None, ge=0, le=4)
    push_partial_total: int | None = Field(default=None, ge=0, le=17)
    design_r_size_subscore: int | None = Field(default=None, ge=0)
    granulation_percent: float | None = Field(default=None, ge=0, le=100)
    notes: list[str] = Field(default_factory=list)


class ColorCues(BaseModel):
    """Color-derived fractions used as staging evidence.

    These persist the deep-tissue (maroon/purple), erythema, and open-bed signals
    the classifier computed, so the staging engine's evidence can point at real,
    reported fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    maroon_purple_fraction: float = Field(ge=0, le=1)
    erythema_fraction: float = Field(ge=0, le=1)
    open_bed_fraction: float = Field(ge=0, le=1)


class PeriwoundFindings(BaseModel):
    """Color/erythema analysis of the skin ring around the wound."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    erythema_index: float | None = None  # a* elevation vs. healthy reference ring
    maceration_suspected: bool = False
    analyzed_ring_px: int = Field(default=0, ge=0)


class Metrics(BaseModel):
    """The complete quantitative description of one wound."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    geometry: GeometryMetrics
    tissue: TissueComposition
    color_cues: ColorCues
    depth_proxy: DepthProxy | None = None
    periwound: PeriwoundFindings | None = None
    healing_scores: HealingScores | None = None
    skin_intact: bool = Field(description="Whether unbroken skin dominates the region.")
    undermining_assessable: bool = Field(
        default=False,
        description="Always False for 2D images; undermining needs probing.",
    )
