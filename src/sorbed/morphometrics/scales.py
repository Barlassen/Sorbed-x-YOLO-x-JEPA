"""Computable sub-scores of validated wound assessment instruments.

Only the image-derivable items are produced (size, tissue/granulation). Items
needing palpation or probing — exudate, depth, undermining, induration — are left
for a clinician, so totals are explicitly partial. Score bands are transcribed
from PUSH Tool 3.0 and DESIGN-R(R)2020; see ``docs/CLINICAL.md``.
"""

from __future__ import annotations

from sorbed.domain.enums import TissueClass
from sorbed.domain.metrics import HealingScores, TissueComposition

# PUSH 3.0 size sub-score: upper bound (cm^2, exclusive) -> score.
_PUSH_SIZE_BANDS: tuple[tuple[float, int], ...] = (
    (0.0, 0), (0.3, 1), (0.6, 2), (1.0, 3), (2.0, 4), (3.0, 5),
    (4.0, 6), (8.0, 7), (12.0, 8), (24.0, 9),
)
_PUSH_SIZE_MAX = 10

# PUSH 3.0 tissue sub-score by the worst tissue present.
_PUSH_TISSUE_SCORE: dict[TissueClass, int] = {
    TissueClass.EPITHELIAL: 1,
    TissueClass.GRANULATION: 2,
    TissueClass.SLOUGH: 3,
    TissueClass.ESCHAR: 4,
}

# DESIGN-R(R)2020 size item: upper bound (cm^2, exclusive) -> score.
_DESIGN_R_SIZE_BANDS: tuple[tuple[float, int], ...] = (
    (0.0, 0), (4.0, 3), (16.0, 6), (36.0, 8), (64.0, 9), (100.0, 12),
)
_DESIGN_R_SIZE_MAX = 15


def compute_healing_scores(
    composition: TissueComposition,
    area_cm2: float | None,
) -> HealingScores:
    """Assemble the computable healing sub-scores."""
    notes: list[str] = []

    push_size = _band_score(area_cm2, _PUSH_SIZE_BANDS, _PUSH_SIZE_MAX)
    design_size = _band_score(area_cm2, _DESIGN_R_SIZE_BANDS, _DESIGN_R_SIZE_MAX)
    if area_cm2 is None:
        notes.append("Size sub-scores omitted: image is uncalibrated (no scale reference).")

    push_tissue = _worst_tissue_score(composition)
    granulation = round(composition.fraction_of(TissueClass.GRANULATION) * 100.0, 1)

    push_partial = None
    if push_size is not None and push_tissue is not None:
        push_partial = push_size + push_tissue
        notes.append("PUSH total is partial: the exudate sub-score requires clinician input.")

    return HealingScores(
        push_size_subscore=push_size,
        push_tissue_subscore=push_tissue,
        push_partial_total=push_partial,
        design_r_size_subscore=design_size,
        granulation_percent=granulation,
        notes=notes,
    )


def _band_score(
    area_cm2: float | None, bands: tuple[tuple[float, int], ...], top_score: int
) -> int | None:
    """Return the score of the first band the area falls under, else the max.

    ``bands[0]`` is the zero band; the rest carry exclusive upper bounds.
    """
    if area_cm2 is None:
        return None
    if area_cm2 <= 0.0:
        return 0
    for upper, value in bands[1:]:
        if area_cm2 < upper:
            return value
    return top_score


def _worst_tissue_score(composition: TissueComposition) -> int | None:
    present = [c for c in _PUSH_TISSUE_SCORE if composition.fraction_of(c) > 0.01]
    if not present:
        return None
    return max(_PUSH_TISSUE_SCORE[c] for c in present)
