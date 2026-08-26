"""Color-based wound-bed tissue classification.

Implements the clinical Red-Yellow-Black proxy extended with epithelial, intact-
skin, and a deep-red/maroon/purple (deep-tissue) cue, using CIE-Lab and HSV
thresholds drawn from the tissue color signatures in ``docs/CLINICAL.md``:

* granulation — beefy/bright red (elevated a*)
* slough      — yellow/tan/green (elevated b*, lower a*)
* eschar      — black/brown (low L*)
* epithelial  — pale pink/pearly (high L*, low chroma)
* intact skin — skin-toned, un-wound-like
* maroon/purple — deep-red/maroon/purple, the deep-tissue-injury cue

Color thresholds are brittle to lighting by nature (hence the color-normalization
step upstream); this classifier reports per-tissue **percent area**, not hard
per-pixel truth, and pairs with a learned backend for accuracy. Classification is
fully vectorized so it stays fast on multi-megapixel phone photos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage import color

from sorbed.domain.enums import TissueClass
from sorbed.domain.metrics import TissueComposition

# Deterministic ordering for label maps and palettes.
TISSUE_ORDER: tuple[TissueClass, ...] = (
    TissueClass.BACKGROUND,
    TissueClass.EPITHELIAL,
    TissueClass.GRANULATION,
    TissueClass.SLOUGH,
    TissueClass.ESCHAR,
    TissueClass.INTACT_SKIN,
    TissueClass.UNKNOWN,
)
_INDEX = {cls: i for i, cls in enumerate(TISSUE_ORDER)}


@dataclass(frozen=True)
class TissueAnalysis:
    """Per-pixel labels, composition, and derived color cues for staging."""

    label_map: np.ndarray  # int (H, W); index into TISSUE_ORDER
    composition: TissueComposition
    maroon_purple_fraction: float  # deep-tissue-injury cue over the region
    erythema_fraction: float  # non-blanchable-erythema cue (intact skin)
    open_bed_fraction: float  # granulation + slough + eschar
    per_pixel_confidence: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def skin_appears_intact(self) -> bool:
        """The region reads as discolored intact skin, not an open wound bed."""
        return self.open_bed_fraction < 0.15


class ColorTissueClassifier:
    """Classifies wound-bed pixels by their Lab/HSV color signature."""

    name = "color_model"

    def classify(self, rgb: np.ndarray, wound_mask: np.ndarray) -> TissueAnalysis:
        h, w = wound_mask.shape
        label_map = np.full((h, w), _INDEX[TissueClass.BACKGROUND], dtype=np.int16)
        confidence = np.zeros((h, w), dtype=np.float32)

        if not wound_mask.any():
            return TissueAnalysis(
                label_map=label_map,
                composition=self._empty_composition(),
                maroon_purple_fraction=0.0,
                erythema_fraction=0.0,
                open_bed_fraction=0.0,
                per_pixel_confidence=confidence,
            )

        lab = color.rgb2lab(rgb)
        hsv = color.rgb2hsv(rgb)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        hue, val = hsv[..., 0] * 360.0, hsv[..., 2]
        chroma = np.sqrt(a**2 + b**2)

        m = wound_mask
        unassigned = m.copy()

        def take(condition: np.ndarray) -> np.ndarray:
            sel = condition & unassigned
            unassigned[sel] = False
            return sel

        # Priority order: darkness and deep-tissue color before the red/yellow split.
        # Eschar is dark AND desaturated (black/brown necrosis); a dark but vivid
        # red is granulation, not eschar, so low chroma is required.
        eschar = take(((L < 35) & (chroma < 25)) | (val < 0.20))
        maroon = take((a > 12) & (L < 55) & (b < 14) & ((hue < 20) | (hue > 300)))
        slough = take((b > 18) & (a < 22) & (L > 40))
        granulation = take((a > 18) & (chroma > 20))
        epithelial = take((L > 65) & (chroma < 20) & (a > 2))
        intact = take((L > 45) & (L < 85) & (a > 5) & (a < 25) & (b > 8) & (b < 35))
        unknown = unassigned & m

        # maroon/purple is reported as intact-skin discoloration and flagged for DTI.
        intact = intact | maroon

        assignments = {
            TissueClass.ESCHAR: eschar,
            TissueClass.SLOUGH: slough,
            TissueClass.GRANULATION: granulation,
            TissueClass.EPITHELIAL: epithelial,
            TissueClass.INTACT_SKIN: intact,
            TissueClass.UNKNOWN: unknown,
        }
        conf_maps = {
            TissueClass.ESCHAR: np.clip((32 - L) / 32 + 0.4, 0, 1),
            TissueClass.SLOUGH: np.clip(b / 60 + 0.3, 0, 1),
            TissueClass.GRANULATION: np.clip(a / 50 + 0.3, 0, 1),
            TissueClass.EPITHELIAL: np.full_like(L, 0.5),
            TissueClass.INTACT_SKIN: np.full_like(L, 0.45),
            TissueClass.UNKNOWN: np.full_like(L, 0.2),
        }
        for cls, sel in assignments.items():
            label_map[sel] = _INDEX[cls]
            confidence[sel] = conf_maps[cls][sel].astype(np.float32)

        total = int(m.sum())
        counts = {cls: int(sel.sum()) for cls, sel in assignments.items()}
        conf_sum = {cls: float(confidence[sel].sum()) for cls, sel in assignments.items()}
        composition = self._build_composition(counts, conf_sum, total)

        erythema = int((granulation & (L > 55) & (a < 35)).sum())
        open_bed = (counts[TissueClass.GRANULATION] + counts[TissueClass.SLOUGH]
                    + counts[TissueClass.ESCHAR]) / total

        return TissueAnalysis(
            label_map=label_map,
            composition=composition,
            maroon_purple_fraction=round(int(maroon.sum()) / total, 4),
            erythema_fraction=round(erythema / total, 4),
            open_bed_fraction=round(open_bed, 4),
            per_pixel_confidence=confidence,
        )

    def _build_composition(
        self,
        counts: dict[TissueClass, int],
        conf_sum: dict[TissueClass, float],
        total: int,
    ) -> TissueComposition:
        present = {c: n for c, n in counts.items() if n > 0}
        if not present or total == 0:
            return self._empty_composition()
        raw = {c: n / total for c, n in present.items()}
        norm = sum(raw.values())
        fractions = {c: round(f / norm, 6) for c, f in raw.items()}
        # Correct any rounding drift onto the dominant class so the sum is exactly 1.
        dominant = max(fractions.items(), key=lambda kv: kv[1])[0]
        fractions[dominant] = round(fractions[dominant] + (1.0 - sum(fractions.values())), 6)
        areas_px = {c: float(counts[c]) for c in present}
        mean_conf = {c: round(conf_sum[c] / counts[c], 4) for c in present}
        return TissueComposition(
            fractions=fractions,
            areas_px=areas_px,
            dominant=dominant,
            mean_confidence=mean_conf,
        )

    @staticmethod
    def _empty_composition() -> TissueComposition:
        return TissueComposition(
            fractions={TissueClass.UNKNOWN: 1.0},
            areas_px={TissueClass.UNKNOWN: 0.0},
            dominant=TissueClass.UNKNOWN,
            mean_confidence={TissueClass.UNKNOWN: 0.0},
        )


def tissue_index(cls: TissueClass) -> int:
    """Return the label-map index for a tissue class."""
    return _INDEX[cls]
