"""A single, colorblind-aware color source for tissue classes.

Colors are chosen to evoke the clinical tissue color while staying distinguishable
for common color-vision deficiencies. This is the only place tissue colors are
defined, so overlays, guides, and legends always agree.
"""

from __future__ import annotations

from sorbed.domain.enums import PressureInjuryStage, TissueClass

# RGB 0-255. Hues nod to the clinical signature (red granulation, yellow slough,
# dark eschar) while keeping luminance separated for accessibility.
TISSUE_COLORS: dict[TissueClass, tuple[int, int, int]] = {
    TissueClass.EPITHELIAL: (245, 176, 200),  # pale pink
    TissueClass.GRANULATION: (214, 40, 57),  # beefy red
    TissueClass.SLOUGH: (240, 200, 70),  # yellow
    TissueClass.ESCHAR: (40, 40, 45),  # near-black
    TissueClass.ADIPOSE: (250, 232, 150),  # pale yellow
    TissueClass.MUSCLE: (140, 30, 60),  # dark red
    TissueClass.TENDON_BONE: (238, 238, 220),  # off-white
    TissueClass.INTACT_SKIN: (150, 190, 235),  # blue (non-clinical, for contrast)
    TissueClass.BACKGROUND: (0, 0, 0),
    TissueClass.UNKNOWN: (130, 130, 130),  # gray
}

CONTOUR_COLOR = (0, 229, 255)  # cyan wound outline
LENGTH_COLOR = (0, 229, 255)
WIDTH_COLOR = (255, 145, 0)
SCALEBAR_COLOR = (255, 255, 255)

# Per-stage box colors for the detection-style output. Distinct hues, readable
# on wound photos, and consistent between the box and its label tab.

STAGE_COLORS: dict[PressureInjuryStage, tuple[int, int, int]] = {
    PressureInjuryStage.STAGE_1: (255, 214, 0),  # amber
    PressureInjuryStage.STAGE_2: (255, 138, 101),  # salmon
    PressureInjuryStage.STAGE_3: (255, 111, 0),  # orange
    PressureInjuryStage.STAGE_4: (233, 30, 99),  # magenta-red
    PressureInjuryStage.UNSTAGEABLE: (0, 200, 83),  # green
    PressureInjuryStage.DEEP_TISSUE: (213, 0, 0),  # deep red
    PressureInjuryStage.MUCOSAL: (124, 77, 255),  # violet
    PressureInjuryStage.NOT_PRESSURE_INJURY: (120, 144, 156),  # blue-gray
    PressureInjuryStage.INDETERMINATE: (158, 158, 158),  # gray
}


def tissue_color(cls: TissueClass) -> tuple[int, int, int]:
    return TISSUE_COLORS.get(cls, TISSUE_COLORS[TissueClass.UNKNOWN])


def stage_color(stage: PressureInjuryStage) -> tuple[int, int, int]:
    return STAGE_COLORS.get(stage, (158, 158, 158))
