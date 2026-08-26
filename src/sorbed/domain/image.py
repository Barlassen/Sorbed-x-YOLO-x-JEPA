"""Serializable metadata and calibration contracts for an analyzed image.

Pixel buffers are carried out-of-band (see :class:`sorbed.imaging.RasterImage`);
these models hold only what belongs in a report.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sorbed.domain.enums import CalibrationStatus


class ImageMetadata(BaseModel):
    """Provenance and technical description of a decoded image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_format: str = Field(description="Detected container format, e.g. 'jpeg', 'dicom'.")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    channels: int = Field(ge=1, le=4)
    bit_depth: int = Field(ge=8, le=16)
    color_space: str = Field(default="sRGB")
    icc_converted: bool = False
    exif_oriented: bool = False
    # A whitelist of non-identifying DICOM/EXIF tags. Patient identifiers are
    # never propagated here (see the de-identification step in the decoders).
    retained_tags: dict[str, str] = Field(default_factory=dict)
    pixel_sha256: str = Field(description="SHA-256 of the normalized pixel buffer.")

    @property
    def total_pixels(self) -> int:
        return self.width_px * self.height_px


class Calibration(BaseModel):
    """Real-world scale, or an explicit record that none was available.

    ``mm_per_px`` is ``None`` exactly when ``status`` is ``UNCALIBRATED``. The
    pipeline never substitutes a guessed scale; downstream physical measurements
    become ``None`` instead.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mm_per_px: float | None = Field(default=None, gt=0)
    status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    uncertainty_pct: float | None = Field(default=None, ge=0)
    source_detail: str | None = None

    @model_validator(mode="after")
    def _consistency(self) -> Calibration:
        calibrated = self.status is not CalibrationStatus.UNCALIBRATED
        if calibrated and self.mm_per_px is None:
            raise ValueError(f"status {self.status} requires mm_per_px to be set")
        if not calibrated and self.mm_per_px is not None:
            raise ValueError("uncalibrated status must not carry an mm_per_px value")
        return self

    @property
    def is_calibrated(self) -> bool:
        return self.mm_per_px is not None

    def to_mm(self, pixels: float) -> float | None:
        """Convert a pixel length to millimeters, or ``None`` if uncalibrated."""
        if self.mm_per_px is None:
            return None
        return pixels * self.mm_per_px

    def to_mm2(self, pixel_area: float) -> float | None:
        """Convert a pixel area to square millimeters, or ``None`` if uncalibrated."""
        if self.mm_per_px is None:
            return None
        return pixel_area * (self.mm_per_px**2)
