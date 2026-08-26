"""DICOM decoder via pydicom.

Handles both photographic (RGB / YBR) and monochrome DICOM. Applies the Modality
LUT (rescale slope/intercept) and VOI LUT/windowing for correct display mapping,
reads pixel spacing for real-world calibration, and de-identifies: only a
whitelist of non-identifying tags is retained; patient identifiers never leave
the decoder.
"""

from __future__ import annotations

import contextlib

import numpy as np

from sorbed.domain.enums import CalibrationStatus
from sorbed.io.decoders.base import DecodedImage, DecodeError, Decoder, DependencyMissingError
from sorbed.io.normalize import ensure_canonical, to_float01
from sorbed.io.sniff import ImageFormat

# Tags safe to keep in a report — none can identify a patient.
_RETAINED_TAGS = (
    "Modality",
    "BodyPartExamined",
    "PhotometricInterpretation",
    "Manufacturer",
    "AnatomicRegionSequence",
)


class DicomDecoder(Decoder):
    formats = (ImageFormat.DICOM,)

    @property
    def available(self) -> bool:
        try:
            import pydicom  # noqa: F401
        except ImportError:
            return False
        return True

    def decode(self, data: bytes, *, filename: str | None = None) -> DecodedImage:
        try:
            import pydicom

            try:  # pydicom >= 3.0
                from pydicom.pixels import apply_modality_lut, apply_voi_lut
            except ImportError:  # pydicom < 3.0
                from pydicom.pixel_data_handlers.util import (
                    apply_modality_lut,
                    apply_voi_lut,
                )
        except ImportError as exc:
            raise DependencyMissingError(
                "DICOM support needs the 'formats' extra: pip install 'sorbed[formats]'"
            ) from exc

        from io import BytesIO

        try:
            ds = pydicom.dcmread(BytesIO(data), force=True)
            raw = ds.pixel_array
        except Exception as exc:
            raise DecodeError(f"pydicom could not read DICOM pixels: {exc}") from exc

        photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        samples = int(getattr(ds, "SamplesPerPixel", 1))

        if samples >= 3:
            rgb = self._to_rgb(ds, raw, photometric)
            bit_depth = 8
            rgb_f = to_float01(rgb)
        else:
            mono = apply_modality_lut(raw, ds)
            with contextlib.suppress(Exception):
                mono = apply_voi_lut(mono, ds)
            mono = mono.astype(np.float32)
            lo, hi = float(np.min(mono)), float(np.max(mono))
            norm = (mono - lo) / (hi - lo) if hi > lo else np.zeros_like(mono)
            if photometric == "MONOCHROME1":  # inverted grayscale
                norm = 1.0 - norm
            rgb_f = np.repeat(norm[:, :, None], 3, axis=2)
            bit_depth = 16 if raw.dtype.itemsize >= 2 else 8

        rgb = ensure_canonical(rgb_f)

        mm_per_px, status = self._pixel_spacing(ds)
        retained = {
            tag: str(getattr(ds, tag))
            for tag in _RETAINED_TAGS
            if getattr(ds, tag, None) not in (None, "")
        }

        return DecodedImage(
            rgb=rgb,
            alpha=None,
            bit_depth=bit_depth,
            channels=3 if samples >= 3 else 1,
            color_space="sRGB",
            icc_converted=False,
            exif_oriented=False,
            retained_tags=retained,
            mm_per_px=mm_per_px,
            calibration_status=status,
        )

    @staticmethod
    def _to_rgb(ds: object, raw: np.ndarray, photometric: str) -> np.ndarray:
        if photometric.startswith("YBR"):
            try:  # pydicom >= 3.0
                from pydicom.pixels import convert_color_space
            except ImportError:  # pydicom < 3.0
                from pydicom.pixel_data_handlers.util import convert_color_space

            return convert_color_space(raw, photometric, "RGB")
        return raw

    @staticmethod
    def _pixel_spacing(ds: object) -> tuple[float | None, CalibrationStatus | None]:
        for attr in ("ImagerPixelSpacing", "PixelSpacing", "NominalScannedPixelSpacing"):
            spacing = getattr(ds, attr, None)
            if spacing is not None and len(spacing) >= 2:
                row_mm, col_mm = float(spacing[0]), float(spacing[1])
                mean_mm = (row_mm + col_mm) / 2.0
                if mean_mm > 0:
                    return mean_mm, CalibrationStatus.DICOM_SPACING
        return None, None
