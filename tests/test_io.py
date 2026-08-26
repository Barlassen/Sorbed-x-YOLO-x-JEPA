"""Image ingestion tests: format detection, round-trip, DICOM, de-identification."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from sorbed.domain.enums import CalibrationStatus
from sorbed.io import ImageFormat, load_image, sniff_format
from tests.synth import make_wound


def test_sniff_png_jpeg():
    wound = make_wound()
    assert sniff_format(wound.to_png_bytes()) is ImageFormat.PNG
    assert sniff_format(wound.to_jpeg_bytes()) is ImageFormat.JPEG


def test_png_round_trip_shape_and_dtype():
    wound = make_wound(h=120, w=160)
    img = load_image(wound.to_png_bytes())
    assert img.pixels.shape == (120, 160, 3)
    assert img.pixels.dtype == np.float32
    assert float(img.pixels.min()) >= 0.0 and float(img.pixels.max()) <= 1.0


def test_manual_calibration_overrides():
    img = load_image(make_wound().to_png_bytes(), mm_per_px=0.25)
    assert img.calibration.status is CalibrationStatus.MANUAL
    assert img.calibration.to_mm2(4) == pytest.approx(0.25)


def test_uncalibrated_by_default():
    img = load_image(make_wound().to_png_bytes())
    assert img.calibration.status is CalibrationStatus.UNCALIBRATED
    assert img.calibration.mm_per_px is None


def test_content_sniff_ignores_extension():
    # PNG bytes with a lying ".jpg" name are still detected as PNG.
    data = make_wound().to_png_bytes()
    assert sniff_format(data, filename="wound.jpg") is ImageFormat.PNG


@pytest.mark.requires_formats
def test_dicom_calibration_and_phi_stripping():
    pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    arr = np.zeros((64, 96, 3), dtype=np.uint8)
    arr[10:50, 20:70] = (180, 60, 60)
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.PatientName = "SECRET^PATIENT"
    ds.PatientID = "MRN-123"
    ds.Modality = "XC"
    ds.BodyPartExamined = "SACRUM"
    ds.Rows, ds.Columns = 64, 96
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.5, 0.5]
    ds.PixelData = arr.tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)

    img = load_image(buf.getvalue())
    assert img.calibration.status is CalibrationStatus.DICOM_SPACING
    assert img.calibration.mm_per_px == pytest.approx(0.5)
    # No patient identifier survives into the retained tags.
    joined = " ".join(img.metadata.retained_tags.values())
    assert "SECRET" not in joined and "MRN" not in joined
    assert img.metadata.retained_tags.get("BodyPartExamined") == "SACRUM"


def test_unknown_format_rejected():
    from sorbed.io import UnsupportedFormatError

    with pytest.raises(UnsupportedFormatError):
        load_image(b"not an image at all, just text bytes here.......")


def _png(rgb: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray((rgb * 255).astype(np.uint8)).save(b, "PNG")
    return b.getvalue()
