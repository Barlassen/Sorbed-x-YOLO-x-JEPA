"""Analysis + longitudinal comparison endpoints.

Patient images are processed entirely in memory. Uploaded bytes are decoded,
analyzed, and discarded when the request ends; nothing is written to disk here.
"""

from __future__ import annotations

import base64
import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from sorbed.api.deps import ApiSettings, get_analyzer, get_api_settings
from sorbed.api.rendering import detection_thumb, rendered_panels
from sorbed.api.schemas import AnalyzeResponse, TrendResponse
from sorbed.io import ImageFormat, load_image, sniff_format
from sorbed.pipeline import AnalyzeOptions
from sorbed.pipeline.analyzer import AnalysisBundle
from sorbed.trend import compute_trend, timepoint_from_analysis
from sorbed.visualize.guide import render_guide

router = APIRouter(tags=["analyze"])

_TRUTHY = {"1", "true", "yes", "on"}


def _flag(request: Request, name: str, form_value: bool) -> bool:
    raw = request.query_params.get(name)
    if raw is not None:
        return raw.strip().lower() in _TRUTHY
    return form_value


def _validate_size(request: Request, data: bytes, max_bytes: int) -> None:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > max_bytes:
        raise HTTPException(status_code=413, detail=f"upload exceeds the {max_bytes} byte limit")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"upload exceeds the {max_bytes} byte limit")
    if not data:
        raise HTTPException(status_code=422, detail="uploaded file is empty")


def _analyze_bytes(
    request: Request, data: bytes, filename: str | None, options: AnalyzeOptions
) -> AnalysisBundle:
    fmt = sniff_format(data, filename=filename)
    if fmt is ImageFormat.UNKNOWN:
        raise HTTPException(
            status_code=415, detail="could not detect a supported image format from the upload"
        )
    try:
        image = load_image(data, mm_per_px=options.mm_per_px)
        return get_analyzer(request).analyze(image, options)
    except HTTPException:
        raise
    except Exception as exc:  # surface any pipeline failure to the caller
        raise HTTPException(status_code=422, detail=f"analysis failed: {exc}") from exc


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(..., description="The wound photograph to analyze."),  # noqa: B008
    mm_per_px: float | None = Form(default=None),
    marker_mm: float | None = Form(default=None),
    coin_mm: float | None = Form(default=None),
    include_images: bool = Form(default=True),
    include_guide: bool = Form(default=False),
    api_settings: ApiSettings = Depends(get_api_settings),  # noqa: B008
) -> AnalyzeResponse:
    """Analyze one uploaded image; return the WoundAnalysis and rendered panels."""
    data = await file.read()
    _validate_size(request, data, api_settings.max_upload_bytes)
    options = AnalyzeOptions(
        mm_per_px=mm_per_px, marker_length_mm=marker_mm, coin_diameter_mm=coin_mm
    )
    bundle = _analyze_bytes(request, data, file.filename, options)

    images = rendered_panels(bundle) if _flag(request, "include_images", include_images) else {}
    guide_b64: str | None = None
    if _flag(request, "include_guide", include_guide):
        guide = render_guide(
            bundle.analysis,
            bundle.display_image.to_uint8_rgb(),
            bundle.wound_mask,
            bundle.tissue_label_map,
        )
        buf = io.BytesIO()
        guide.save(buf, format="PNG")
        guide_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return AnalyzeResponse(
        analysis=bundle.analysis.model_dump(mode="json"), images=images, guide_png_base64=guide_b64
    )


@router.post("/compare", response_model=TrendResponse)
async def compare(
    request: Request,
    files: list[UploadFile] = File(..., description="Two+ images of the SAME wound, in order."),  # noqa: B008
    days: str | None = Form(default=None, description="Comma-separated days, e.g. 0,14,28."),
    mm_per_px: float | None = Form(default=None),
    patient: str | None = Form(default=None),
    api_settings: ApiSettings = Depends(get_api_settings),  # noqa: B008
) -> TrendResponse:
    """Compare visits of the same wound over time and return the healing trend."""
    if len(files) < 2:
        raise HTTPException(status_code=422, detail="provide at least two images (visits)")
    day_values = _parse_days(days, len(files))
    if day_values is None:
        raise HTTPException(
            status_code=422, detail="--days must be a comma list matching the number of images"
        )

    options = AnalyzeOptions(mm_per_px=mm_per_px)
    points, thumbs = [], []
    for upload, day in zip(files, day_values, strict=True):
        data = await upload.read()
        _validate_size(request, data, api_settings.max_upload_bytes)
        bundle = _analyze_bytes(request, data, upload.filename, options)
        points.append(timepoint_from_analysis(bundle.analysis, day=day, label=f"Day {day:g}"))
        thumbs.append(detection_thumb(bundle))

    trend = compute_trend(points, patient_ref=patient)
    return TrendResponse(trend=trend.model_dump(mode="json"), thumbnails=thumbs)


def _parse_days(days: str | None, count: int) -> list[float] | None:
    if days is None or not days.strip():
        return [float(i) for i in range(count)]
    try:
        parsed = [float(p.strip()) for p in days.split(",") if p.strip()]
    except ValueError:
        return None
    return parsed if len(parsed) == count else None
