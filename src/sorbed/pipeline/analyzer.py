"""The end-to-end wound analysis orchestrator.

Runs the full pipeline — load, calibrate, normalize, estimate skin tone, segment,
classify tissue, measure, stage, explain — and assembles a
:class:`~sorbed.domain.analysis.WoundAnalysis`. Every field it fills is computed
here from real pixels; nothing is fabricated. Intermediate rasters (the wound
mask and tissue label map) are returned alongside the analysis for visualization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from sorbed.config.settings import Settings, get_settings
from sorbed.domain.analysis import ModelProvenance, WoundAnalysis
from sorbed.domain.metrics import ColorCues, Metrics
from sorbed.explain.narrative import build_narrative
from sorbed.imaging import RasterImage
from sorbed.io.loader import load_image
from sorbed.morphometrics import (
    analyze_periwound,
    compute_healing_scores,
    measure_geometry,
    shading_depth_proxy,
)
from sorbed.pipeline.backends import (
    build_segmenter,
    build_staging_engine,
    build_tissue_classifier,
)
from sorbed.preprocess.calibration import resolve_calibration
from sorbed.preprocess.color_norm import gray_world_normalize
from sorbed.preprocess.skin_tone import estimate_skin_tone
from sorbed.staging.features import build_features


@dataclass(frozen=True)
class AnalyzeOptions:
    """Per-request options that override configuration."""

    mm_per_px: float | None = None
    marker_length_mm: float | None = None
    coin_diameter_mm: float | None = None
    head_vector: tuple[float, float] | None = None


@dataclass(frozen=True)
class AnalysisBundle:
    """The analysis plus the rasters needed to render it."""

    analysis: WoundAnalysis
    display_image: RasterImage
    wound_mask: np.ndarray = field(repr=False)
    tissue_label_map: np.ndarray = field(repr=False)


class WoundAnalyzer:
    """Owns the configured backends and runs the pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._segmenter = build_segmenter(self._settings)
        self._tissue = build_tissue_classifier(self._settings)
        self._staging = build_staging_engine(self._settings)

    def analyze(self, image: RasterImage, options: AnalyzeOptions | None = None) -> AnalysisBundle:
        options = options or AnalyzeOptions()
        timings: dict[str, float] = {}

        with _Timer(timings, "calibration"):
            calibration = resolve_calibration(
                image,
                detect_fiducials=self._settings.detect_fiducials,
                marker_length_mm=options.marker_length_mm,
                coin_diameter_mm=options.coin_diameter_mm,
            )
            image = image.with_calibration(calibration)

        display_image = image
        # Color normalization stabilizes tissue *color* classification, but a
        # learned segmentation model must see the raw image distribution it was
        # trained on. So segmentation runs on the original image; normalization
        # feeds only the tissue/color analysis.
        analysis_pixels = image.pixels
        if self._settings.apply_color_normalization:
            with _Timer(timings, "color_normalization"):
                valid = image.alpha > 0.5 if image.alpha is not None else None
                analysis_pixels = gray_world_normalize(image.pixels, valid)

        with _Timer(timings, "segmentation"):
            seg = self._segmenter.segment(image)

        with _Timer(timings, "skin_tone"):
            # Sample real skin from the periwound ring, now that the wound is known.
            skin_tone = estimate_skin_tone(analysis_pixels, seg.wound_mask)

        with _Timer(timings, "tissue"):
            tissue = self._tissue.classify(analysis_pixels, seg.wound_mask)

        with _Timer(timings, "morphometrics"):
            metrics = self._build_metrics(analysis_pixels, seg.wound_mask, tissue, calibration,
                                          options.head_vector)

        with _Timer(timings, "staging"):
            features = build_features(metrics, seg.confidence)
            decision = self._staging.decide(
                features, skin_tone=skin_tone, is_calibrated=calibration.is_calibrated
            )

        narrative = build_narrative(decision, metrics)
        decision = decision.model_copy(update={"narrative": narrative})

        analysis = WoundAnalysis(
            analysis_id=uuid4(),
            created_at=datetime.now(UTC),
            image=image.metadata,
            calibration=calibration,
            skin_tone_band=skin_tone,
            metrics=metrics,
            decision=decision,
            provenance=ModelProvenance(
                segmentation_backend=self._segmenter.name,
                tissue_backend=self._tissue.name,
                staging_backend=self._staging.name,
                weights_sha256=_weights(seg.weights_sha256, self._segmenter.name),
            ),
            stage_timings_ms={k: round(v, 2) for k, v in timings.items()},
            config_digest=self._settings.digest(),
        )
        return AnalysisBundle(
            analysis=analysis,
            display_image=display_image,
            wound_mask=seg.wound_mask,
            tissue_label_map=tissue.label_map,
        )

    def _depth_proxy(self, pixels: np.ndarray, wound_mask: np.ndarray):
        """Compute the depth cue with the configured backend.

        ``depth_backend="yolo"`` uses a YOLO26 monocular-depth model; any failure
        (missing ultralytics, download error) falls back to the weight-free shading
        proxy so a depth backend can never break an analysis.
        """
        if self._settings.depth_backend == "yolo":
            try:
                from sorbed.morphometrics.yolo_depth import yolo_depth_proxy

                proxy = yolo_depth_proxy(pixels, wound_mask)
                if proxy is not None:
                    return proxy
            except Exception:  # noqa: BLE001 — never let the depth cue break analysis
                pass
        return shading_depth_proxy(pixels, wound_mask)

    def _build_metrics(
        self,
        pixels: np.ndarray,
        wound_mask: np.ndarray,
        tissue: object,
        calibration: object,
        head_vector: tuple[float, float] | None,
    ) -> Metrics:
        geometry = measure_geometry(wound_mask, calibration, head_vector=head_vector)
        composition = tissue.composition  # type: ignore[attr-defined]
        depth = self._depth_proxy(pixels, wound_mask)
        periwound = analyze_periwound(pixels, wound_mask)
        healing = compute_healing_scores(composition, geometry.area_cm2)
        cues = ColorCues(
            maroon_purple_fraction=tissue.maroon_purple_fraction,  # type: ignore[attr-defined]
            erythema_fraction=tissue.erythema_fraction,  # type: ignore[attr-defined]
            open_bed_fraction=tissue.open_bed_fraction,  # type: ignore[attr-defined]
        )
        return Metrics(
            geometry=geometry,
            tissue=composition,
            color_cues=cues,
            depth_proxy=depth,
            periwound=periwound,
            healing_scores=healing,
            skin_intact=tissue.skin_appears_intact,  # type: ignore[attr-defined]
        )


def analyze_image(
    source: str | Path | bytes,
    *,
    settings: Settings | None = None,
    options: AnalyzeOptions | None = None,
) -> AnalysisBundle:
    """Load ``source`` and run the full analysis in one call."""
    options = options or AnalyzeOptions()
    image = load_image(source, mm_per_px=options.mm_per_px)
    return WoundAnalyzer(settings).analyze(image, options)


class _Timer:
    """Context manager recording elapsed milliseconds into a dict."""

    def __init__(self, sink: dict[str, float], key: str) -> None:
        self._sink = sink
        self._key = key
        self._start = 0.0

    def __enter__(self) -> None:
        self._start = time.perf_counter()

    def __exit__(self, *exc: object) -> None:
        self._sink[self._key] = (time.perf_counter() - self._start) * 1000.0


def _weights(sha: str | None, backend: str) -> dict[str, str]:
    return {backend: sha} if sha else {}
