#!/usr/bin/env python3
"""Generate the directive-grounded PDF reports for one wound image.

Runs the Sorbed pipeline on an image, grounds the result in a clinical directive
pack (for example the HD_T86 / Acibadem pressure-injury directive built with
``scripts/build_directive_pack.py``), and renders the summary and grading PDF
reports. With ``--seg-onnx`` the learned ONNX segmenter is used instead of the
classical backend.

Example
-------
    python scripts/generate_reports.py \\
        --image wound.png --mm-per-px 0.08 --patient-ref "PID-2026-0431 · Sakral" \\
        --directive-pack var/directive_packs/hd_t86 \\
        --seg-onnx models/release/seg_unetpp.onnx --out-dir reports
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image", type=Path, required=True, help="Wound photograph.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--mm-per-px", type=float, default=None,
                        help="Known scale in millimetres per pixel (else pixel units).")
    parser.add_argument("--patient-ref", type=str, default="")
    parser.add_argument("--directive-pack", type=Path, default=None,
                        help="Directive pack directory (else $SORBED_DIRECTIVE_PACK).")
    parser.add_argument("--seg-onnx", type=Path, default=None,
                        help="Learned ONNX segmenter; classical backend if omitted.")
    parser.add_argument("--which", type=str, default="summary,grading",
                        help="Comma list of reports to render: summary, grading.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.directive_pack is not None:
        os.environ["SORBED_DIRECTIVE_PACK"] = str(Path(args.directive_pack).resolve())
    if args.seg_onnx is not None:
        os.environ["SORBED_SEGMENTATION_BACKEND"] = "onnx"
        os.environ["SORBED_ONNX_MODEL"] = str(Path(args.seg_onnx).resolve())

    from sorbed.guidelines import build_guideline_context, load_default_pack
    from sorbed.morphometrics import relative_depth_field
    from sorbed.pipeline import AnalyzeOptions, analyze_image
    from sorbed.report.assets import file_image_data_uri, png_data_uri
    from sorbed.report.pdf import html_to_pdf
    from sorbed.report.templates import build_grading_report_html, build_summary_report_html
    from sorbed.visualize.dashboard import render_depth_overlay
    from sorbed.visualize.detection import render_detection
    from sorbed.visualize.overlay import render_mask, render_schematic, render_tissue_overlay

    which = {w.strip() for w in args.which.split(",") if w.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    pack = load_default_pack()
    bundle = analyze_image(str(args.image), options=AnalyzeOptions(mm_per_px=args.mm_per_px))
    analysis = bundle.analysis
    ctx = build_guideline_context(analysis, pack)
    disp = bundle.display_image.to_uint8_rgb()

    images = {
        "photo": file_image_data_uri(args.image, max_side=760),
        "overlay": png_data_uri(
            render_tissue_overlay(disp, bundle.tissue_label_map, bundle.wound_mask), max_side=760),
        "detection": png_data_uri(render_detection(analysis, disp), max_side=760),
        "depth": png_data_uri(
            render_depth_overlay(
                disp, relative_depth_field(bundle.display_image.pixels, bundle.wound_mask),
                bundle.wound_mask), max_side=760),
        "mask": png_data_uri(render_mask(bundle.wound_mask), max_side=520),
        "schematic": png_data_uri(
            render_schematic(bundle.tissue_label_map, bundle.wound_mask), max_side=520),
    }
    if ctx.stage and ctx.stage.figure:
        images["directive_stage"] = file_image_data_uri(ctx.stage.figure, max_side=520)

    written: list[Path] = []
    if "summary" in which:
        html = build_summary_report_html(
            analysis=analysis, guideline_ctx=ctx,
            images={"photo": images["photo"], "overlay": images["overlay"]},
            patient_ref=args.patient_ref, generated=now)
        dest = args.out_dir / "sorbed_summary_report.pdf"
        html_to_pdf(html, dest)
        written.append(dest)
    if "grading" in which:
        html = build_grading_report_html(
            analysis=analysis, guideline_ctx=ctx, images=images,
            patient_ref=args.patient_ref, generated=now)
        dest = args.out_dir / "sorbed_grading_report.pdf"
        html_to_pdf(html, dest)
        written.append(dest)

    print(f"stage={analysis.decision.stage} confidence={analysis.decision.confidence:.2f} "
          f"directive={'grounded' if ctx.available else 'none'}")
    for d in written:
        print(f"wrote {d} ({d.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
