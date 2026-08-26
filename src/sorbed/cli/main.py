"""The ``sorbed`` command-line interface.

Commands share the exact pipeline the API uses; no analysis logic lives here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from sorbed.cli.feedback_cmd import feedback_app
from sorbed.config.settings import get_settings
from sorbed.domain.analysis import WoundAnalysis
from sorbed.io import DecoderRegistry, load_image, sniff_format
from sorbed.pipeline import AnalyzeOptions, analyze_image
from sorbed.report.builder import DEFAULT_ARTIFACTS, write_report
from sorbed.version import __version__

app = typer.Typer(
    add_completion=False,
    help="Explainable pressure-injury (bedsore) image analysis. Decision support, "
    "not a diagnosis — every result must be reviewed by a clinician.",
    no_args_is_help=True,
)
app.add_typer(feedback_app, name="feedback")
console = Console()
err_console = Console(stderr=True)


@app.command()
def analyze(
    image: Annotated[Path, typer.Argument(help="Path to the wound image (any supported format).")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path("reports"),
    mm_per_px: Annotated[
        float | None, typer.Option("--mm-per-px", help="Known scale in millimeters per pixel.")
    ] = None,
    marker_mm: Annotated[
        float | None, typer.Option("--marker-mm", help="ArUco marker side length in mm.")
    ] = None,
    coin_mm: Annotated[
        float | None, typer.Option("--coin-mm", help="Reference coin diameter in mm.")
    ] = None,
    fmt: Annotated[
        str, typer.Option("--format", "-f", help="Comma list of artifacts, or 'all'.")
    ] = "all",
    json_out: Annotated[
        bool, typer.Option("--json", help="Print the analysis JSON to stdout.")
    ] = False,
) -> None:
    """Analyze one wound image and write a report."""
    if not image.exists():
        err_console.print(f"[red]No such file:[/red] {image}")
        raise typer.Exit(2)

    artifacts = DEFAULT_ARTIFACTS if fmt.strip() == "all" else tuple(
        p.strip() for p in fmt.split(",") if p.strip()
    )
    options = AnalyzeOptions(
        mm_per_px=mm_per_px, marker_length_mm=marker_mm, coin_diameter_mm=coin_mm
    )

    try:
        bundle = analyze_image(image, options=options)
    except Exception as exc:  # surface a clean message, not a traceback
        err_console.print(f"[red]Analysis failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    report = write_report(bundle, out, artifacts=artifacts)
    if json_out:
        console.print_json(bundle.analysis.model_dump_json())
    else:
        _print_summary(bundle.analysis)
        console.print(f"\n[green]Wrote {len(report.artifacts)} artifact(s) to[/green] {out}/")


@app.command()
def compare(
    images: Annotated[list[Path], typer.Argument(help="Two+ images of the same wound, in order.")],
    days: Annotated[
        str | None, typer.Option("--days", help="Comma days since baseline, e.g. 0,14,28.")
    ] = None,
    mm_per_px: Annotated[
        float | None, typer.Option("--mm-per-px", help="Scale in mm per pixel (all visits).")
    ] = None,
    patient: Annotated[
        str | None, typer.Option("--patient", help="Patient/wound reference label.")
    ] = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path("reports"),
) -> None:
    """Compare visits of the same wound over time and chart the healing trend."""
    from sorbed.trend import compute_trend, timepoint_from_analysis
    from sorbed.visualize.detection import render_detection
    from sorbed.visualize.trend import render_trend_dashboard

    if len(images) < 2:
        err_console.print("[red]Provide at least two images (visits) to compare.[/red]")
        raise typer.Exit(2)
    for img in images:
        if not img.exists():
            err_console.print(f"[red]No such file:[/red] {img}")
            raise typer.Exit(2)

    day_values = _parse_days(days, len(images))
    if day_values is None:
        err_console.print("[red]--days must be a comma list matching the number of images.[/red]")
        raise typer.Exit(2)

    options = AnalyzeOptions(mm_per_px=mm_per_px)
    points, thumbs = [], []
    for img, day in zip(images, day_values, strict=True):
        bundle = analyze_image(img, options=options)
        label = f"Day {day:g}"
        points.append(timepoint_from_analysis(bundle.analysis, day=day, label=label))
        thumbs.append(render_detection(bundle.analysis, bundle.display_image.to_uint8_rgb()))

    trend = compute_trend(points, patient_ref=patient)
    out.mkdir(parents=True, exist_ok=True)
    dash = render_trend_dashboard(trend, thumbs)
    dash.save(out / "trend_dashboard.png")
    (out / "trend.json").write_text(trend.model_dump_json(indent=2), encoding="utf-8")

    console.print(f"\n[bold]Healing trend[/bold] — {trend.trajectory.upper()}")
    console.print(f"Area reduction: {trend.percent_area_reduction:+.0f}% over "
                  f"{day_values[-1] - day_values[0]:g} days")
    if trend.healing_rate_per_week is not None:
        console.print(f"Healing rate: {trend.healing_rate_per_week:+.2f} {trend.unit}/week")
    if trend.projected_days_to_closure is not None:
        console.print(f"Projected closure: ~{trend.projected_days_to_closure:.0f} days")
    if trend.likely_to_heal is not None:
        console.print(f"4-week PAR: {trend.par_at_4_weeks:.0f}%  "
                      f"({'on track to heal' if trend.likely_to_heal else 'below 40% threshold'})")
    console.print(f"\n[green]Wrote trend dashboard + JSON to[/green] {out}/")


def _parse_days(days: str | None, count: int) -> list[float] | None:
    if days is None:
        return [float(i) for i in range(count)]
    try:
        parsed = [float(p.strip()) for p in days.split(",") if p.strip()]
    except ValueError:
        return None
    return parsed if len(parsed) == count else None


@app.command()
def inspect(
    image: Annotated[Path, typer.Argument(help="Image to inspect (no grading).")],
) -> None:
    """Decode an image and report its metadata and detected scale, without grading."""
    if not image.exists():
        err_console.print(f"[red]No such file:[/red] {image}")
        raise typer.Exit(2)
    data = image.read_bytes()
    fmt = sniff_format(data, filename=image.name)
    loaded = load_image(data)
    table = Table(title=f"{image.name}", show_header=False)
    table.add_row("Detected format", fmt.value)
    table.add_row("Dimensions", f"{loaded.metadata.width_px} × {loaded.metadata.height_px} px")
    table.add_row("Bit depth", str(loaded.metadata.bit_depth))
    table.add_row("Calibration", loaded.calibration.status.value)
    if loaded.calibration.mm_per_px:
        table.add_row("Scale", f"{loaded.calibration.mm_per_px:.4f} mm/px")
    if loaded.metadata.retained_tags:
        table.add_row("Retained tags", json.dumps(loaded.metadata.retained_tags))
    console.print(table)


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Port.")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code changes.")] = False,
) -> None:
    """Launch the web UI and HTTP API (needs the 'api' extra)."""
    try:
        import uvicorn
    except ImportError as exc:
        err_console.print("[red]The API needs the 'api' extra:[/red] pip install 'sorbed[api]'")
        raise typer.Exit(1) from exc
    console.print(f"[green]Sorbed[/green] web UI + API on http://{host}:{port}  (Ctrl-C to stop)")
    uvicorn.run("sorbed.api.app:app", host=host, port=port, reload=reload)


@app.command()
def formats() -> None:
    """List image formats and whether a decoder for each is available."""
    table = Table("Format", "Available")
    for fmt, ok in DecoderRegistry().supported_formats().items():
        table.add_row(fmt, "[green]yes[/green]" if ok else "[yellow]needs extras[/yellow]")
    console.print(table)


@app.command()
def schema(
    out: Annotated[Path | None, typer.Option("--out", help="Write schema here instead of stdout.")]
    = None,
) -> None:
    """Print the JSON Schema of the analysis result contract."""
    doc = json.dumps(WoundAnalysis.model_json_schema(), indent=2)
    if out is not None:
        out.write_text(doc, encoding="utf-8")
        console.print(f"[green]Wrote schema to[/green] {out}")
    else:
        console.print_json(doc)


@app.command()
def config() -> None:
    """Show the effective configuration and its digest."""
    settings = get_settings()
    console.print_json(settings.model_dump_json())
    console.print(f"config digest: {settings.digest()[:16]}")


@app.command()
def version() -> None:
    """Print the Sorbed version."""
    console.print(__version__)


def _print_summary(analysis: WoundAnalysis) -> None:
    d = analysis.decision
    g = analysis.metrics.geometry
    grade = d.stage.value.replace("_", " ").title()
    conf = "withheld" if d.abstained else f"{d.confidence * 100:.0f}%"
    console.print(f"\n[bold]Provisional grade:[/bold] {grade}  ([cyan]{conf}[/cyan])")
    if g.area_cm2 is not None:
        console.print(f"Area: {g.area_cm2:.2f} cm²   Size: {g.length_mm:.0f} × {g.width_mm:.0f} mm")
    else:
        console.print(f"Area: {g.area_px:.0f} px (uncalibrated)")
    tissue = ", ".join(
        f"{c.value.replace('_', ' ')} {f * 100:.0f}%"
        for c, f in sorted(analysis.metrics.tissue.fractions.items(), key=lambda kv: -kv[1])
        if f >= 0.02
    )
    console.print(f"Tissue: {tissue}")
    palette = {"info": "blue", "warning": "yellow", "critical": "red"}
    for c in d.caveats:
        color = palette.get(c.severity.value, "white")
        console.print(f"[{color}]! {c.message}[/{color}]")
    console.print(f"\n[dim]{d.narrative}[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
