# Usage

This document covers running Sorbed from the command line and from Python, calibrating scale, and reading the results. For the clinical reasoning behind the grades, see [`CLINICAL.md`](CLINICAL.md); for the data-and-model description, [`MODELS.md`](MODELS.md).

> Sorbed is decision support, not a diagnosis. Every result must be reviewed by a qualified clinician. See [`../DISCLAIMER.md`](../DISCLAIMER.md).

## Install

The core install runs the whole classical pipeline offline, with no model downloads:

```bash
pip install -e .
```

Add extras only for what you need:

| Extra | Enables |
|---|---|
| `[formats]` | HEIC/HEIF (iPhone), DICOM, camera RAW, BigTIFF decoders |
| `[ml]` | optional ONNX learned segmentation/staging backends |
| `[api]` | the FastAPI service |
| `[report]` | PDF report rendering |
| `[dev]` | ruff, mypy, pytest, hypothesis |
| `[all]` | formats + ml + report + api |

```bash
pip install -e ".[formats,api]"
```

Check which decoders resolved in the current environment with `sorbed formats`.

## CLI

Sorbed installs a single `sorbed` entry point. Running it with no arguments prints help.

### `sorbed analyze`

Analyze one wound image and write a report.

```bash
sorbed analyze IMAGE [OPTIONS]
```

| Option | Meaning |
|---|---|
| `IMAGE` | Path to the wound image (any supported format). |
| `-o`, `--out DIR` | Output directory (default `reports`). |
| `--mm-per-px FLOAT` | Known scale in millimeters per pixel. |
| `--marker-mm FLOAT` | ArUco marker side length in mm (detected in the image). |
| `--coin-mm FLOAT` | Reference coin diameter in mm (detected in the image). |
| `-f`, `--format TEXT` | Comma list of artifacts, or `all` (default). |
| `--json` | Print the analysis JSON to stdout instead of the summary. |

Examples:

```bash
# Plain phone photo, uncalibrated — pixel-space metrics only.
sorbed analyze wound.jpg

# Manual scale, custom output folder.
sorbed analyze wound.jpg --mm-per-px 0.15 --out out/case42

# Calibrate from a 20 mm ArUco marker placed beside the wound.
sorbed analyze wound.heic --marker-mm 20

# Only write the mask and JSON.
sorbed analyze wound.png --format mask,json

# Emit the canonical JSON to stdout (pipe it to jq).
sorbed analyze wound.jpg --mm-per-px 0.15 --json
```

Default console output:

```
Provisional grade: Stage 3  (62%)
Area: 8.40 cm²   Size: 41 × 29 mm
Tissue: granulation 58%, slough 27%, eschar 9%
! Depth-dependent grade — confirm Stage 3 vs 4 at the bedside.

A full-thickness appearance with visible adipose supports Stage 3; ...
```

When the grade is withheld the confidence shows as `withheld`, and when the image is uncalibrated the size line reports pixels: `Area: 74210 px (uncalibrated)`.

The artifacts written to `--out` are named `sorbed_<id>*` (see [Outputs](#interpreting-the-artifacts)).

### `sorbed inspect`

Decode an image and report its metadata and detected scale, without grading it.

```bash
sorbed inspect wound.dcm
```

Prints the detected format, dimensions, bit depth, calibration status, the resolved scale (if any), and any retained metadata tags. This confirms that a DICOM's pixel spacing was read, or that a fiducial was found, before a full analysis is run.

### `sorbed formats`

List every image format and whether a decoder for it is available in the current environment (`yes` vs `needs extras`).

### `sorbed schema`

Print the JSON Schema of the analysis result contract (`WoundAnalysis`), or write it to a file:

```bash
sorbed schema --out analysis.schema.json
```

### `sorbed config`

Show the effective configuration and its digest (the same digest recorded in every analysis under `config_digest`).

### `sorbed version`

Print the installed Sorbed version.

## Calibrating scale

Physical measurements (cm², mm) require knowing how many millimeters one pixel spans. Sorbed does not fabricate this — with no scale, area and length come back as `null` and only pixel metrics are reported. There are three ways to supply it:

1. **Manual `--mm-per-px`.** When the scale is already known (e.g. from a fixed-distance imaging rig), pass it directly: `--mm-per-px 0.15`. Recorded as calibration status `manual_mm_per_px`.
2. **ArUco fiducial marker (`--marker-mm`).** Place a printed ArUco marker of known side length flat in the wound plane and pass that length in millimeters. Sorbed detects the marker and derives the scale. Status `fiducial_marker`.
3. **Reference coin (`--coin-mm`).** Place a coin of known diameter in the wound plane and pass the diameter. Status `fiducial_marker` via a circular reference.

DICOM images can carry pixel spacing in their headers; when present it is read automatically (status `dicom_pixel_spacing`) and no flag is needed. Keep the reference object in the **same plane** as the wound — a marker lying on the mattress beside a wound on a curved heel will mis-scale.

## Interpreting the JSON

The canonical `WoundAnalysis` JSON is the authoritative record; every image artifact is a rendering of it. The most commonly read fields:

### `decision`

```jsonc
{
  "stage": "stage_3",            // stage_1..stage_4, unstageable, deep_tissue_injury,
                                 // mucosal_not_stageable, not_pressure_injury, indeterminate
  "confidence": 0.62,            // calibrated confidence in [0, 1]
  "abstained": false,            // true => Sorbed declined to commit; treat stage as advisory
  "evidence": [                  // ranked trace; each item points at a real metric field
    {
      "code": "adipose_visible",
      "description": "Exposed adipose in the wound bed supports full-thickness loss.",
      "metric_ref": "metrics.tissue.fractions.adipose",
      "observed": 0.11, "threshold": 0.02, "comparison": ">=",
      "direction": "supports", "weight": 1.0, "source": "rule:R_STAGE3"
    }
  ],
  "caveats": [                   // limitations surfaced prominently (severity info/warning/critical)
    { "severity": "warning", "message": "Depth-dependent grade — confirm at the bedside." }
  ],
  "narrative": "..."             // deterministic human-readable summary
}
```

`abstained: true` — or a stage of `indeterminate` — means the evidence did not support a confident grade and Sorbed deferred to a clinician. The `evidence` list is the auditable core of the result: each `metric_ref` resolves to an actual field elsewhere in the JSON, so any claim can be traced back to the number that produced it.

### `metrics.geometry`

```jsonc
{
  "area_px": 74210.0, "perimeter_px": 1103.4,
  "length_px": 273.0, "width_px": 193.0,
  "circularity": 0.71, "solidity": 0.94,
  "wound_fraction_of_image": 0.18,
  "area_cm2": 8.40, "area_mm2": 840.1,     // null when uncalibrated
  "length_mm": 41.0, "width_mm": 29.0      // null when uncalibrated
}
```

Length is the greatest head-to-toe extent, width the greatest extent perpendicular to it (falling back to the mask's major/minor axes when the head direction is unknown — see `axis_convention`). The `*_mm` / `*_cm2` fields are populated only when a scale was resolved.

### `metrics.tissue.fractions`

Per-tissue area fractions over the **wound bed only**, summing to 1:

```jsonc
{ "granulation": 0.58, "slough": 0.27, "eschar": 0.09, "epithelial": 0.06 }
```

`metrics.tissue.dominant` names the largest class, and `metrics.tissue.mean_confidence` gives per-class classifier confidence. As described in [`MODELS.md`](MODELS.md), these are percentage estimates with confidence, not per-pixel claims, and slough and granulation are frequently confused.

### `healing_scores`

Image-derivable sub-scores of the standard monitoring instruments. Items needing palpation or probing are left `null`, so totals are explicitly partial:

```jsonc
{
  "push_size_subscore": 7, "push_tissue_subscore": 3, "push_partial_total": 10,
  "design_r_size_subscore": 12, "granulation_percent": 58.0,
  "notes": ["Exudate not assessable from image; PUSH total is partial."]
}
```

### Other useful fields

- `calibration.status` — how scale was established (`uncalibrated` is normal, not an error).
- `skin_tone_band` — `fitzpatrick_i_iii`, `fitzpatrick_iv_vi`, or `unknown`; drives the equity confidence adjustment (see [`EQUITY.md`](EQUITY.md)).
- `metrics.depth_proxy` — a shading-derived *relative* index in [0, 1] with `is_physical_measurement: false`. It is weak, flagged evidence, **not** a depth.
- `metrics.color_cues` — the maroon/purple, erythema, and open-bed fractions the DTPI/Stage-1 evidence points at.
- `metrics.periwound` — erythema index and suspected maceration in the skin ring around the wound.
- `provenance` — which backends ran and any weight sha256s, for reproducibility.
- `disclaimer` — the standing clinical disclaimer, embedded in every record.

## Interpreting the artifacts

- **Mask** (`*_mask.png`) — the binary wound boundary. Used to verify that Sorbed outlined the wound and not a shadow or dressing.
- **Overlay** (`*_overlay.png`) — tissue classes colored over the original photo. Used to verify that granulation, slough, and eschar landed on the correct regions.
- **Guide** (`*_guide.png`) — the clinician-facing annotated photo: wound contour, length × width, a scale bar (when calibrated), a tissue legend, and the grade banner.
- **Schematic** (`*_schematic.png`) — a synthetic drawing of the sore with no photographic noise, for records or side-by-side monitoring over time.
- **Schematic guide** (`*_schematic_guide.png`) — the schematic annotated with measurements and labels.
- **HTML** (`*.html`) — a single self-contained report bundling the guide image, the grade with its evidence and caveats, metrics, and the disclaimer. Open it in any browser; install `[report]` to also render PDF.

## Python API

The same pipeline the CLI uses is available as a two-line import:

```python
from sorbed.pipeline import analyze_image, AnalyzeOptions

bundle = analyze_image("wound.jpg", options=AnalyzeOptions(mm_per_px=0.15))
analysis = bundle.analysis

print(analysis.decision.stage, round(analysis.decision.confidence, 2))
print(analysis.metrics.geometry.area_cm2)          # None if uncalibrated
print(analysis.metrics.tissue.fractions)
print(analysis.model_dump_json(indent=2))          # the canonical record
```

`AnalyzeOptions` accepts `mm_per_px`, `marker_length_mm`, `coin_diameter_mm`, and `head_vector` (a `(dy, dx)` direction toward the patient's head, used to orient length vs width). `analyze_image` returns an `AnalysisBundle` carrying the `analysis` plus the wound mask and tissue label map used to render artifacts.

To write the report artifacts yourself:

```python
from sorbed.report.builder import write_report

report = write_report(bundle, "reports")          # all artifacts by default
for artifact in report.artifacts:
    print(artifact.kind, artifact.path, artifact.bytes)
```

## HTTP API

With the `[api]` extra installed:

```bash
uvicorn sorbed.api.app:app --reload      # or: make serve
```

The service exposes `POST /v1/analyze`, `GET /v1/health`, and `GET /v1/formats`, running the identical pipeline behind HTTP.
