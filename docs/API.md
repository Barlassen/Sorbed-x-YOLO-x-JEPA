# Sorbed HTTP API

A FastAPI service that runs the Sorbed pressure-injury analysis pipeline over
HTTP. Every endpoint is mounted under the `/v1` prefix.

> **Clinical notice.** Sorbed is decision-support software, **not** a medical
> device and **not** a diagnostic tool. Results are provisional estimates
> computed from a single 2D photograph and must be reviewed by a qualified
> clinician. Depth, undermining, and tunneling cannot be measured from an image,
> and detection of early-stage or deep-tissue injury is less reliable on darker
> skin.

## Privacy posture

- **Processed in memory.** Uploaded images are decoded, analyzed, and discarded
  when the request completes. The analyze endpoint writes nothing to disk.
- **No patient data is persisted by the service.** The `SORBED_OUTPUT_DIR` and
  the mounted `/models` volume exist for operator-initiated report writing and
  model weights, not for storing incoming photographs.
- **Strip PHI before upload.** Remove patient identifiers from filenames and
  image metadata (EXIF/DICOM tags) before sending an image. The service reads
  pixels for analysis; it does not need identifying metadata.
- Responses contain only the computed analysis. No original image bytes are
  returned unless the rendered guide is explicitly requested.

## Running the service

Local (with the `api` extra installed):

```bash
uvicorn sorbed.api.app:app --host 0.0.0.0 --port 8000
```

Docker:

```bash
docker compose up --build
# or
docker build -t sorbed-api . && docker run -p 8000:8000 sorbed-api
```

Interactive docs are served at `http://localhost:8000/docs`
(OpenAPI JSON at `/openapi.json`).

## Configuration

Pipeline behavior is controlled by `SORBED_*` environment variables (see
`sorbed.config.settings.Settings`). Transport behavior uses `SORBED_API_*`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SORBED_API_MAX_UPLOAD_BYTES` | `26214400` (25 MiB) | Reject larger uploads with 413. |
| `SORBED_API_CORS_ORIGINS` | `["*"]` | Allowed CORS origins (JSON list). |
| `SORBED_API_CORS_ALLOW_CREDENTIALS` | `false` | Send CORS credentials. |

## Errors

All errors are returned as JSON with a top-level `error` key, e.g.
`{"error": "could not detect a supported image format from the upload"}`.

---

## Endpoints

### `GET /v1/health`

Liveness. Always `200` while the process is serving.

```bash
curl -s http://localhost:8000/v1/health
```

```json
{ "status": "ok", "version": "0.1.0" }
```

### `GET /v1/ready`

Readiness. Builds the analyzer and its backends; returns `200` when usable and
`503` otherwise.

```bash
curl -s http://localhost:8000/v1/ready
```

```json
{
  "ready": true,
  "backends": {
    "segmentation": "classical",
    "tissue": "color_model",
    "staging": "rule_engine"
  },
  "detail": null
}
```

### `GET /v1/formats`

Which container formats have a currently usable decoder. HEIC/DICOM/RAW require
the optional `formats` extra (installed in the Docker image).

```bash
curl -s http://localhost:8000/v1/formats
```

```json
{
  "png": true, "jpeg": true, "webp": true, "bmp": true, "gif": true,
  "tiff": true, "heif": false, "dicom": false, "raw": false
}
```

### `GET /v1/models`

Configured backends and any learned models discoverable through the optional
model registry (empty when the registry is not installed).

```bash
curl -s http://localhost:8000/v1/models
```

```json
{
  "backends": {
    "segmentation": "classical",
    "tissue": "color_model",
    "staging": "rule_engine"
  },
  "ml_models": []
}
```

### `GET /v1/schema`

The JSON schema of the canonical `WoundAnalysis` result — the single source of
truth for the analysis payload.

```bash
curl -s http://localhost:8000/v1/schema
```

### `POST /v1/analyze`

Analyze one uploaded image. `multipart/form-data`.

**Form fields**

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `file` | file | yes | The wound photograph. |
| `mm_per_px` | float | no | Manual scale; enables physical measurements. |
| `marker_mm` | float | no | Length of a fiducial marker for calibration. |
| `coin_mm` | float | no | Diameter of a reference coin for calibration. |
| `include_guide` | bool | no | Also return the annotated guide as base64 PNG. |

`include_guide` may also be passed as a query parameter
(`?include_guide=true`), which takes precedence over the form value.

**Validation**

- `413` if the upload exceeds `SORBED_API_MAX_UPLOAD_BYTES`.
- `415` if the format cannot be detected from the file content.
- `422` if the file is empty or the pipeline cannot analyze it.

**Example**

```bash
curl -s -X POST http://localhost:8000/v1/analyze \
  -F "file=@wound.jpg" \
  -F "mm_per_px=0.12"
```

With the annotated guide:

```bash
curl -s -X POST "http://localhost:8000/v1/analyze?include_guide=true" \
  -F "file=@wound.png"
```

**Response** (`200`)

```json
{
  "analysis": {
    "schema_version": "1.0.0",
    "analysis_id": "…",
    "created_at": "…",
    "decision": { "stage": "stage_2", "confidence": 0.63, "…": "…" },
    "metrics": { "…": "…" },
    "disclaimer": "Sorbed is clinical decision-support software, not a medical device …"
  },
  "guide_png_base64": null
}
```

`analysis` is the `WoundAnalysis` serialized with `model_dump(mode="json")`.
When `include_guide` is set, `guide_png_base64` holds a base64-encoded PNG of the
annotated clinical guide (decode and save it as a `.png`).
