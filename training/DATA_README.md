# Preparing wound datasets for training

This directory converts wound-imaging datasets on disk into **patient-level,
leakage-free** training manifests. It does not download data. Several of the
relevant datasets are **gated** (an application and data-use agreement are
required) or ship **without an explicit license** — they are placed locally by
hand, then the tooling is pointed at the folder.

> **Scope.** The strongest open segmentation data is diabetic-foot-only and
> carries no severity grade. There is **no** open, permissively-licensed dataset
> that is simultaneously pressure-injury, NPIAP-staged, segmented, and
> longitudinal. The wound-**segmentation** model is built from open data;
> staged/longitudinal pressure-injury data is sourced through a clinical
> partnership. Every `license` tag written into the manifest is hygiene metadata,
> not legal advice — each source's terms should be re-checked before any
> non-research use.

## What the tooling expects

Two on-disk layouts are supported by the adapters in `datasets.py`:

- **`azh_fuseg`** — per-split folders, each with an image and a label directory:

  ```
  <root>/
    train/        images/ *.png   labels/ *.png
    validation/   images/ *.png   labels/ *.png
    test/         images/ *.png   (labels/ may be absent — FUSeg holds them out)
  ```

  A flat `<root>/images` + `<root>/labels` also works. All samples are tagged
  `body_part=foot`.

- **`generic`** — a flat dump paired by filename **stem**:

  ```
  <root>/
    images/   0001.jpg 0002.jpg ...
    masks/    0001.png 0002.png ...   # wound = nonzero (0/255 or 0/1)
  ```

  Accepted image/mask dir names: `images|image|img|imgs` and
  `labels|label|masks|mask|annotations|gt`.

Because AZH/FUSeg and most open dumps do **not** track patients, each image
becomes its own singleton `patient_id` (`"<source>:<stem>"`). The split is still
leakage-free by construction; there is simply no cross-visit grouping to protect.
For a source that *is* longitudinal (repeat visits per patient), write a manifest
directly with real `patient_id` / `visit_index` values (schema below) instead of
using an auto-adapter; this preserves the no-leakage guarantee across visits.

## Manifest schema

Each manifest row (JSONL object / CSV row) has these columns:

| column | meaning |
| --- | --- |
| `image_path` | RGB photo path (relative to `--paths-relative-to`, else absolute) |
| `mask_path` | binary mask path; empty string when the source has no mask |
| `body_part` | anatomical site: `foot`, `sacrum`, `mixed`, `unknown`, … |
| `stage_label` | optional severity grade (e.g. `stage_2`); empty when ungraded |
| `patient_id` | grouping key for leakage-free splits |
| `visit_index` | 0-based longitudinal visit index (`0` if not longitudinal) |
| `source` | dataset id (`azh`, `fuseg`, `piid`, …) |
| `license` | usage tag copied from the source |

## Dataset-by-dataset placement

Only place datasets whose terms have been satisfied. URLs below are the datasets'
own access pages — **do not** assume a direct download link where a dataset is
gated.

### AZH Chronic Wound + FUSeg (open, research-only) — starting point

- Source: `github.com/uwm-bigdata/wound-segmentation`.
- Clone the repo; its `data/` folder holds the AZH and FUSeg image/label splits.
- Point a source at the challenge folder with `adapter: azh_fuseg`.
- Foot-ulcer binary masks; no severity grade; no explicit data license — treat as
  research-only.

### DFUC 2020 / 2021 / 2022 / 2024 (GATED — data-use agreement)

- Access: `dfu-challenge.github.io` and the corresponding Grand Challenge pages.
- An application and DUA acceptance by email/registration are required. There is
  no open download link. After the data is received, arrange it as a `generic`
  layout (segmentation editions) and set `license: dfu-dua` (non-commercial).
- DFUC 2021 provides infection/ischaemia **patch labels** (not full masks); use it
  for the grading head, not segmentation.

### WoundsDB / chronicwounddatabase.eu (open, registration)

- Access: register at `chronicwounddatabase.eu` via a browser (the site blocks
  scrapers). Provides RGB + thermal + depth/3D with expert outlines; some repeat
  visits (weakly longitudinal), mixed body parts.
- Export the RGB photos and expert outline masks into a `generic` layout. If
  per-patient visit grouping can be recovered, write `patient_id`/`visit_index`
  to keep visits from leaking.

### Medetec (open, free reuse)

- Access: `medetec.co.uk`. Aetiology-labelled clinical stock photos; **no masks**
  ship with them. Use as an unlabeled/pseudo-labeled pool or after deriving masks.
  Set `require_masks: false` if importing image-only.

### PIID (open, no stated license) — Stage 1–4 classification

- Access: `github.com/FU-MedicalAI/PIID`. Whole-image stage labels, **no masks**,
  no DTI/unstageable, small (~1,091). Import with `require_masks: false` and fill
  `stage_label` from the class folder names. No explicit license — research risk.
- **Guideline caveat:** most sources describe PIID's stages as **EPUAP I–IV**, not
  NPIAP — confirm against the dataset card before relying on it for the teacher.
  Stages 1–4 share the NPIAP/EPUAP 2019 joint definitions, so the numeric labels
  line up; only the provenance note differs. DTI/Unstageable are absent, so PIID
  cannot supply those two classes.

### Kaggle mixed sets (open, inherit source licenses)

- `kaggle.com/datasets/leoscode/wound-segmentation-images` (2,760-image merge of
  Medetec + FUSeg + WSNET) and `kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu`.
  Import segmentation merges with `adapter: generic`. The 2,760 merge already
  dedupes Medetec/FUSeg/WSNET — do not also import those separately.

### WSNet / WOUNDSEG (open, research)

- `github.com/subbareddy248/WSNET`. Multi-type wound masks (8 wound types), mixed
  body parts. `adapter: generic`.

## Running the prep

Single source:

```bash
python training/data_prep.py \
  --data-root /data/briefer/datasets/wound-segmentation/data/Foot_Ulcer_Segmentation_Challenge \
  --adapter azh_fuseg --source azh_fuseg --license research-only \
  --out-dir /data/briefer/manifests/azh \
  --paths-relative-to /data/briefer/datasets
```

Multiple sources via config (copy `configs/datasets.example.yaml` to
`configs/datasets.yaml`, edit paths):

```bash
python training/data_prep.py \
  --config training/configs/datasets.yaml \
  --out-dir /data/briefer/manifests/combined \
  --paths-relative-to /data/briefer/datasets
```

Outputs, per run: `train.jsonl` / `train.csv` (and `val.*`, `test.*`) plus
`summary.json` with per-split counts by source, body part, and stage label. The
run asserts no `patient_id` appears in two splits before writing.

## Feeding the training loop

There are two entry points into the training loop, and they are complementary:

- **Directory path (default in `RUNBOOK.md`).** `training/train_seg.py` reads
  `--images`/`--masks` directories directly (paired by stem) and takes an optional
  `--patient-manifest` (a simple `stem,patient_id` CSV) for grouped K-fold. This is
  the shortest route for the open AZH/FUSeg data fetched by `scripts/fetch_fuseg.py`.
- **Manifest path (multi-source / MedSAM / grouped-CV).** `datasets.py` exposes
  `WoundSegmentationDataset`, which reads a `data_prep.py` manifest split and yields
  the `(image CHW float32, mask 1×H×W float32)` tensors, and `finetune_medsam.py`
  consumes the same manifests via `read_manifest`. Use this when aggregating
  several sources or when visit-level leakage control is needed.

Augmentation and colour normalization come from `build_train_transform` /
`build_val_transform` (albumentations, gray-world white-balance + ImageNet
standardization). Install the training extras first:

```bash
pip install torch torchvision segmentation-models-pytorch albumentations
```

(These are **not** part of the Sorbed core install; `datasets.py` imports them
lazily so this module lints and imports without them.)

## Rich combined corpus

The single-dataset flow above builds a foot-ulcer segmenter. The **rich combined
corpus** widens that to every open, non-interactively fetchable wound source the
dataset audit verified, aggregated for all three training tasks:
**segmentation**, the **MedSAM mask-factory** (staged/boxed-but-maskless sets
turned into mask targets via box/point prompts), and **teacher-student
grading** (pressure-injury stage plus orthogonal tissue-type / depth teachers).

Three pieces compose it:

1. **`training/fetch_corpus.py`** pulls the open sources into one data root.
2. **`training/configs/datasets.yaml`** is the manifest that names each source,
   its `root:` (one sub-directory per source under that data root), adapter,
   license, body part, and `require_masks`.
3. **`training/data_prep.py --config …`** scans them into leakage-free
   `train/val/test` manifests (same schema as above).

```bash
# 1. fetch the open sources (git / Mendeley / Kaggle / Roboflow / FUSeg script)
python training/fetch_corpus.py --data-root /data/briefer/corpus

# 2. aggregate into manifests; paths stored relative to the corpus root
python training/data_prep.py \
    --config training/configs/datasets.yaml \
    --out-dir /data/briefer/manifests/combined \
    --paths-relative-to /data/briefer/corpus
```

`fetch_corpus.py --list` prints every source with its fetch method and task
tags; `--dry-run` prints the exact commands without downloading; `--source NAME`
(repeatable) fetches a subset. Nothing is fetched without `--data-root`.

### Fetch methods (all URLs from the dataset audit)

- **`fuseg`** — reuses `scripts/fetch_fuseg.py` (raw GitHub, no login).
- **`git`** — shallow `git clone` of a public repo.
- **`mendeley`** — Mendeley Data public files API (CC-BY, no login). If the
  public API is unreachable/changed, the tool prints the landing-page URL and
  the expected local path instead of a guessed link.
- **`kaggle`** — `kaggle` CLI; needs a free API token at `~/.kaggle/kaggle.json`.
- **`roboflow`** — Roboflow SDK export; needs a free `ROBOFLOW_API_KEY`.
- **`manual`** — gated / registration / audit-**unverified** host. Never
  downloaded: the tool prints the exact human step and the local path where the
  prepared data must land, and the source sits **commented** in `datasets.yaml`.

### Per-task source lists

**Segmentation (real masks):** `azh_fuseg` (foot), `lowerlimb_feet` (2,686
masks, lower limb/feet), `co2wounds_v2` (764, COCO + binary), `wsnet` (8 wound
types, mixed). Gated add-ons: `dfuc`, `woundsdb`.

**MedSAM mask-factory (staged/boxed → masks):** `roboflow_stages` (stage
bounding boxes → box prompts), `kaggle_stages` and `piid` (class-only → point
prompts), plus `lowerlimb_feet` box seeds. Gated add-on: `medetec` (maskless
pseudo-label pool).

**Teacher-student grading:**
- *Stage head* — `piid` (EPUAP I–IV, confirm attribution), `roboflow_stages`
  and `kaggle_stages` (stage 1–4, **guideline unconfirmed — flag**).
- *Tissue / depth heads (orthogonal, DESIGN-R-style sub-scores)* — `dfutissue`
  (8 classes incl. tendon/bone), `complexwounddb` (5 classes). Gated/unverified:
  `wounds_307`, `woundtissue_147`.

### Layout, flags, and caveats

The adapters understand two layouts only — `azh_fuseg` (per-split `images/` +
`labels/`) and `generic` (flat `images/` + `masks/` paired by stem). Sources
that do not already ship one (Mendeley zips, Roboflow COCO exports, the tissue
mono-repos) must be **re-arranged into a `generic` layout** after fetching; set
`require_masks: false` for the classification/staging sources so their maskless
images still import for the grading head and MedSAM inputs. `fetch_corpus.py`
nests two sources in sub-folders (`dfutissue/DFUTissue`,
`complexwounddb/ComplexWoundDB`), which the `root:` paths in `datasets.yaml`
already account for.

Audit flags that travel with the config: guideline attribution is
**unconfirmed** for `roboflow_stages` / `kaggle_stages` and PIID (EPUAP vs
NPIAP); licenses are **unstated** for `dfutissue` and `complexwounddb`; and the
hosts for `wounds_307` and `woundtissue_147` are **UNVERIFIED** — their download
URLs are not confirmed. As stated above, there is no open dataset that is
simultaneously pressure-injury, NPIAP-staged, segmented, and longitudinal; the
combined corpus approximates it by fusing separate open axes, and true
staged/longitudinal pressure-injury data requires a clinical partnership.

### Tissue-mask arrange step (for the single 7-class segmenter)

The single-model tissue segmenter (`configs/seg_tissue_segformer.yaml`) needs
masks in ONE unified order — `0 background, 1 epithelial, 2 granulation,
3 slough, 4 eschar, 5 adipose, 6 deep_structure`. No public tissue set ships
this schema, so each tissue source is re-labelled once with
`training/arrange_tissue.py` and an operator-confirmed mapping file
(`configs/tissue_maps/*.yaml`):

```bash
python training/arrange_tissue.py \
    --images /data/sorbed/corpus/dfutissue/DFUTissue/images \
    --masks  /data/sorbed/corpus/dfutissue/DFUTissue/labels \
    --mapping training/configs/tissue_maps/dfutissue.yaml \
    --out /data/sorbed/corpus/arranged/dfutissue
```

It writes `images/` + `masks/` in the unified order plus a `mapping_report.json`
listing every native mask value and its pixel count — **read it and confirm the
mapping before training**; an unmapped value is an error by default so an unknown
tissue can never be silently folded into background. Point the tissue sources'
`root:` in a data-prep config at these arranged directories and keep their
`mask_kind: tissue`; leave the binary wound sets as `mask_kind: binary` (they
supervise localization via the loss's partial-label superset term).

De-duplicate before splitting so a shared photo cannot land in two splits and
inflate the score:

```bash
python training/dedup.py \
    --manifest /data/sorbed/manifests/combined/train.jsonl \
    --out /data/sorbed/manifests/combined/train.dedup.jsonl \
    --hamming 6 --source-priority dfutissue woundtissue wounds_307 azh_fuseg
```

`--source-priority` keeps the representative from the higher-priority (better-
masked) source; the emitted `*.dedup_report.json` records every dropped pair.
