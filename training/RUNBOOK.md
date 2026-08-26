# Sorbed segmenter — server training runbook (H200 MIG)

End-to-end sequence for training the wound-segmentation model on the shared
H200 box and copying the exported ONNX weights back. Everything runs under
**tmux + a Python venv**. No secrets live in the repo — hosts are passed via
environment variables (or the defaults baked into the scripts below).

## 0. Target server at a glance

| Item        | Value                                             |
| ----------- | ------------------------------------------------- |
| GPU         | 1× NVIDIA H200 NVL, **MIG enabled**               |
| Usable VRAM | ~40 GB (a single MIG slice, **not** the full 141 GB) |
| CUDA/driver | CUDA 12.8, driver 570                             |
| OS/user     | Linux, root                                       |
| Workdir     | `/data/briefer`                                   |
| Access      | `ssh -p 30405 root@10.6.110.10`                   |
| venv        | `/data/briefer/sorbed-venv`                       |

Because MIG is on, the GPU is exposed as MIG **instances**, not as device
index 0. Training must be pinned to a MIG instance **UUID** via
`CUDA_VISIBLE_DEVICES` (see step 4). The `~40 GB` budget is why the default
config (`training/configs/seg_unetpp_effnet.yaml`) uses `batch_size: 8` at
`768 px`, which fits in one slice. Raise `--input-size` to 1024 or
lower the batch size for a larger MIG profile or on OOM.

The scripts referenced below live in `training/server/`:

| Script          | Runs on | Purpose                                        |
| --------------- | ------- | ---------------------------------------------- |
| `transfer.sh`   | laptop  | rsync the repo to the server                   |
| `setup_env.sh`  | server  | create venv + install torch (CUDA 12.x) + deps |
| `run_tmux.sh`   | server  | launch training in a detached tmux session     |
| `monitor.sh`    | server  | nvidia-smi snapshot + tail the run log         |

---

## 1. Transfer the repo (laptop → server)

From the repo root **on the laptop**:

```bash
bash training/server/transfer.sh
```

This rsyncs the working tree to `/data/briefer/<repo-name>/` over
`ssh -p 30405 root@10.6.110.10`, excluding `.git/`, virtualenvs, caches, and
large local `data/`, `artifacts/`, `logs/`, and `var/directive_packs/`. Override
any of the connection settings via env vars if needed:

```bash
SSH_HOST=10.6.110.10 SSH_PORT=30405 SSH_USER=root \
REMOTE_DIR=/data/briefer/sorbed \
bash training/server/transfer.sh
```

Preview what would be sent without writing anything:

```bash
DRY_RUN=1 bash training/server/transfer.sh
```

SSH authentication is whatever the `ssh-agent` / `~/.ssh/config` already
provides — no keys or passwords are stored in the repo.

---

## 2. Open a tmux session and set up the environment (on the server)

SSH in and **always work inside tmux** so a dropped connection never kills a
long job:

```bash
ssh -p 30405 root@10.6.110.10
tmux new -s sorbed-setup          # or: tmux attach -t sorbed-setup
cd /data/briefer/sorbed           # the REMOTE_DIR from step 1
bash training/server/setup_env.sh
```

`setup_env.sh` creates the venv at `/data/briefer/sorbed-venv`, installs
`torch`/`torchvision` from the **cu124** wheel index (override with
`CUDA_TAG=cu128` for the cu128 build — driver 570 runs both), installs
`training/requirements-train.txt`, installs the `sorbed` package editable with
the `[ml]` extra, and prints a CUDA sanity check.

Detach from tmux any time with **Ctrl-b** then **d**. The setup does not need to
stay attached; running it inside tmux means an interrupted SSH session does not
abort the (long) pip install.

---

## 3. Prepare data (on the server)

Nothing here downloads gated data automatically. Two paths:

### 3a. Open foot-ulcer data (AZH + FUSeg) — scripted

The public `uwm-bigdata/wound-segmentation` images are fetched by the repo's
own helper (real, committed images — no login):

```bash
source /data/briefer/sorbed-venv/bin/activate
cd /data/briefer/sorbed
python scripts/fetch_fuseg.py --out data/fuseg --split train --limit 1000
python scripts/fetch_fuseg.py --out data/fuseg --split validation --limit 200
```

This yields the layout `run_tmux.sh` expects by default:

```
data/fuseg/train/images/*.png      data/fuseg/train/labels/*.png
data/fuseg/validation/images/*.png data/fuseg/validation/labels/*.png
```

### 3b. Gated / licensed data — manual placement

Datasets that require a data-use agreement (or ship without a license) are
**not** downloadable by tooling. Follow `training/DATA_README.md` to place each
dataset by hand under `/data/briefer/datasets/<name>/`, then build
leakage-free, patient-level manifests:

```bash
python training/data_prep.py \
    --data-root /data/briefer/datasets/azh \
    --adapter azh_fuseg --source azh --license research-only \
    --out-dir /data/briefer/manifests/azh
```

`training/train_seg.py` can consume either a plain `--images`/`--masks` pair
(3a) or a patient manifest (`--patient-manifest` + `--folds`/`--fold`) for
grouped cross-validation. The rest of this runbook uses the simple
`--images`/`--masks` path from 3a.

### 3c. Single-model tissue segmenter — all wound types, HD_T86 7-class

This is the production path from [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md): **one**
7-class tissue segmenter whose output drives the whole HD_T86 report. It trains
on the combined corpus with partial labels — tissue-mask sets supervise the six
tissue classes; the large binary wound sets supervise localization.

```bash
source /data/briefer/sorbed-venv/bin/activate
cd /data/briefer/sorbed
export PYTHONUNBUFFERED=1

# (1) fetch the open sources
python training/fetch_corpus.py --data-root /data/briefer/corpus

# (2) re-label each TISSUE set to the unified 7-class order, then VERIFY the
#     emitted mapping_report.json before trusting the mapping (see DATA_README).
for s in dfutissue woundtissue wounds_307 complexwounddb; do
  python training/arrange_tissue.py \
      --images /data/briefer/corpus/$s/images \
      --masks  /data/briefer/corpus/$s/masks \
      --mapping training/configs/tissue_maps/$s.yaml \
      --out /data/briefer/corpus/arranged/$s
done
# then point each tissue source's root: in a data-prep config at
# /data/briefer/corpus/arranged/<name> and keep mask_kind: tissue.

# (3) build leakage-free patient-level manifests
python training/data_prep.py \
    --config training/configs/datasets.yaml \
    --out-dir /data/briefer/manifests/combined \
    --paths-relative-to /data/briefer/corpus

# (4) drop cross-source duplicate photos BEFORE training
python training/dedup.py \
    --manifest /data/briefer/manifests/combined/train.jsonl \
    --out /data/briefer/manifests/combined/train.dedup.jsonl \
    --hamming 6 --source-priority dfutissue woundtissue wounds_307 azh_fuseg

# (5) train the 7-class segmenter (launch under tmux — see step 4)
python -m training.train_seg \
    --config training/configs/seg_tissue_segformer.yaml \
    --manifest /data/briefer/manifests/combined/train.dedup.csv \
    --out-dir artifacts/seg_tissue
```

The trainer prints `partial-label superset supervision on` and the
tissue/binary row counts when the mixed manifest is detected. It exports
`artifacts/seg_tissue/model.onnx` (`logits[1,7,512,512]`) with a sha256.

### 3d. Clean previous runs of THIS project only

Scope every delete to Sorbed's own directories — never touch other projects on
the shared box. Safe to run between attempts:

```bash
# artifacts, manifests, and logs from earlier Sorbed runs only
rm -rf /data/briefer/sorbed/artifacts/*
rm -rf /data/briefer/manifests/*
rm -rf /data/briefer/corpus/arranged/*
rm -f  /data/briefer/sorbed/training/*.log
# kill only a prior Sorbed tmux training session (by our known name)
tmux has-session -t sorbed-train 2>/dev/null && tmux kill-session -t sorbed-train
```

Do **not** delete `/data/briefer/corpus/<source>` raw downloads unless you intend
to re-fetch them, and do not touch anything outside `/data/briefer`.

---

## 4. Find the MIG UUID and launch training (on the server)

List the MIG instances:

```bash
nvidia-smi -L
```

Output looks like:

```
GPU 0: NVIDIA H200 NVL (UUID: GPU-xxxxxxxx-....)
  MIG 3g.40gb Device 0: (UUID: MIG-1a2b3c4d-5e6f-7a8b-9c0d-112233445566)
```

Copy the **`MIG-...`** UUID (the whole thing) — that is what pins the run to a
single ~40 GB slice. Launch training in a detached tmux session:

```bash
MIG_UUID=MIG-1a2b3c4d-5e6f-7a8b-9c0d-112233445566 \
bash training/server/run_tmux.sh
```

`run_tmux.sh` starts a tmux session named `sorbed-train`, activates the venv,
exports `CUDA_VISIBLE_DEVICES=$MIG_UUID`, and runs:

```
python -m training.train_seg \
    --config training/configs/seg_unetpp_effnet.yaml \
    --images <REPO>/data/fuseg/train/images \
    --masks  <REPO>/data/fuseg/train/labels \
    --out-dir <REPO>/artifacts/seg_unetpp \
    --device cuda
```

Output is teed to `logs/sorbed-train-<timestamp>.log` (symlinked as
`logs/latest.log`) and mirrored in the tmux pane. Common overrides:

```bash
# Different config (SegFormer transformer recipe):
CONFIG=training/configs/seg_segformer.yaml \
MIG_UUID=MIG-... bash training/server/run_tmux.sh

# Explicit data dirs + extra flags (shorter run, smaller batch):
IMAGES_DIR=/data/briefer/datasets/azh/train/images \
MASKS_DIR=/data/briefer/datasets/azh/train/labels \
EXTRA_ARGS="--epochs 40 --batch-size 6" \
MIG_UUID=MIG-... bash training/server/run_tmux.sh
```

The script refuses to start if `MIG_UUID` is unset, if the venv is missing, if
the data directories don't exist, or if a `sorbed-train` session already runs.

---

## 5. Monitor (on the server)

From any SSH session (no need to attach to the training tmux):

```bash
bash training/server/monitor.sh
```

This prints a `nvidia-smi` snapshot (including the per-MIG memory breakdown and
`nvidia-smi -L`), the tmux session state, then follows `logs/latest.log`.
**Ctrl-C stops the tail, not the training run.** One-shot snapshot without
following:

```bash
NO_FOLLOW=1 bash training/server/monitor.sh
```

### Re-attach / detach / stop (safe tmux operations)

```bash
tmux ls                              # list sessions
tmux attach -t sorbed-train          # watch the live pane
# detach again (leaves training running):  Ctrl-b  then  d

tmux kill-session -t sorbed-train    # stop the training run
```

Attaching and detaching never interrupt the job — only `kill-session` (or the
process finishing) ends it. The pane is configured with `remain-on-exit on`, so
after training finishes the final output stays visible on attach; run
`tmux kill-session -t sorbed-train` to clean it up.

---

## 6. Export & verify ONNX (on the server)

The config sets `export_onnx: true`, so `train_seg` writes the ONNX
model at the end of the run from the best-by-val-Dice checkpoint. Artifacts in
`artifacts/seg_unetpp/`:

```
best.pt      # best-by-validation-Dice checkpoint
last.pt      # most recent epoch
model.onnx   # exported graph, (1,3,H,W) input -> (1,1,H,W) foreground logits
tb/          # TensorBoard logs
```

The run log prints the `model.onnx` SHA-256. Verify the graph loads and runs on
CPU with onnxruntime (installed by `setup_env.sh`):

```bash
source /data/briefer/sorbed-venv/bin/activate
python - <<'PY'
import numpy as np, onnx, onnxruntime as ort
path = "artifacts/seg_unetpp/model.onnx"
onnx.checker.check_model(onnx.load(path))
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
i = sess.get_inputs()[0]
h, w = i.shape[2], i.shape[3]
out = sess.run(None, {i.name: np.zeros((1, 3, h, w), np.float32)})[0]
print("ok:", i.shape, "->", out.shape)
PY
```

To re-export from an existing checkpoint (e.g. after copying a
`best.pt` around), re-running `train_seg` with `--epochs 0` is **not** supported;
instead re-run training, or load `best.pt` and call
`training.models.export_onnx` directly.

---

## 7. Copy weights back (server → laptop)

Pull only the required artifacts — run this **from the laptop**:

```bash
rsync -avhP -e "ssh -p 30405" \
    root@10.6.110.10:/data/briefer/sorbed/artifacts/seg_unetpp/model.onnx \
    ./artifacts/seg_unetpp/

# Optional: bring the checkpoint + TensorBoard logs too.
rsync -avhP -e "ssh -p 30405" \
    root@10.6.110.10:/data/briefer/sorbed/artifacts/seg_unetpp/best.pt \
    ./artifacts/seg_unetpp/
```

Then wire the model into Sorbed on the target machine by either registering its
SHA-256 in `models/registry.json` (see `docs/TRAINING.md`) or pointing the
backend straight at it:

```bash
export SORBED_ONNX_MODEL=$(pwd)/artifacts/seg_unetpp/model.onnx
```

---

## One-shot curl bootstrap (clone + setup + data + launch)

Steps 1–4 above are the explicit, inspectable path. Once the branch is on
GitHub, `training/server/bootstrap.sh` collapses them into a single command run
on the server — it clones (or fast-forwards) the repo, runs
`setup_env.sh`, fetches the open FUSeg data with `scripts/fetch_fuseg.py`, and
launches training via `run_tmux.sh`. It is idempotent and safe to re-run.

SSH in (inside tmux, so the clone/install survives a dropped connection), grab
the MIG UUID, then curl-and-go:

```bash
ssh -p 30405 root@10.6.110.10
tmux new -s sorbed-boot
nvidia-smi -L                                   # copy the MIG-... UUID

export MIG_UUID=MIG-xxxxxxxx-....
curl -fsSL \
  https://raw.githubusercontent.com/ArioMoniri/Sorbed/claude/bedsore-grading-system-wcd4ol/training/server/bootstrap.sh \
  | bash
```

Everything is env-overridable (the script streams over a pipe, so there is no
argv — configuration is entirely by environment variable):

| Variable | Default | Purpose |
| --- | --- | --- |
| `MIG_UUID` | — (required) | MIG instance UUID to pin the run (skip if `SKIP_TRAIN=1`) |
| `REPO_DIR` | `/data/briefer/sorbed` | checkout location |
| `BRANCH` | `claude/bedsore-grading-system-wcd4ol` | branch to track |
| `CONFIG` | `training/configs/seg_unetpp_effnet.yaml` | training YAML |
| `CUDA_TAG` | `cu128` | torch wheel index (driver 570 / CUDA 12.8) |
| `SKIP_SETUP=1` | off | reuse an existing venv (no install) |
| `SKIP_DATA=1` | off | data already placed; skip the FUSeg download |
| `SKIP_TRAIN=1` | off | stop after setup+data; do not launch training |

The script prints the monitor / attach / stop commands when it finishes. To
launch a different recipe (e.g. the SegFormer config) override `CONFIG`:

```bash
CONFIG=training/configs/seg_segformer.yaml MIG_UUID=MIG-... \
  curl -fsSL https://raw.githubusercontent.com/ArioMoniri/Sorbed/claude/bedsore-grading-system-wcd4ol/training/server/bootstrap.sh | bash
```

To review the script before running it (recommended for any
piped-to-shell command), download it first:

```bash
curl -fsSL -o bootstrap.sh \
  https://raw.githubusercontent.com/ArioMoniri/Sorbed/claude/bedsore-grading-system-wcd4ol/training/server/bootstrap.sh
less bootstrap.sh && MIG_UUID=MIG-... bash bootstrap.sh
```

---

## Out-of-memory safety (no crashes on the shared slice)

The trainers guard both RAM and VRAM (`training/memory.py`), so a memory spike
degrades throughput instead of killing the run:

- **VRAM OOM → automatic recovery.** On a CUDA out-of-memory error the training
  step empties the cache and **retries the batch split into more micro-batches**
  (gradient accumulation — the effective batch is unchanged). The working split
  is remembered, so later steps skip straight to it. The log shows
  `[mem] CUDA OOM — retrying batch in N micro-batches`; validation
  forward passes shrink the same way.
- **Proactive cap.** Set `--vram-fraction 0.92` (or the `vram_fraction:` config
  key) to cap the process to a share of the slice so it OOMs *catchably* early
  rather than starving the device. `run_tmux.sh` also exports
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to cut fragmentation OOMs.
- **RAM.** `--num-workers` is auto-capped by available system RAM and CPU count,
  so a large value can't OOM the host. Lower `--input-size` (768 → 512) or
  `--batch-size` if a run still can't fit.

If OOM persists even at micro-batch = 1, the run stops with a clear message to
lower `--input-size`/`--batch-size` rather than raising a bare CUDA crash.

## MedSAM / SAM weights without an API key (mask-factory path)

The optional MedSAM mask-factory needs SAM-family weights. Fetch them with **no
HuggingFace token and no API key** (public URLs), then fine-tune offline:

```bash
source /data/briefer/sorbed-venv/bin/activate
python -m training.fetch_models --list                         # see all no-auth sources
python -m training.fetch_models medsam-vit-base --out /data/briefer/models
python -m training.finetune_medsam train \
    --weights-dir /data/briefer/models/medsam-vit-base \
    --manifest /data/briefer/manifests/combined/train.csv \
    --out-dir artifacts/medsam
```

`fetch_models` also serves `sam-vit-base/large/huge` (transformers format) and the
original Meta SAM / SAM-2.1 CDN checkpoints. `SAM 3` is gated (needs a token) and
is not offered.

## Quick reference

```bash
# laptop
bash training/server/transfer.sh

# server (inside tmux)
cd /data/briefer/sorbed
bash training/server/setup_env.sh
python scripts/fetch_fuseg.py --out data/fuseg --split train --limit 1000
python scripts/fetch_fuseg.py --out data/fuseg --split validation --limit 200
nvidia-smi -L                                   # copy the MIG-... UUID
MIG_UUID=MIG-... bash training/server/run_tmux.sh
bash training/server/monitor.sh                 # Ctrl-C stops only the tail

# laptop (after training)
rsync -avhP -e "ssh -p 30405" \
  root@10.6.110.10:/data/briefer/sorbed/artifacts/seg_unetpp/model.onnx ./artifacts/seg_unetpp/
```
