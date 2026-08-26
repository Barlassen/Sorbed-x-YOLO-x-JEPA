# Model Cards & Weight Provenance

Sorbed **never vendors model weights** in this repository. Every learned artifact
the system can load is described by an entry in the model registry
(`src/sorbed/models/registry.py`, with real entries kept in a local
`models/registry.json`), which doubles as a lightweight model card.

## The provenance policy

Each set of weights is:

1. **Downloaded only on explicit request.** Nothing is fetched during install,
   import, or a normal analysis run. The weight-free `classical` backend always
   works offline, so the pipeline is fully functional with zero downloads.
2. **SHA-256 verified on arrival.** `download_weights()` streams the file, hashes
   it, and **fails closed** — the file is deleted and an error raised — if the
   digest does not match the value recorded in the registry. Unverified bytes
   never reach inference.
3. **Format-restricted.** Only `.onnx` and `.safetensors` are accepted. Sorbed
   does not load pickle-based checkpoints, which can execute arbitrary code on
   load.
4. **License-recorded.** Every entry names its license. Code dependencies here
   are uniformly permissive (Apache-2.0 / MIT / BSD); the constraints that matter
   are on the training *data* (see `docs/MODELS.md`) and must be checked per
   dataset before any deployment.

## What lives here

- **`classical.md`** — the card for the weight-free classical segmentation
  backend (the always-available default).
- **`TEMPLATE.md`** — copy this to `<model-name>.md` when you register a trained
  model. Fill in the dataset, license, metrics, intended use, limitations, and —
  importantly — the skin-tone evaluation.

## Registering a trained model

After exporting a model with `scripts/train_segmenter.py` (see
`docs/TRAINING.md`):

1. Compute its SHA-256 (the script prints it).
2. Add an entry to `models/registry.json` with the URL you host it at, the
   digest, the license, `backend: "onnx"`, and the `input_size` it expects.
3. Copy `TEMPLATE.md` to `model_cards/<name>.md` and complete it.

An entry without a completed model card should be treated as unreviewed and not
used clinically.
