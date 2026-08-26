# Model Card — <MODEL NAME>

Copy this file to `model_cards/<registry-name>.md` and complete every section
before using the model for anything beyond experimentation. Delete the guidance
in angle brackets as you fill it in.

## Summary

- **Name:** `<registry name, matches models/registry.json>`
- **Backend:** `onnx`
- **Architecture:** `<e.g. U-Net, EfficientNet-b0 encoder>`
- **Input size:** `<H x W, e.g. 512 x 512>`
- **Output:** `<e.g. (1,1,H,W) foreground logit>`
- **License:** `<weights license>`
- **SHA-256:** `<hex digest recorded in the registry>`
- **Download URL:** `<where the weights are hosted>`
- **Version / date:** `<version and export date>`

## Training data

- **Dataset(s):** `<name(s), size(s), and source URL(s)>`
- **Data license(s):** `<per-dataset; note any non-commercial restriction>`
- **Split:** `<train/val fractions, seed, any held-out test set>`
- **Preprocessing / augmentation:** `<resize, normalization (mean/std), flips, ...>`

Be explicit about redistribution rights. A permissive *code* stack does not make
a non-commercial dataset commercially usable (see `docs/MODELS.md`).

## Metrics

- **Validation Dice:** `<value>`
- **Test Dice / IoU (if a held-out set exists):** `<value>`
- **Reproducibility:** `<single-center? author-reported? independently checked?>`

State plainly whether these numbers are single-center and/or author-reported.

## Intended use

- `<what clinical/triage/research use this model is appropriate for>`
- `<who the intended users are>`

## Out-of-scope / prohibited use

- `<e.g. autonomous staging, unstageable/DTPI determination from RGB alone>`
- `<populations or imaging conditions it was not evaluated on>`

## Limitations

- `<failure modes: lighting, occlusion, dressings, rulers, multi-wound>`
- `<domain shift from training data to deployment cameras>`
- **Not a substitute for clinician judgment.** Report as decision support with
  uncertainty.

## Skin-tone evaluation

This section is **required**. Report performance stratified by skin tone
(e.g. Fitzpatrick I–II / III–IV / V–VI or Monk scale groups):

| Skin-tone group | N images | Dice | Notes |
|---|---|---|---|
| `<group>` | `<n>` | `<value>` | `<observations>` |

If a stratified evaluation was not performed, say so explicitly and treat the
model as unvalidated for fairness — do not deploy it clinically until this is
filled in.

## Provenance & verification

- Downloaded on request only; SHA-256 verified on arrival (fails closed).
- Format: `.onnx` (no pickle-based formats loaded).
- Registered in `models/registry.json` with the digest above.
