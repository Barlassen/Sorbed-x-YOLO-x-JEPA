# Longitudinal healing-trend tracking

Sorbed can compare several photos of the **same wound over time** and quantify
whether it is healing. Run:

```bash
sorbed compare visit1.jpg visit2.jpg visit3.jpg --days 0,14,28 --mm-per-px 0.12 \
    --patient "sacrum-01" --out reports
```

It analyzes each visit, then writes `trend_dashboard.png` (a visit filmstrip, a
wound-area line with a projected-closure line, tissue composition over time, and
the healing statistics) and `trend.json` (the machine-readable `HealingTrend`).

## What it computes

Each metric is a computation over the per-visit analyses, grounded in the
wound-healing literature:

| Metric | Definition | Basis |
|---|---|---|
| **Percent area reduction (PAR)** | `(A₀ − Aₜ) / A₀ × 100` | An early predictor of eventual healing. |
| **4-week PAR + "on track" flag** | PAR at ~28 days vs a threshold | Failure to reduce ~**40%** (venous, Kantor & Margolis 2000) to ~**50%** (diabetic foot, Sheehan 2003) by week 4 predicts non-healing. Sorbed flags the 40% line by default. |
| **Healing velocity** | Linear-fit area change per week | Simple rate; size-dependent. |
| **Edge advance (Gilman)** | `(A₁ − A₂) / meanPerimeter / Δt` → mm/day | Perimeter-normalizing makes the rate independent of wound size/shape — the biologically meaningful inward advance of the wound margin (Gilman 1990). |
| **Projected closure** | Linear extrapolation of area → 0 | A first-order estimate; the literature uses a **log-linear / delayed-exponential** fit (Cukjati 2001) for chronic wounds. Shown only when the trend is shrinking. |
| **PUSH trend** | Change in the (partial) PUSH total across visits | A falling PUSH total indicates healing (NPUAP/NPIAP). |
| **Trajectory** | healing / stalled / deteriorating | From the weekly percent-area rate. |

## Limitation: no public longitudinal data

There is no verified, public, human, per-patient *serial* wound-image dataset at
scale. The public wound datasets — AZH/FUSeg, the DFU challenges, Medetec,
CO2Wounds-V2, the Kaggle sets — are **cross-sectional** (many wounds, one image
each, no visit linkage or dates), even when the underlying images were collected
over years. The verified serial imaging datasets are **animal** (murine/porcine),
and the human healing-trajectory studies use **private registry data that is not
released**.

Consequences reflected in the design:

- Sorbed provides the **validated deterministic metrics** above. It does **not**
  include a learned "will-heal" predictor, because there is no public serial data
  to train or validate one. A labeled timeline must be built first.
- **Calibration and capture consistency across visits is the dominant error
  source.** Ruler length×width overestimates true area by ~44%; photo-planimetry
  is more accurate but still requires a perpendicular camera and a scale reference
  at *every* visit. Sorbed reports sizes in pixels (not cm²) unless every visit is
  calibrated, and every trend carries a caveat to this effect.

## Feature status

- **Implemented:** per-visit → timeline model (`WoundTimePoint` → `HealingTrend`),
  PAR, 4-week PAR flag, Gilman rate, PUSH trend, linear projection, the trend
  dashboard, and the `sorbed compare` command.
- **Not yet implemented:** log-linear / delayed-exponential closure projection
  with confidence bounds; per-etiology thresholds (40% venous / 50% diabetic-foot);
  DESIGN-R and BWAT trend lines; a persisted patient→wound→visit store with EXIF
  stripping and face/tattoo/text redaction for stored timelines; optional
  feature-based visit-to-visit registration for overlay/regional healing maps
  (kept approximate and clearly labeled); and an evaluated healing-trajectory model
  contingent on the availability of real serial data.

## Sources

- Sheehan et al., *Diabetes Care* 2003 (4-week PAR, diabetic foot): <https://pubmed.ncbi.nlm.nih.gov/12766127/>
- Kantor & Margolis, *Br J Dermatol* 2000 (4-week PAR, venous): <https://pubmed.ncbi.nlm.nih.gov/10809855/>
- Gwilym et al., *Adv Wound Care* 2023 (systematic review): <https://journals.sagepub.com/doi/abs/10.1089/wound.2021.0203>
- Gilman, perimeter-normalized healing rate: <https://pmc.ncbi.nlm.nih.gov/articles/PMC3013292/>
- Cukjati et al., *Med Biol Eng Comput* 2001 (trajectory models): <https://lbk.fe.uni-lj.si/pdfs/mbe2001a.pdf>
- PUSH Tool validation: <https://pubmed.ncbi.nlm.nih.gov/9362591/>
- DESIGN-R predictive validity: <https://pubmed.ncbi.nlm.nih.gov/22747950/>
- Planimetry measurement error: <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0134622>
- FUSeg / AZH dataset (cross-sectional): <https://github.com/uwm-bigdata/wound-segmentation>
- CO2Wounds-V2: <https://ieee-dataport.org/open-access/co2wounds-v2-extended-chronic-wounds-dataset-leprosy-patients-segmentation-and>

Author-reported thresholds are clinical conventions, not universal constants;
apply them per wound etiology and confirm with a clinician.
