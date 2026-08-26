# Skin-Tone Equity

Pressure-injury assessment has a well-documented equity problem, and any tool that reads a wound from color must account for it. This document states Sorbed's position, what the software does about it, and what contributors must do to maintain it.

> A low-stage, low-confidence, or abstained result on darker skin does **not** rule out injury. When Sorbed is uncertain on a darkly pigmented image, that uncertainty is a prompt to examine the patient more closely — not reassurance. See also [`../DISCLAIMER.md`](../DISCLAIMER.md) and the equity discussion in [`CLINICAL.md`](CLINICAL.md).

## Why detection is harder on darker skin

Two of the staging categories are defined largely by *color change*, and these are the two that surface RGB photography reads poorly on darkly pigmented skin:

- **Stage 1** is non-blanchable erythema of intact skin. In lightly pigmented skin this reads as redness. In darkly pigmented skin it often does not read as red — it presents as a subtle deviation from the surrounding tissue: hyperpigmentation, or a purplish-to-bluish hue that is easy to miss against a higher baseline melanin signal.
- **Deep Tissue Pressure Injury (DTPI)** is persistent deep red, maroon, or purple discoloration, or a blood-filled blister, over intact skin. Its color signature blends into darker baseline pigmentation, and its severity lies *beneath* an intact surface where a camera cannot follow.

The clinical literature has repeatedly found that early-stage pressure injuries are underdetected in patients with darker skin tones, contributing to later-stage presentations at first diagnosis. The National Pressure Injury Advisory Panel has convened dedicated state-of-the-science work on pressure-injury assessment in darkly pigmented skin, emphasizing that visual inspection alone is insufficient and that assessment must incorporate skin temperature, firmness/induration, and edema — signals no photograph carries. Consult NPIAP's published dark-skin assessment guidance directly for the authoritative wording; it governs over this summary.

An automated color analyzer does not escape this constraint. Left uncorrected, it inherits and can amplify the same bias, because the color contrast it keys on is genuinely weaker in the pixels.

## What Sorbed does about it

Sorbed's design is to be measurably more cautious and explicit on darker skin:

1. **ITA-based skin-tone estimation.** The pipeline estimates an Individual Typology Angle from the periwound/intact-skin pixels and maps it to a coarse band (`fitzpatrick_i_iii`, `fitzpatrick_iv_vi`, or `unknown`), recorded on every analysis as `skin_tone_band`. The band is deliberately coarse — a modulator of caution, not a demographic claim about the patient.
2. **Lowered confidence and an explicit warning.** On the darker band, the staging engine lowers its confidence and attaches an explicit caveat to the decision. The same color evidence that would read as adequate on lighter skin is treated as weaker.
3. **Abstention over a false negative.** For the color-driven categories — Stage 1 and DTPI in particular — Sorbed returns **Indeterminate** and defers to a clinician rather than issue a confident low-stage result it cannot support. The design prioritizes avoiding under-calling injury on dark skin.
4. **Pointing at what the camera cannot see.** The caveats and narrative direct the clinician to assess **temperature, firmness/induration, and edema** at the bedside — the non-visual signs that carry the diagnosis when color does not.

None of this "corrects" for skin tone in a way that could manufacture false confidence. The system becomes more cautious on darker skin, not more assertive.

## What contributors must do

Equity here is a maintenance responsibility, not a one-time feature. Changes to segmentation, tissue classification, color cues, the skin-tone estimator, or the staging/confidence logic must:

- **Evaluate performance stratified by skin tone.** Do not report a single aggregate metric. Break out segmentation and staging performance by skin-tone band. A model that performs well on average but degrades on the darker band is not shippable. Aggregate numbers hide the failure mode that matters here.
- **Report the stratified numbers in the PR.** Make the per-band breakdown visible in review. A change that improves the mean while widening the gap must be justified explicitly or rejected.
- **Do not remove or weaken the dark-skin caveat or the abstention paths** without an equity review. These are safety behaviors, not tunable defaults.
- **Prefer abstention to a confident false negative.** On a darkly pigmented, color-ambiguous case, the correct behavior is to defer to the clinician.
- **Seek diverse evaluation data.** The field's public datasets skew toward lighter skin; state that limitation explicitly (see [`MODELS.md`](MODELS.md)) rather than letting it silently bias validation. Do not represent the data as more representative than it is.

## Summary

The physics of RGB imaging limits how reliably dark-skin pressure injuries can be detected from color. Sorbed's approach is to make that difficulty explicit rather than hide it behind a confident number: it flags the harder cases, lowers its confidence, abstains when the evidence is thin, and points the clinician at the temperature, firmness, and edema that a photograph does not show.
