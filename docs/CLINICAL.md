# Clinical Basis & Grading Logic

This document describes the clinical reasoning that Sorbed encodes. It serves two audiences: clinicians assessing whether the system respects established staging doctrine, and developers who need an unambiguous specification of the rules the software implements. Where the two might diverge, the clinical meaning takes precedence; the code implements the guideline.

Sorbed is a decision-support tool. It does not diagnose and is designed to defer to a clinician on cases that RGB photography cannot resolve. The sections below describe how the system stages a pressure injury, what tissue it recognizes, what it measures, how those measurements feed the standard monitoring scales, and the deterministic rule layer that turns machine perception into a grade.

## Staging systems and terminology

Sorbed follows the National Pressure Injury Advisory Panel (NPIAP) 2016 staging system as carried forward in the 2019 EPUAP–NPIAP–PPPIA International Guideline. Two vocabularies coexist in the literature, and the software supports both:

- **NPIAP** uses the term *pressure injury* with Arabic numerals (Stage 1 through Stage 4).
- **EPUAP** uses the term *pressure ulcer* with the word *Category* (Category I through IV).

The naming differs; the underlying tissue criteria are identical. A Stage 3 pressure injury and a Category III pressure ulcer describe the same wound. One principle governs the whole scheme and is enforced throughout Sorbed: **a wound is never back-staged.** A healing Stage 4 injury does not become a Stage 2 as it fills with granulation tissue — it remains a "healing Stage 4." Staging records the greatest anatomical depth of tissue loss ever reached, not the wound's current appearance on its way to closure.

### Stage 1 — Non-blanchable erythema of intact skin

Intact skin with localized, **non-blanchable** erythema. In darkly pigmented skin the presentation may not read as redness; it appears instead as a deviation in color from the surrounding tissue — hyperpigmentation, or a purplish-to-bluish hue. Deep red, maroon, or purple discoloration is *not* Stage 1; that presentation indicates deep tissue injury. Stage 1 may be accompanied by changes in temperature, firmness, or edema before any color change is visible.

Observable characteristics: an intact epidermis, a localized color change, and the absence of slough, eschar, granulation, or any exposed subcutaneous tissue.

### Stage 2 — Partial-thickness skin loss with exposed dermis

Partial-thickness loss of skin exposing the dermis. The wound bed is viable, pink or red, and **moist**. This stage also covers an intact or ruptured **serum-filled** blister. No adipose (fat) is visible, and there is no granulation tissue, slough, or eschar. Stage 2 does not describe moisture-associated skin damage (MASD/IAD), skin tears, or burns, all of which have distinct etiologies even when they look superficially similar.

### Stage 3 — Full-thickness skin loss

Full-thickness loss in which **adipose (fat) is visible** in the wound bed. Granulation tissue and epibole (rolled wound edges) are often present, and slough or eschar may partially obscure the bed. No deeper structure is present: no fascia, muscle, tendon, ligament, cartilage, or bone. The visibility of fat is the defining threshold that separates Stage 3 from Stage 2.

### Stage 4 — Full-thickness skin and tissue loss

Full-thickness loss with **exposed or directly palpable fascia, muscle, tendon, ligament, cartilage, or bone.** Epibole, undermining, and tunneling are common. The presence of any of these structural tissues distinguishes Stage 4 from Stage 3.

### Unstageable — Obscured full-thickness loss

A full-thickness injury whose base is **obscured by slough or eschar** to the point that the true extent of tissue loss cannot be confirmed. It is a hidden Stage 3 or Stage 4; once the bed is cleaned or debrided and the depth becomes visible, it is restaged accordingly. One clinical caution is encoded here: **stable eschar** — dry, adherent, intact, with no surrounding erythema or fluctuance — on an ischemic limb or heel serves as the body's natural cover and should **not** be debrided.

### Deep Tissue Pressure Injury (DTPI)

Persistent, non-blanchable **deep red, maroon, or purple** discoloration, or epidermal separation revealing a dark wound bed or a **blood-filled** blister. DTPI can evolve rapidly to reveal the true extent of injury, or it can resolve without tissue loss. Along with Stage 1, DTPI is among the hardest presentations to detect in darkly pigmented skin, where the discoloration blends with baseline pigmentation. Because its trajectory is uncertain and its surface appearance understates the underlying damage, DTPI is one of the cases Sorbed routes to a clinician rather than committing to a numeric stage.

### Special situations

- **Medical Device Related Pressure Injury (MDRPI):** results from a device (tubing, mask, brace) and typically **conforms to the shape of the device**. It is staged on the ordinary 1–4 / Unstageable / DTPI scale; the device relationship is a flag, not a separate stage.
- **Mucosal Membrane Pressure Injury:** found on mucous membranes with a history of a device at that location. Because mucosal tissue lacks the layered structure the staging system depends on, these injuries **cannot be staged.**

## Tissue types and their appearance

Staging in Sorbed is driven by *which tissues are present*, so the tissue vocabulary is foundational. Each type has a characteristic color and texture that the perception layer is trained to recognize:

| Tissue | Appearance | Staging significance |
|---|---|---|
| Epithelial | Pale pink or pearly, matte; migrating edge or islands across the bed | Sign of resurfacing/closure |
| Granulation (healthy) | Beefy, bright red, moist, cobblestone texture | Healthy repair tissue |
| Granulation (unhealthy) | Pale, dusky, or dark dull red; friable | Poor perfusion / stalled healing |
| Slough | Yellow, tan, gray, green, or brown; soft, stringy, adherent | Devitalized; can obscure bed |
| Eschar | Black or brown; dry, hard, leathery | Devitalized; can obscure bed |
| Hypergranulation | Raised **above** the wound margin; friable | Overgrowth impeding closure |
| Exposed fat (adipose) | Pale yellow, globular | Establishes **Stage 3** |
| Exposed muscle | Dark red, striated | Establishes **Stage 4** |
| Tendon | Yellow-white, shiny | Establishes **Stage 4** |
| Bone | White or tan, hard | Establishes **Stage 4** |

The older **Red-Yellow-Black (RYB)** system is supported as a legacy proxy: red maps to granulation, yellow to slough, and black to eschar. It is a coarse summary and less expressive than the full inventory above, and Sorbed treats it as a derived view rather than the primary classification.

The key point for developers is that **stage gating is tissue-presence driven, not percentage driven.** Visible fat forces the stage to at least 3; any visible structural tissue forces Stage 4; a bed obscured by slough or eschar forces Unstageable; and intact skin with maroon or purple discoloration indicates DTPI. Tissue *percentages* do not change the stage — they feed the monitoring scales described below, which track healing over time rather than establishing depth.

## Metrics and measurement conventions

Sorbed records wound geometry using conventions that are standard in wound care and defined precisely enough to compute reproducibly:

- **Length** is the greatest **head-to-toe** dimension. Sorbed uses a clock-face convention in which 12:00 points toward the patient's head.
- **Width** is the greatest **side-to-side** dimension measured **perpendicular** to the length.
- **Area** is approximated as length × width in cm². True planimetry (tracing the wound outline) is more accurate but requires a scale fiducial in the image to convert pixels to centimeters.
- **Depth** is measured by probing to the deepest point. It is **not derivable from a 2D RGB photograph** — recovering it requires 3D capture such as stereo photography, structured light, or LiDAR. When no such data is present, depth is left unknown rather than estimated.
- **Undermining** and **tunneling** are recorded as a depth plus a clock position. Neither is visible on the wound surface, so both are user-entered.
- **Wound edges** are characterized as attached, epibole (rolled), or undermined.
- **Periwound skin** (within roughly 4 cm of the margin) is assessed for maceration (white, soggy), erythema, induration, edema, and callus.
- **Exudate** is recorded by amount (none / scant / small / moderate / large) and by type (serous / sanguineous / serosanguineous / purulent).

The clock convention anchors every positional measurement: **12:00 is toward the patient's head.**

## Monitoring scales

Staging captures depth; it does not capture healing. For longitudinal tracking, Sorbed computes the established wound-monitoring scales and specifies which sub-items an image can support and which require a clinician at the bedside.

### PUSH Tool 3.0

The Pressure Ulcer Scale for Healing produces a total from 0 to 17, where 0 is a healed wound and lower is better. It has three subscores:

- **Size (0–10)** by area band: 0 = 0 cm²; 1 = <0.3; 2 = 0.3–0.6; 3 = 0.7–1.0; 4 = 1.1–2.0; 5 = 2.1–3.0; 6 = 3.1–4.0; 7 = 4.1–8.0; 8 = 8.1–12.0; 9 = 12.1–24.0; 10 = >24.0.
- **Exudate amount (0–3).**
- **Tissue type (0–4)** by the **worst** tissue present: 0 = closed, 1 = epithelial, 2 = granulation, 3 = slough, 4 = necrotic/eschar.

Size and tissue type are computable from an image; exudate is confirmed by the clinician.

### BWAT (Bates-Jensen Wound Assessment Tool)

Formerly the Pressure Sore Status Tool (PSST). BWAT scores **13 items on a 1–5 scale** (total 13–65, higher is worse) plus two purely descriptive items (location and shape). The scored items are: size, depth, edges, undermining, necrotic tissue type, necrotic tissue amount, exudate type, exudate amount, periwound skin color, peripheral edema, peripheral induration, granulation, and epithelialization.

Computable from an image: size, edges, necrotic tissue type and amount, periwound skin color, granulation, and epithelialization. User-entered: depth, undermining, peripheral edema, peripheral induration, and exudate.

### Sessing Scale

Introduced by Ferrell in 1995, the Sessing Scale is a 7-point ordinal measure (0–6) that monitors healing by capturing granulation, infection, necrosis, and eschar, independent of size and depth. It is partly computable — the tissue-appearance components can be estimated — but infection and odor are not observable in a photograph. The per-level anchor text should be taken from the original publication before implementing the scoring thresholds.

### DESIGN-R®2020

Developed by the Japanese Society of Pressure Ulcers (JSPU), DESIGN-R®2020 sums seven items to a total of 0–66 and classifies severity as slight (≤9), moderate (10–18), or severe (≥19). The items are Depth (the 2020 revision added a "DTI suspicion" option), Exudate, Size (length × width), Inflammation/infection (2020 added a "3C" option for suspected critical colonization), Granulation (percentage of healthy granulation), Necrotic tissue, and Pocket (undermining).

Computable: size, granulation percentage, and necrotic tissue. User-entered: depth, pocket, exudate, and inflammation/infection.

## Decision logic

The scales and tissue inventory feed a deterministic rule layer that sits *on top of* the machine-learning perception. The ML models answer perceptual questions — is the skin intact, what tissue is in the bed, what is the dominant color — and this rule layer turns those answers into a stage using explicit, auditable logic. The same inputs always produce the same stage.

```
IF mucosal location → "Mucosal – not stageable"
IF device-shaped lesion → flag MDRPI, then continue staging
IF skin INTACT (or blister only):
   IF color ∈ {deep red, maroon, purple} OR blood-filled blister OR dark bed → DTPI
   ELIF non-blanchable erythema/color change, no break → Stage 1
   ELSE (serum blister / shallow open, moist pink-red bed, NO fat, NO slough/eschar, NO granulation) → Stage 2
IF skin BROKEN and full-thickness:
   IF base OBSCURED by slough/eschar → Unstageable
   ELIF fascia/muscle/tendon/ligament/cartilage/bone visible or palpable → Stage 4
   ELIF adipose (fat) visible → Stage 3
```

The ordering is significant. Mucosal and device checks come first because they change how everything downstream is interpreted. Within intact skin, DTPI is tested before Stage 1 so that maroon and purple discoloration is not misfiled as ordinary erythema. Within broken skin, the obscured-bed check precedes the depth checks because a depth that cannot be seen cannot be asserted.

## Explainability payload

Each Sorbed result carries an evidence trace that can be inspected and, where necessary, overruled:

- a **skin integrity** flag (intact vs. broken);
- a **tissue inventory** with per-type percentage and confidence, plus an inspectable overlay showing where each tissue was detected;
- the **dominant color evidence** behind DTPI and Stage 1 decisions;
- the **depth cue source** — whether depth was measured or is unknown, and whether any structural tissue was visible;
- **edge and periwound findings**;
- the computed **metrics**, including the PUSH, BWAT, and DESIGN-R subscores.

The system reports **calibrated confidence**. Sorbed reports how sure it is, and it **abstains — deferring to the clinician — on low-confidence results and on the DTPI and Unstageable categories**, which are, in principle, not fully resolvable from a single surface photograph.

## Sources

The clinical content above is drawn from the following bodies. Primary PDFs should be obtained directly from these sources for the verbatim scoring anchors, which govern in any case of discrepancy with this summary.

- [NPIAP — pressure injury staging system](https://npiap.com/)
- [EPUAP — pressure ulcer classification](https://www.epuap.org/)
- [Prevention and Treatment of Pressure Ulcers/Injuries: Clinical Practice Guideline (2019 International Guideline)](https://internationalguideline.com/)
- PUSH Tool 3.0 (NPUAP/NPIAP) — obtain the official tool document directly for the size and tissue-type anchors
- BWAT / Bates-Jensen Wound Assessment Tool — obtain the original instrument for the 1–5 item anchors
- DESIGN-R®2020 (Japanese Society of Pressure Ulcers, JSPU) — obtain the official scoring sheet for item definitions

Note: the per-level anchor text for the Sessing Scale and the exact wording of several BWAT and DESIGN-R items must be verified against the primary publications before the corresponding scoring thresholds are finalized in code.
