"""Tissue detection, A vs B, on the pressure-injury tissue ruler (ComplexWoundDB, 27 images x 4 experts).

A  unsupervised (mentor Ario's idea, colour arm): per image k-means on Lab colour inside the wound,
   each cluster named by its colour (dark+grey -> necrosis, pale+grey -> other, else hue: red ->
   granulation, yellow -> slough), then a continuity rule (small islands absorbed by their
   surroundings). ``A-raw`` is the same without the continuity rule, to test that rule on its own.
B  supervised: per-pixel random forest on colour/texture/distance-to-border features, trained on
   DFUTissue TrainVal (94 foot ulcers: fibrin / granulation / callus).

Anti-cheating: A's one free parameter (the red/yellow hue boundary) is tuned on DFUTissue TrainVal,
B is trained there too; the 27 ComplexWoundDB images are touched once, for the final score.

Scores are pooled Dice per tissue (granulation G, slough S, necrosis N), against each of the 4
experts and averaged; 95% CIs by bootstrapping images. Two settings:
  R1  inside the experts' consensus wound region  -> pure tissue quality
  R2  inside OUR segmenter's mask                 -> the full chain as Sorbed would run it
Reference rows: human ceiling (expert vs expert, R1) and a trivial "everything is granulation".

    python -m training.tissue_ab_compare --out runs/tissue_ab
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

DFUT = Path("data/corpus/dfutissue/DFUTissue/Labeled/Original")
CWDB = Path("data/corpus/complexwounddb/ComplexWoundDB")
SEG_OLD = "runs/segment/runs_wound/yolo26n_fuseg/weights/best.pt"
SEG_NEW = "runs/segment/runs/segment/combined_v1/weights/best.pt"

BG, G, S, N, O = 0, 1, 2, 3, 4  # background, granulation, slough/fibrin, necrosis, other
CLASSES = {"G": G, "S": S, "N": N}
SIDE = 160  # every wound crop is rescaled so its longer side is SIDE px (DFUTissue crops are ~100-200 px)

# ComplexWoundDB colour code (RGB); anti-aliased edges are snapped to the nearest entry.
_CW_PALETTE = np.array([(255, 255, 255), (237, 28, 36), (255, 242, 0), (0, 0, 0), (163, 73, 164), (163, 35, 142)], np.float32)
_CW_LABEL = np.array([BG, G, S, N, O, O])
_DFUT_LABEL = np.array([BG, S, G, O])  # 0 bg, 1 fibrin, 2 granulation, 3 callus


# ---------------------------------------------------------------- data
def cw_expert_labels(name: str) -> list[np.ndarray]:
    out = []
    for e in sorted((CWDB / "annotations/masks").iterdir()):
        bgr = cv2.imread(str(e / name))
        if bgr is None:
            continue
        rgb = bgr[..., ::-1].reshape(-1, 1, 3).astype(np.float32)
        idx = np.argmin(((rgb - _CW_PALETTE[None]) ** 2).sum(-1), axis=1)
        out.append(_CW_LABEL[idx].reshape(bgr.shape[:2]))
    return out


def crop_resize(img, region, *others):
    """Crop to the region's padded bbox and rescale (longer side SIDE). Returns crop, region, others, box."""
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int(0.05 * (y1 - y0)) + 1, int(0.05 * (x1 - x0)) + 1
    y0, y1 = max(0, y0 - py), min(img.shape[0], y1 + py)
    x0, x1 = max(0, x0 - px), min(img.shape[1], x1 + px)
    h, w = y1 - y0, x1 - x0
    f = SIDE / max(h, w)
    size = (max(1, round(w * f)), max(1, round(h * f)))
    rz = lambda a, interp: cv2.resize(a[y0:y1, x0:x1], size, interpolation=interp)  # noqa: E731
    crop = rz(img, cv2.INTER_AREA)
    reg = rz(region.astype(np.uint8), cv2.INTER_NEAREST) > 0
    return crop, reg, [rz(o.astype(np.uint8), cv2.INTER_NEAREST) for o in others], (y0, y1, x0, x1)


def paste(label_crop, box, shape):
    y0, y1, x0, x1 = box
    full = np.zeros(shape, np.uint8)
    full[y0:y1, x0:x1] = cv2.resize(label_crop.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return full


def lab_std(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.dstack([lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128])


# ---------------------------------------------------------------- A: unsupervised colour + continuity
def name_cluster(L, a, b, hue0):
    chroma, hue = np.hypot(a, b), np.degrees(np.arctan2(b, a))
    if L < 30 and chroma < 20:
        return N
    if L > 60 and chroma < 12:
        return O
    return S if hue > hue0 else G


def continuity(lbl, region, frac=0.03):
    """Ario's continuity rule, light version: islands < frac of the wound join their surroundings."""
    lbl = lbl.copy()
    min_px = max(4, int(frac * region.sum()))
    for _ in range(2):
        for c in (G, S, N, O):
            n, comp, stats, _ = cv2.connectedComponentsWithStats((lbl == c).astype(np.uint8), connectivity=8)
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] >= min_px:
                    continue
                isl = comp == k
                ring = (cv2.dilate(isl.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~isl & region
                vals = lbl[ring]
                vals = vals[vals != c]
                if len(vals):
                    lbl[isl] = np.bincount(vals).argmax()
    lbl[~region] = BG
    return lbl


def method_a(bgr, region, hue0, clean=True, k=4, seed=0):
    lab = lab_std(bgr)
    px = lab[region]
    lbl = np.zeros(region.shape, np.uint8)
    if len(px) < k * 5:
        return lbl
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(px[:: max(1, len(px) // 4000)])
    names = np.array([name_cluster(*c, hue0) for c in km.cluster_centers_])
    lbl[region] = names[km.predict(px)]
    return continuity(lbl, region) if clean else lbl


# ---------------------------------------------------------------- B: supervised pixel forest
def features(bgr, region):
    lab = lab_std(bgr)
    f = [lab]
    for s in (1.5, 4.0):
        f.append(cv2.GaussianBlur(lab, (0, 0), s))
    L = lab[..., 0]
    m, m2 = cv2.GaussianBlur(L, (0, 0), 3), cv2.GaussianBlur(L * L, (0, 0), 3)
    f.append(np.sqrt(np.clip(m2 - m * m, 0, None))[..., None])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    f.append(hsv[..., 1:])
    gx, gy = cv2.Sobel(cv2.GaussianBlur(L, (0, 0), 1.5), cv2.CV_32F, 1, 0), cv2.Sobel(cv2.GaussianBlur(L, (0, 0), 1.5), cv2.CV_32F, 0, 1)
    f.append(np.hypot(gx, gy)[..., None])
    dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    f.append((dist / (dist.max() + 1e-6))[..., None])  # 0 at the border, 1 at the centre: layer position
    return np.dstack(f)


def dfut_items(split):
    for p in sorted((DFUT / "Images" / split).glob("*.png")):
        ann = cv2.imread(str(DFUT / "Annotations" / split / p.name), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(p))
        if ann is None or img is None or not (ann > 0).any():
            continue
        yield p.stem, img, _DFUT_LABEL[ann]


def train_b(seed=0, per_img=4000):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _, img, gt in dfut_items("TrainVal"):
        crop, reg, (g,), _ = crop_resize(img, gt > 0, gt)
        F = features(crop, reg)[reg]
        t = g[reg]
        idx = rng.choice(len(t), min(per_img, len(t)), replace=False)
        X.append(F[idx]), y.append(t[idx])
    rf = RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=5, class_weight="balanced",
                                n_jobs=-1, random_state=seed)
    return rf.fit(np.concatenate(X), np.concatenate(y))


def method_b(rf, bgr, region):
    lbl = np.zeros(region.shape, np.uint8)
    if region.sum():
        lbl[region] = rf.predict(features(bgr, region)[region])
    return lbl


# ---------------------------------------------------------------- scoring
def inter_sums(pred, gt, region):
    """[class] -> (intersection, |pred|+|gt|) restricted to region, for G/S/N."""
    out = np.zeros((3, 2))
    for i, c in enumerate(CLASSES.values()):
        p, g = (pred == c) & region, (gt == c) & region
        out[i] = (p & g).sum(), p.sum() + g.sum()
    return out


def pooled(arr):
    """arr [img, expert, class, 2] -> Dice per class (pooled over images, mean over experts)."""
    s = arr.sum(0)
    d = 2 * s[..., 0] / np.maximum(s[..., 1], 1)
    d[s[..., 1] == 0] = np.nan
    return np.nanmean(d, 0)


def boot(arr, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([pooled(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n)])


def fmt(point, bs):
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return {k: (None if np.isnan(point[i]) else f"{point[i]:.3f} [{lo[i]:.3f}, {hi[i]:.3f}]") for i, k in enumerate(CLASSES)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/tissue_ab")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    # 1) tune A's hue boundary on DFUTissue TrainVal (mean of fibrin & granulation Dice)
    train = [(crop_resize(img, gt > 0, gt)) for _, img, gt in dfut_items("TrainVal")]
    best = None
    for hue0 in range(30, 82, 4):
        acc = np.zeros((3, 2))
        for crop, reg, (g,), _ in train:
            acc += inter_sums(method_a(crop, reg, hue0), g, reg)
        d = 2 * acc[:2, 0] / np.maximum(acc[:2, 1], 1)
        if best is None or d.mean() > best[1]:
            best = (hue0, d.mean())
    hue0 = best[0]
    report["A_hue_boundary_deg"] = hue0
    print(f"A: hue boundary tuned on DFUTissue TrainVal = {hue0} deg (G/S mean Dice {best[1]:.3f})")

    rf = train_b()
    print("B: random forest trained on DFUTissue TrainVal")

    methods = {
        "A": lambda im, r: method_a(im, r, hue0),
        "A-raw": lambda im, r: method_a(im, r, hue0, clean=False),
        "B": lambda im, r: method_b(rf, im, r),
        "trivial(all G)": lambda im, r: np.where(r, G, BG).astype(np.uint8),
    }

    # 2) in-domain sanity: DFUTissue Test (16), wound region from the annotation
    dt = {m: [] for m in methods}
    for _, img, gt in dfut_items("Test"):
        crop, reg, (g,), _ = crop_resize(img, gt > 0, gt)
        for m, f in methods.items():
            dt[m].append(inter_sums(f(crop, reg), g, reg)[None])
    report["dfut_test_16"] = {m: fmt(pooled(np.array(v)), boot(np.array(v), args.boot)) for m, v in dt.items()}

    # 3) the ruler: ComplexWoundDB 27 x 4 experts, R1 (consensus region) and R2 (our segmenter)
    from ultralytics import YOLO

    segs = (YOLO(SEG_OLD), YOLO(SEG_NEW))

    def our_mask(bgr):
        for m in segs:
            r = m.predict(bgr, conf=0.01, verbose=False, retina_masks=True)[0]
            if r.masks is not None and len(r.masks.data):
                u = r.masks.data.cpu().numpy().max(0)
                if u.shape != bgr.shape[:2]:
                    u = cv2.resize(u.astype(np.float32), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                return u > 0.5
        return np.zeros(bgr.shape[:2], bool)

    R = {s: {m: [] for m in methods} for s in ("R1", "R2")}
    human = []
    for p in sorted((CWDB / "images").glob("*.png"), key=lambda q: int(q.stem)):
        img = cv2.imread(str(p))
        experts = [e for e in cw_expert_labels(p.name) if e.shape == img.shape[:2]]
        if len(experts) < 2:
            continue
        consensus = np.sum([e != BG for e in experts], 0) >= 3
        full = np.ones(img.shape[:2], bool)
        hrow = [inter_sums(experts[j], experts[i], consensus) for i in range(len(experts)) for j in range(len(experts)) if i != j]
        human.append(np.array(hrow))
        for setting, region in (("R1", consensus), ("R2", our_mask(img))):
            for m, f in methods.items():
                if region.sum() < 30:
                    pred = np.zeros(img.shape[:2], np.uint8)
                else:
                    crop, reg, _, box = crop_resize(img, region)
                    pred = paste(f(crop, reg), box, img.shape[:2])
                scope = consensus if setting == "R1" else full
                R[setting][m].append(np.array([inter_sums(pred, e, scope) for e in experts]))

    for setting in R:
        report[f"cwdb_{setting}"] = {}
        for m, v in R[setting].items():
            arr = np.array(v)  # [img, expert, class, 2]
            report[f"cwdb_{setting}"][m] = fmt(pooled(arr), boot(arr, args.boot))
    harr = np.array(human)  # [img, pair, class, 2]
    report["cwdb_R1"]["HUMAN (expert vs expert)"] = fmt(pooled(harr), boot(harr, args.boot))

    # paired A - B difference on the main score (mean of G and S), R1 and R2
    for setting in R:
        a, b = np.array(R[setting]["A"]), np.array(R[setting]["B"])
        rng = np.random.default_rng(1)
        diffs = []
        for _ in range(args.boot):
            idx = rng.integers(0, len(a), len(a))
            diffs.append(np.nanmean(pooled(a[idx])[:2]) - np.nanmean(pooled(b[idx])[:2]))
        pa, pb = np.nanmean(pooled(a)[:2]), np.nanmean(pooled(b)[:2])
        diffs = np.array(diffs)
        report[f"A_minus_B_GS_{setting}"] = {
            "A": round(float(pa), 3), "B": round(float(pb), 3), "diff": round(float(pa - pb), 3),
            "ci95": [round(float(x), 3) for x in np.percentile(diffs, [2.5, 97.5])],
            "P(A>B)": round(float((diffs > 0).mean()), 3),
        }

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
