"""
auto_slot_finder.py - derive parking-slot quads from the painted stall lines.

Pipeline
  1. Isolate thin white paint (top-hat filter), split into horizontal / vertical masks.
  2. Find stall blocks (x-ranges where horizontal paint exists); split double-loaded
     blocks at the vertical divider.
  3. Per block, fit a uniform stall lattice (pitch + phase) to the paint, by scoring a
     comb against the row profile. Cars occlude paint, the lattice bridges the gaps.
  4. Trim bands that fall on concrete islands, estimate the block's tilt, emit quads.

Only needs numpy + opencv. Output format is unchanged:
  {"P1": {"spot_id": "P1", "polygon_norm": [[x, y] x 4]}}   # TL, TR, BR, BL, normalized
"""
import argparse
import json
import os

import cv2
import numpy as np


# ----------------------------------------------------------------------------- masks
def paint_masks(img, thresh=28):
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    def mask(th_k, open_k):
        k = lambda s: cv2.getStructuringElement(cv2.MORPH_RECT, s)
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k(th_k))   # keeps features thinner than th_k
        _, bw = cv2.threshold(th, thresh, 255, cv2.THRESH_BINARY)
        return cv2.morphologyEx(bw, cv2.MORPH_OPEN, k(open_k))   # keeps only long strokes

    t = int(h * 0.012) | 1                                       # > paint thickness, < car size
    hmask = mask((1, t), (max(15, int(w * 0.018)), 1))           # horizontal stall dividers
    vmask = mask((t, 1), (1, max(15, int(h * 0.042))))           # vertical column dividers
    return hmask, vmask


def island_mask(img):
    """Tan concrete islands / curbs: H~17, S~35-55, V~180-210 (asphalt V<135, white cars S<20)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (10, 28, 160), (28, 75, 245))
    return cv2.medianBlur(m, 9)


# ----------------------------------------------------------------------------- columns
def find_columns(hmask, vmask, w, h):
    prof = hmask.sum(0).astype(np.float32) / 255
    sig = max(1, int(w * 0.004))
    g = np.exp(-0.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2)
    prof = np.convolve(prof, g / g.sum(), mode="same")
    on = prof > max(3.0, 0.15 * prof.max())

    # contiguous runs, merging small gaps
    runs, start = [], None
    for x, v in enumerate(on):
        if v and start is None:
            start = x
        if not v and start is not None:
            runs.append([start, x - 1]); start = None
    if start is not None:
        runs.append([start, w - 1])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < w * 0.012:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    cols = []
    for a, b in merged:
        if b - a < w * 0.06:
            continue
        xs = np.nonzero(hmask[:, a:b + 1])[1] + a
        lo, hi = int(np.percentile(xs, 1)), int(np.percentile(xs, 99))
        # double-loaded block: split at the vertical divider if there is one
        if hi - lo > w * 0.12:
            vp = vmask[:, lo:hi].sum(0).astype(np.float32) / 255
            m0, m1 = int(0.3 * (hi - lo)), int(0.7 * (hi - lo))
            if vp[m0:m1].max() > 0.05 * h:
                mid = lo + m0 + int(np.argmax(vp[m0:m1]))
                cols += [(lo, mid), (mid, hi)]
                continue
        cols.append((lo, hi))
    return cols


# ----------------------------------------------------------------------------- lattice
def fit_lattice(hmask, x0, x1, h):
    """Return (pitch, y0, rms, n_supported) for lines y_k = y0 + k*pitch."""
    prof = hmask[:, x0:x1].sum(1).astype(np.float32) / 255
    g = np.exp(-0.5 * (np.arange(-5, 6) / 1.5) ** 2)
    s = np.convolve(prof, g / g.sum(), mode="same")

    d = s - s.mean()
    ac = np.correlate(d, d, "full")[len(s) - 1:]
    ac = ac / (len(s) - np.arange(len(s)))                         # unbiased
    lo, hi = int(0.04 * h), int(0.10 * h)
    p0 = lo + int(np.argmax(ac[lo:hi]))

    best = (-1.0, p0, 0.0)                                         # comb search
    for p in np.arange(0.94 * p0, 1.06 * p0, 0.25):
        for ph in np.arange(0, p, 1.0):
            ys = np.clip(np.round(np.arange(ph, h, p)).astype(int), 0, h - 1)
            sc = s[ys].sum()
            if sc > best[0]:
                best = (sc, p, ph)
    _, p, ph = best

    # snap teeth to real paint, then least-squares refine (pitch + phase)
    thr = 0.25 * s.max()
    for _ in range(3):
        ks, ys = [], []
        for k, y in enumerate(np.arange(ph, h, p)):
            a, b = int(max(0, y - 0.15 * p)), int(min(h, y + 0.15 * p)) + 1
            if b - a < 3:
                continue
            j = a + int(np.argmax(s[a:b]))
            if s[j] >= thr:
                ks.append(k); ys.append(j)
        if len(ks) < 3:
            break
        A = np.vstack([np.ones(len(ks)), ks]).T
        (ph, p), *_ = np.linalg.lstsq(A, np.array(ys, float), rcond=None)
        res = np.array(ys) - (ph + p * np.array(ks))
        keep = np.abs(res) < 0.06 * p
        if keep.sum() >= 3 and not keep.all():
            A = A[keep]
            (ph, p), *_ = np.linalg.lstsq(A, np.array(ys, float)[keep], rcond=None)
    ph = ph % p
    ks = np.arange(0, int((h - ph) // p) + 1)
    return float(p), float(ph), float(np.sqrt(np.mean(res[keep] ** 2))) if len(ks) else 0.0, len(ys)


def block_tilt(hmask, x0, x1, ys, p):
    slopes = []
    for y in ys:
        a, b = int(max(0, y - 0.15 * p)), int(y + 0.15 * p)
        yy, xx = np.nonzero(hmask[a:b, x0:x1])
        if len(xx) > 0.35 * (x1 - x0) and np.ptp(xx) > 0.5 * (x1 - x0):
            slopes.append(np.polyfit(xx, yy + a, 1)[0])
    return float(np.median(slopes)) if slopes else 0.0


def usable_bands(isl, x0, x1, p, y0, h):
    """Bands inside the image that are not concrete island; keep the longest contiguous run."""
    ys = np.arange(y0, h + p, p)
    flags = []
    for ya, yb in zip(ys[:-1], ys[1:]):
        if ya < -0.15 * p or yb > h + 0.15 * p:
            flags.append(False); continue
        a, b = int(max(0, ya)), int(min(h, yb))
        flags.append(isl[a:b, x0:x1].mean() / 255 < 0.45)
    best, cur, start, bs = (0, 0), 0, 0, 0
    for i, f in enumerate(flags + [False]):
        if f:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur > best[0]:
                best, bs = (cur, i), start
            cur = 0
    return ys, range(bs, bs + best[0])



def report_overlaps(registry, w, h):
    """Print any pair of slots whose polygons overlap by more than 1% of the smaller slot."""
    import itertools
    polys = {k: (np.array(v["polygon_norm"]) * [w, h]).astype(np.float32) for k, v in registry.items()}
    bad = []
    for (a, pa), (b, pb) in itertools.combinations(polys.items(), 2):
        if abs(pa[:, 0].mean() - pb[:, 0].mean()) > 0.25 * w:      # far apart, skip
            continue
        area, _ = cv2.intersectConvexConvex(pa, pb)
        if area > 0.01 * min(cv2.contourArea(pa), cv2.contourArea(pb)):
            bad.append((a, b))
    print("Overlap check: " + ("none - all slots are disjoint" if not bad else f"{len(bad)} overlapping pairs: {bad[:6]}"))
    return bad

# ----------------------------------------------------------------------------- main
def find_slots_automatically(image_path, output_json_path="data/roi_registry.json",
                             preview_path="data/auto_discovery_test.png", debug_path=None,
                             thresh=28):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Missing {image_path}! Make sure reference frame exists.")
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    hmask, vmask = paint_masks(img, thresh)
    isl = island_mask(img)
    cols = find_columns(hmask, vmask, w, h)
    if not cols:
        raise RuntimeError("No stall paint found - try lowering --thresh.")

    registry, sid = {}, 1
    preview = img.copy()
    dbg = img.copy()
    dbg[hmask > 0] = (255, 0, 255)
    dbg[vmask > 0] = (0, 165, 255)
    palette = [(0, 255, 0), (255, 255, 0), (0, 200, 255), (255, 0, 255), (255, 128, 0), (0, 128, 255)]

    print(f"Frame {w}x{h}: found {len(cols)} stall columns")
    for ci, (x0, x1) in enumerate(cols):
        p, y0, rms, n_sup = fit_lattice(hmask, x0, x1, h)
        ys, run = usable_bands(isl, x0, x1, p, y0, h)
        line_ys = [ys[i] for i in run] + ([ys[run[-1] + 1]] if len(run) else [])
        slope = block_tilt(hmask, x0, x1, line_ys[:-1] if line_ys else [], p)
        xc = (x0 + x1) / 2
        col = palette[ci % len(palette)]
        print(f"  col {ci + 1}: x={x0}-{x1}  pitch={p:.1f}px  stalls={len(run)}  "
              f"paint-lines={n_sup}  fit-rms={rms:.1f}px  tilt={np.degrees(np.arctan(slope)):.2f}deg")
        if n_sup < 0.5 * len(run):
            print(f"    ! only {n_sup}/{len(run)} lines backed by paint - check this column")

        for i in run:
            ya, yb = ys[i], ys[i + 1]
            quad = [[x0, ya + slope * (x0 - xc)], [x1, ya + slope * (x1 - xc)],
                    [x1, yb + slope * (x1 - xc)], [x0, yb + slope * (x0 - xc)]]
            quad = np.clip(np.array(quad), [0, 0], [w - 1, h - 1])
            spot = f"P{sid}"
            registry[spot] = {"spot_id": spot, "polygon_norm": (quad / [w, h]).round(5).tolist()}
            pts = quad.astype(np.int32)
            cv2.polylines(preview, [pts], True, col, 2)
            cv2.putText(preview, spot, (int(pts[0][0]) + 4, int(pts[0][1]) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            sid += 1
        for y in line_ys:
            cv2.line(dbg, (x0, int(y + slope * (x0 - xc))), (x1, int(y + slope * (x1 - xc))), (0, 255, 0), 1)

    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(registry, f, indent=2)
    os.makedirs(os.path.dirname(preview_path) or ".", exist_ok=True)
    cv2.imwrite(preview_path, preview)
    if debug_path:
        cv2.imwrite(debug_path, dbg)

    report_overlaps(registry, w, h)
    print(f"Generated {len(registry)} slots -> {output_json_path}")
    print(f"Verification overlay -> {preview_path}")
    return registry


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default="data/roi_registry.json")
    ap.add_argument("--preview", default="data/auto_discovery_test.png")
    ap.add_argument("--debug", default=None, help="save paint mask + fitted lines for troubleshooting")
    ap.add_argument("--thresh", type=int, default=28, help="paint brightness threshold")
    a = ap.parse_args()
    frame = a.image or ("data/cropped_frame_0.png" if os.path.exists("data/cropped_frame_0.png")
                        else "data/frame_0.png")
    find_slots_automatically(frame, a.out, a.preview, a.debug, a.thresh)