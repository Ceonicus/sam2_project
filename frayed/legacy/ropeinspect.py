import os
import gc
import json
import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
from skimage.morphology import skeletonize
from scipy.interpolate import UnivariateSpline

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
test_dir        = "/opt/sam2_project/input_images2"
output_dir      = "/opt/sam2_project/pipeline_b_output"
device          = "cuda"
STRIP_STEP      = 8
TEXT_PROMPT     = ["thin wire rope"]
MIN_COMPONENT_PIXELS = 100

# ── RUST path knobs (locked-in, working) ──
RUST_THRESHOLD     = 0.5
MIN_RUST_AREA      = 150
RUST_COL_FRAC      = 0.10
RUST_MIN_RUN       = 1
RUST_GAP_CLOSE     = 4
USE_GLINT_REJECT   = True
USE_RUST_COLFILTER = True
USE_RUST_AREA      = True

# ── FRAY path knobs (Path 2: wisp-in-band detection) ──
USE_FRAY          = True
USE_FRAY_SHAPE    = True  # detects frays already absorbed into the SAM rope mask
USE_FRAY_RING     = False # optional legacy bright-wisp ring cue; noisy on dark machinery
USE_FRAY_EDGE     = True  # detects thin visible strand tips not absorbed into SAM mask
FRAY_BAND_PX      = 30    # how far outside the rope to look for wisps
FRAY_V_MIN        = 90    # a ring pixel is "wisp" if brighter than this
FRAY_COL_MIN_FRAC = 0.08  # column is "frayed" if >= this frac of its ring is wisp.
                          # Raised from 0.04: real fray densely fills the ring;
                          # scattered clean-rope specks only fill a sliver.
FRAY_MIN_RUN      = 2     # consecutive frayed columns required
FRAY_GAP_CLOSE    = 3     # bridge small gaps within a fray run
FRAY_MIN_CLUSTER  = 18    # drop wisp blobs smaller than this (px) BEFORE column
                          # scoring — kills isolated speckle at the source so only
                          # packed clusters (real fray) contribute
FRAY_MAX_AREA     = 5000  # reject only very large machine-part blobs
FRAY_INSTANCE_MIN_AREA = 50  # min area for a reported fray site
FRAY_THIN_ONLY    = True  # keep only THIN bright structures (wisps), drop solid
                          # bright blobs (pulleys, housings) by shape
FRAY_THIN_KERNEL  = 9     # opening kernel: bright structures wider than this are
                          # treated as solid hardware and removed
FRAY_PROTRUDE_PX  = 5     # min one-sided mask expansion beyond the normal rope edge
FRAY_WIDTH_GAIN   = 0.35  # local rope silhouette must be this much wider than normal
FRAY_SCORE_PX     = 12    # protrusion size that maps to score 1.0
FRAY_WEAK_PROTRUDE_PX = 2.5 # lower-confidence protrusion for the second/smaller fray
FRAY_WEAK_WIDTH_GAIN  = 0.10
FRAY_WEAK_SCORE       = 0.36
FRAY_EDGE_V_MIN       = 58   # edge cue: visible strand must be at least this bright
FRAY_EDGE_REL_MIN     = 8    # and brighter than local background by this amount
FRAY_EDGE_MIN_AREA    = 6
FRAY_EDGE_MAX_AREA    = 220

# ── DEBUG ──
DEBUG_MODE = True
# ─────────────────────────────────────────────

os.makedirs(output_dir, exist_ok=True)
if DEBUG_MODE:
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

print(f"Loading SAM3 on {device}...")
overrides = dict(conf=0.5, task="segment", mode="predict",
                 model="sam3.pt", device=device, half=True, save=False)
predictor = SAM3SemanticPredictor(overrides=overrides)

# ─────────────────────────────────────────────
#  SEGMENTATION / GEOMETRY  (unchanged from the rust pipeline)
# ─────────────────────────────────────────────
def get_rope_mask(img_path, image):
    img_h, img_w = image.shape[:2]
    total_pixels = img_h * img_w
    predictor.set_image(img_path)
    try:
        results = predictor(text=TEXT_PROMPT)
    except RuntimeError as e:
        if "CUDA" in str(e) or "NVML" in str(e):
            print("  CUDA OOM — retrying...")
            gc.collect(); results = predictor(text=TEXT_PROMPT)
        else:
            raise
    if not results or results[0].masks is None:
        print("  SAM3 found no ropes."); gc.collect(); return None
    sam_masks = results[0].masks.data.cpu().numpy()
    print(f"  SAM3 found {len(sam_masks)} mask(s).")
    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    for m in sam_masks:
        m_r = cv2.resize(m.astype(np.float32),(img_w,img_h),interpolation=cv2.INTER_NEAREST)
        m_b = (m_r > 0.5).astype(np.uint8)
        if m_b.sum()/total_pixels > 0.15: continue
        contours,_ = cv2.findContours(m_b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        largest = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largest)
        if max(w,h)/(min(w,h)+1e-8) < 3.0: continue
        combined = np.maximum(combined, m_b)
    if combined.sum() == 0:
        gc.collect(); return None
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k)
    gc.collect()
    return combined

def split_ropes(rope_mask):
    n_comp, comp_labels = cv2.connectedComponents(rope_mask.astype(np.uint8))
    ropes = []
    for comp in range(1, n_comp):
        single = (comp_labels == comp).astype(np.uint8)
        if single.sum() < MIN_COMPONENT_PIXELS:
            continue
        ropes.append(single)
    return ropes

def extract_centerline(mask):
    skel = skeletonize(mask > 0).astype(np.uint8)
    ys_raw, xs_raw = np.where(skel)
    if len(xs_raw) < 10:
        ys_box, xs_box = np.where(mask > 0)
        order = np.argsort(xs_box); xs_raw, ys_raw = xs_box[order], ys_box[order]
    order  = np.argsort(xs_raw)
    xs_raw = xs_raw[order].astype(float); ys_raw = ys_raw[order].astype(float)
    xs_u, inv = np.unique(xs_raw.astype(int), return_inverse=True)
    ys_u = np.array([np.median(ys_raw[inv==i]) for i in range(len(xs_u))])
    if len(xs_u) < 4: return None, None, None
    spline = UnivariateSpline(xs_u, ys_u, k=3, s=len(xs_u)*5)
    deriv  = spline.derivative()
    xs = np.arange(xs_u[0], xs_u[-1], STRIP_STEP, dtype=float)
    if len(xs) < 2: return None, None, None
    ys = spline(xs)
    return xs, ys, np.degrees(np.arctan(deriv(xs)))

def measure_rope_half_width(mask, xs, ys):
    widths = []
    step = max(1, len(xs)//30)
    for cx,cy in zip(xs[::step], ys[::step]):
        cx,cy = int(round(cx)), int(round(cy))
        if cx < 0 or cx >= mask.shape[1]: continue
        rope_rows = np.where(mask[:,cx] > 0)[0]
        if len(rope_rows) < 2: continue
        local = rope_rows[np.abs(rope_rows-cy) <= 40]
        if len(local) >= 2: widths.append(int(local[-1]-local[0]))
    if not widths: return 10
    median_w = int(np.median(widths)); half_w = max(4, median_w//2+2)
    print(f"  [Width] Rope width ~{median_w}px → half-width={half_w}px")
    return half_w

def rectify_rope(image, mask):
    xs, ys, angles = extract_centerline(mask)
    if xs is None: return None,None,None,None,None
    half_w = measure_rope_half_width(mask, xs, ys)
    h, w = image.shape[:2]; strip_h = half_w*2
    strips_bgr, strips_mask, kept_xs, kept_ys = [], [], [], []
    for cx,cy,ang in zip(xs,ys,angles):
        cx,cy = int(round(cx)), int(round(cy))
        M = cv2.getRotationMatrix2D((cx,cy),ang,1.0)
        ri = cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        rm = cv2.warpAffine(mask.astype(np.float32),M,(w,h),flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        rm = (rm > 0.3).astype(np.float32)
        y0,y1 = cy-half_w, cy+half_w
        if y0<0 or y1>h: continue
        cb = ri[y0:y1,cx:cx+1]; cm = rm[y0:y1,cx:cx+1]
        if cb.shape[0] != strip_h: continue
        strips_bgr.append(cb); strips_mask.append(cm)
        kept_xs.append(cx); kept_ys.append(cy)
    if not strips_bgr: return None,None,None,None,half_w
    return (np.concatenate(strips_bgr,axis=1), np.concatenate(strips_mask,axis=1),
            np.array(kept_xs,dtype=float), np.array(kept_ys,dtype=float), half_w)

def trim_strip_edges(strip_bgr, strip_mask, xs, ys):
    n = strip_bgr.shape[1]; trim = max(4, n//33)
    if n - 2*trim < 1: return strip_bgr, strip_mask, xs, ys
    return (strip_bgr[:,trim:n-trim], strip_mask[:,trim:n-trim],
            xs[trim:len(xs)-trim], ys[trim:len(ys)-trim])

# ─────────────────────────────────────────────
#  Shared column-run filter (used by both rust and fray)
# ─────────────────────────────────────────────
def apply_run_filter(rusty, min_run, gap_close):
    n = len(rusty)
    # bridge gaps
    i = 0
    while i < n:
        if not rusty[i]:
            j = i
            while j < n and not rusty[j]: j += 1
            if i > 0 and j < n and (j - i) <= gap_close:
                rusty[i:j] = True
            i = j
        else:
            i += 1
    # drop short runs
    i = 0
    while i < n:
        if rusty[i]:
            j = i
            while j < n and rusty[j]: j += 1
            if (j - i) < min_run: rusty[i:j] = False
            i = j
        else:
            i += 1
    return rusty

def paint_columns(rusty, csc, xs, ys, image_shape, rope_mask, half_w):
    h, w = image_shape[:2]
    out = np.zeros((h, w), dtype=np.float32)
    n = min(len(rusty), len(xs))
    r = max(2, half_w // 2)
    for i in range(n):
        if not rusty[i]: continue
        cx, cy = int(round(xs[i])), int(round(ys[i]))
        if 0 <= cx < w and 0 <= cy < h:
            out[max(0,cy-r):min(h,cy+r+1), max(0,cx-r):min(w,cx+r+1)] = np.maximum(
                out[max(0,cy-r):min(h,cy+r+1), max(0,cx-r):min(w,cx+r+1)], csc[i])
    return out * (rope_mask > 0).astype(np.float32)

# ─────────────────────────────────────────────
#  RUST detector  (unchanged logic)
# ─────────────────────────────────────────────
def hsv_rust_map(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1,1), dtype=np.float32)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    color = (H >= 8) & (H <= 22) & (S >= 55) & (V >= 100)
    if USE_GLINT_REJECT:
        valid = V[V > 30]
        med = float(np.median(valid)) if valid.size else 128.0
        not_glint = ~((V > med + 15) & (S < 90))
    else:
        not_glint = np.ones_like(V, dtype=bool)
    rust = (color & not_glint).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    rust = cv2.morphologyEx(rust, cv2.MORPH_OPEN, k)
    rust = cv2.morphologyEx(rust, cv2.MORPH_DILATE, k, iterations=1)
    return rust.astype(np.float32)/255.0

def rust_score_for_rope(strip_bgr, xs, ys, image_shape, single, half_w):
    hsv_strip = hsv_rust_map(strip_bgr)
    n = strip_bgr.shape[1]
    rusty = np.zeros(n, dtype=bool); csc = np.zeros(n, dtype=np.float32)
    for i in range(n):
        col = hsv_strip[:, i]
        if float((col > 0.5).mean()) >= RUST_COL_FRAC:
            rusty[i] = True; csc[i] = float(col.max())
    if USE_RUST_COLFILTER:
        rusty = apply_run_filter(rusty, RUST_MIN_RUN, RUST_GAP_CLOSE)
    return paint_columns(rusty, csc, xs, ys, image_shape, single, half_w), hsv_strip

# ─────────────────────────────────────────────
#  FRAY detector  (Path 2: wisps in a band outside the rope)
# ─────────────────────────────────────────────
def fray_score_for_rope(image, single, xs, ys, half_w, cue_maps=None):
    """Detect broken strands sticking OUT of the rope silhouette.
    Uses two cues:
      1) silhouette protrusions already captured by SAM in the rope mask
      2) optional bright thin wisps in a ring just outside the rope mask"""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ring = (dilated rope) minus (rope) — where wisps would stick out
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FRAY_BAND_PX, FRAY_BAND_PX))
    dilated = cv2.dilate(single, k)
    ring = ((dilated > 0) & (single == 0)).astype(np.uint8)

    wisp = np.zeros((h, w), dtype=np.uint8)
    if USE_FRAY_RING:
        # wisps = bright pixels inside the ring
        wisp = ((gray >= FRAY_V_MIN) & (ring > 0)).astype(np.uint8)
        # clean up isolated speckle
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        wisp = cv2.morphologyEx(wisp, cv2.MORPH_OPEN, kk)

        # Drop SOLID bright blobs (pulleys, housings) but keep THIN wisps:
        # an opening with a big kernel removes structures thinner than the kernel,
        # leaving only the solid parts; subtracting that keeps only the thin bits.
        if FRAY_THIN_ONLY:
            big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FRAY_THIN_KERNEL, FRAY_THIN_KERNEL))
            solid = cv2.morphologyEx(wisp, cv2.MORPH_OPEN, big)   # only wide/solid regions
            wisp = cv2.subtract(wisp, solid)                      # keep thin structures only

        # Drop small isolated specks: keep only wisp blobs >= FRAY_MIN_CLUSTER px.
        # Real fray packs into sizable clusters; clean-rope noise is scattered dots.
        if FRAY_MIN_CLUSTER > 1:
            n_lbl, lbl_img, stats, _ = cv2.connectedComponentsWithStats(wisp)
            cleaned = np.zeros_like(wisp)
            for lbl in range(1, n_lbl):
                if stats[lbl, cv2.CC_STAT_AREA] >= FRAY_MIN_CLUSTER:
                    cleaned[lbl_img == lbl] = 1
            wisp = cleaned

    out = np.zeros((h, w), dtype=np.float32)
    n = len(xs)
    frayed = np.zeros(n, dtype=bool); fsc = np.zeros(n, dtype=np.float32)
    upper = np.zeros(n, dtype=np.float32)
    lower = np.zeros(n, dtype=np.float32)
    span  = np.zeros(n, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    # Cue 1: a real broken strand often deforms the SAM silhouette.  Measure the
    # normal rope edge, then flag columns where one side protrudes and the local
    # mask is much wider than the rope's median width.
    if USE_FRAY_SHAPE:
        win = max(FRAY_BAND_PX + half_w, half_w * 3)
        for i in range(n):
            cx, cy = int(round(xs[i])), float(ys[i])
            if not (0 <= cx < w): continue
            rows = np.where(single[:, cx] > 0)[0]
            if len(rows) < 2: continue
            rows = rows[np.abs(rows - cy) <= win]
            if len(rows) < 2: continue
            top, bot = float(rows.min()), float(rows.max())
            upper[i] = cy - top
            lower[i] = bot - cy
            span[i] = bot - top + 1.0
            valid[i] = True

        if valid.sum() >= 5:
            med_upper = float(np.median(upper[valid]))
            med_lower = float(np.median(lower[valid]))
            med_span  = float(np.median(span[valid]))
            min_span  = med_span * (1.0 + FRAY_WIDTH_GAIN)
            weak_span = med_span * (1.0 + FRAY_WEAK_WIDTH_GAIN)
            for i in np.where(valid)[0]:
                excess = max(upper[i] - med_upper, lower[i] - med_lower)
                if excess >= FRAY_PROTRUDE_PX and span[i] >= min_span:
                    frayed[i] = True
                    fsc[i] = max(fsc[i], min(1.0, excess / FRAY_SCORE_PX))
                elif excess >= FRAY_WEAK_PROTRUDE_PX and span[i] >= weak_span:
                    frayed[i] = True
                    fsc[i] = max(fsc[i], FRAY_WEAK_SCORE)

    if cue_maps is not None:
        shape_frayed = frayed.copy()
        shape_scores = fsc.copy()

    if USE_FRAY_RING:
        # Cue 2: per column, count wisp pixels in a vertical window around the centerline
        win = FRAY_BAND_PX + half_w
        for i in range(n):
            cx, cy = int(round(xs[i])), int(round(ys[i]))
            if not (0 <= cx < w): continue
            y0, y1 = max(0, cy-win), min(h, cy+win)
            col_ring  = ring[y0:y1, cx]
            col_wisp  = wisp[y0:y1, cx]
            ring_n = int(col_ring.sum())
            if ring_n < 1: continue
            frac = float(col_wisp.sum()) / float(ring_n)
            if frac >= FRAY_COL_MIN_FRAC:
                frayed[i] = True
                fsc[i] = max(fsc[i], min(1.0, frac / 0.25))   # 25% ring filled = score 1.0

    frayed = apply_run_filter(frayed, FRAY_MIN_RUN, FRAY_GAP_CLOSE)

    # Paint a compact blob at flagged columns.  For shape frays, bias the blob
    # toward the protruding side so the overlay lands on the broken strand.
    r = max(3, half_w)
    for i in range(n):
        if not frayed[i]: continue
        cx, cy = int(round(xs[i])), int(round(ys[i]))
        if 0 <= cx < w and 0 <= cy < h:
            draw_y = cy
            if valid[i]:
                draw_y = int(round(cy + max(0.0, lower[i] - upper[i]) * 0.5))
            cv2.circle(out, (cx, draw_y), r, float(max(fsc[i], 0.35)), -1)

    if cue_maps is not None:
        # Observe the shape cue separately; never change the combined score map.
        shape_out = np.zeros_like(out)
        shape_frayed = apply_run_filter(shape_frayed, FRAY_MIN_RUN, FRAY_GAP_CLOSE)
        for i in np.where(shape_frayed)[0]:
            cx, cy = int(round(xs[i])), int(round(ys[i]))
            if 0 <= cx < w and 0 <= cy < h:
                draw_y = int(round(cy + max(0.0, lower[i] - upper[i]) * 0.5)) if valid[i] else cy
                cv2.circle(shape_out, (cx, draw_y), r, float(max(shape_scores[i], 0.35)), -1)
        cue_maps["shape"] = shape_out
        cue_maps["edge"] = np.zeros_like(out)

    if USE_FRAY_EDGE:
        blur = cv2.GaussianBlur(gray, (0, 0), 5)
        rel = gray.astype(np.int16) - blur.astype(np.int16)
        edge = (((gray >= FRAY_EDGE_V_MIN) & (rel >= FRAY_EDGE_REL_MIN) & (ring > 0)) |
                ((gray >= FRAY_V_MIN) & (ring > 0))).astype(np.uint8) * 255
        edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)))
        n_lbl, lbl_img, stats, cent = cv2.connectedComponentsWithStats(edge)
        x_min = int(max(0, np.min(xs) + 60)) if len(xs) else 0
        x_max = int(min(w - 1, np.max(xs) - 60)) if len(xs) else w - 1
        edge_r = max(3, min(half_w, 7))
        for lbl in range(1, n_lbl):
            x, y, bw, bh, area = [int(v) for v in stats[lbl]]
            if area < FRAY_EDGE_MIN_AREA or area > FRAY_EDGE_MAX_AREA:
                continue
            if bw < 5 or bh > 35:
                continue
            cx, cy = cent[lbl]
            if cx < x_min or cx > x_max:
                continue
            score = min(0.48, FRAY_WEAK_SCORE + area / 700.0)
            cv2.circle(out, (int(round(cx)), int(round(cy))), edge_r, float(score), -1)
            if cue_maps is not None:
                cv2.circle(cue_maps["edge"], (int(round(cx)), int(round(cy))), edge_r, float(score), -1)
    return out, wisp, ring

# ─────────────────────────────────────────────
#  blob filter + report
# ─────────────────────────────────────────────
def filter_and_blobs(score_map, rope_mask_u8, threshold, min_area, use_area, max_area=None):
    binm = ((score_map > threshold) & (rope_mask_u8 > 0)).astype(np.uint8) * 255
    n_lbl, lbl_img, stats, _ = cv2.connectedComponentsWithStats(binm)
    if use_area:
        for lbl in range(1, n_lbl):
            area = stats[lbl, cv2.CC_STAT_AREA]
            too_small = area < min_area
            too_big   = (max_area is not None) and (area > max_area)
            if too_small or too_big:
                binm[lbl_img == lbl] = 0
    floor = min_area if use_area else 1
    kept = [(int(stats[l,0]),int(stats[l,1]),int(stats[l,2]),int(stats[l,3]),int(stats[l,4]))
            for l in range(1, n_lbl)
            if stats[l, cv2.CC_STAT_AREA] >= floor
            and (max_area is None or stats[l, cv2.CC_STAT_AREA] <= max_area)]
    return binm, kept

def filter_fray_instances(score_map, fray_region, threshold, shape_score=None,
                          edge_score=None, diagnostics=None):
    """Retain consistent evidence, then suppress nearby duplicate sites.

    Quality is the shape-score upper quartile, not painted area. A strong
    median can stand alone; localized strong shape evidence needs a bright
    strand's corroboration. The strong floor comes from the existing shape
    cue's minimum protrusion/score scale, not this image's score distribution.
    Without separate cues, require a strong median of the combined score.
    """
    raw = ((score_map > threshold) & (fray_region > 0)).astype(np.uint8) * 255
    n_lbl, lbl_img, stats, centroids = cv2.connectedComponentsWithStats(raw)
    strong_floor = FRAY_PROTRUDE_PX / FRAY_SCORE_PX
    records = []
    candidate_id = 0
    for lbl in range(1, n_lbl):
        x, y, bw, bh, area = [int(v) for v in stats[lbl]]
        mask = lbl_img[y:y+bh, x:x+bw] == lbl
        values = score_map[y:y+bh, x:x+bw][mask]
        shape = shape_score[y:y+bh, x:x+bw][mask] if shape_score is not None else None
        edge = edge_score[y:y+bh, x:x+bw][mask] if edge_score is not None else None
        representative = shape if shape is not None else values
        shape_support = int(np.count_nonzero(shape > threshold)) if shape is not None else None
        edge_support = int(np.count_nonzero(edge > threshold)) if edge is not None else None
        rec = dict(component_id=lbl, candidate_id=None, bbox=(x, y, bw, bh, area),
                   area=area, cx=float(centroids[lbl, 0]), cy=float(centroids[lbl, 1]),
                   quality=float(np.percentile(representative, 75)),
                   representative_median=float(np.median(representative)),
                   max_score=float(values.max()), mean_score=float(values.mean()),
                   p90_score=float(np.percentile(values, 90)),
                   shape_support=shape_support, edge_support=edge_support,
                   shape_fraction=shape_support / area if shape is not None else None,
                   edge_fraction=edge_support / area if edge is not None else None,
                   strong_floor=strong_floor, retained=False)
        if area < FRAY_INSTANCE_MIN_AREA:
            rec["reason"] = "too_small"
        elif area > FRAY_MAX_AREA:
            rec["reason"] = "too_large"
        else:
            candidate_id += 1
            rec["candidate_id"] = candidate_id
            if rec["representative_median"] >= strong_floor:
                rec["reason"] = "strong_median_shape" if shape is not None else "strong_median_combined"
            elif (shape is not None and edge is not None
                  and rec["quality"] >= strong_floor
                  and shape_support >= FRAY_INSTANCE_MIN_AREA
                  and edge_support >= FRAY_INSTANCE_MIN_AREA):
                rec["reason"] = "strong_upper_quartile_shape_with_bright_support"
            else:
                rec["reason"] = "insufficient_evidence"
        rec["evidence_reason"] = rec["reason"]
        records.append(rec)

    def nearby(a, b):
        # Box overlap, or a gap no larger than half the smaller short side.
        # This uses the painted site's local scale, rather than a fixed x gap;
        # proximity in BOTH image dimensions matters. No transitive merging.
        ax, ay, aw, ah, _ = a["bbox"]
        bx, by, bw, bh, _ = b["bbox"]
        dx = max(ax - (bx + bw), bx - (ax + aw), 0)
        dy = max(ay - (by + bh), by - (ay + ah), 0)
        return np.hypot(dx, dy) <= 0.5 * min(aw, ah, bw, bh)

    eligible = [r for r in records if r["reason"].startswith("strong_")]
    ranked = sorted(eligible, key=lambda r: (-r["quality"],
                    -r["representative_median"], -r["mean_score"], r["component_id"]))
    selected = []
    for rec in ranked:
        duplicate = next((s for s in selected if nearby(rec, s)), None)
        if duplicate is not None:
            rec["reason"] = "suppressed_duplicate"
            rec["duplicate_of"] = duplicate["candidate_id"]
        else:
            rec["retained"] = True
            selected.append(rec)
    # Weak satellites do not become independent sites or inflate a kept box.
    for rec in records:
        if rec["reason"] == "insufficient_evidence":
            duplicate = next((s for s in selected if nearby(rec, s)), None)
            if duplicate is not None:
                rec["reason"] = "suppressed_duplicate"
                rec["duplicate_of"] = duplicate["candidate_id"]

    out = np.zeros_like(raw)
    kept = []
    for rec in sorted(selected, key=lambda r: r["bbox"][:2]):
        out[lbl_img == rec["component_id"]] = 255
        kept.append(rec["bbox"])
    if diagnostics is not None:
        diagnostics.extend(records)
    if DEBUG_MODE:
        print(f"  Fray selection: {candidate_id} area-valid candidates -> {len(kept)} sites")
        for rec in records:
            if rec["retained"]:
                print(f"    C{rec['candidate_id']} bbox={rec['bbox'][:4]} area={rec['area']} "
                      f"quality={rec['quality']:.3f} max={rec['max_score']:.3f} "
                      f"mean={rec['mean_score']:.3f} shape={rec['shape_support']} "
                      f"bright={rec['edge_support']} reason={rec['reason']}")
    return out, kept

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run_inference():
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]
    print(f"\n[Inference] {len(test_files)} images. fray={USE_FRAY} "
          f"(band={FRAY_BAND_PX} v_min={FRAY_V_MIN} col_frac={FRAY_COL_MIN_FRAC} "
          f"min_cluster={FRAY_MIN_CLUSTER} min_run={FRAY_MIN_RUN} "
          f"area={FRAY_INSTANCE_MIN_AREA}-{FRAY_MAX_AREA} thin_only={FRAY_THIN_ONLY} "
          f"shape={USE_FRAY_SHAPE} ring={USE_FRAY_RING} edge={USE_FRAY_EDGE} "
          f"selection=evidence_based)")

    for idx, fname in enumerate(test_files):
        img_path = os.path.join(test_dir, fname)
        image = cv2.imread(img_path)
        if image is None: continue
        h, w = image.shape[:2]
        name = os.path.splitext(fname)[0]
        print(f"\n[{idx+1}/{len(test_files)}] {fname}")

        rope_mask = get_rope_mask(img_path, image)
        if rope_mask is None:
            print("  Skipping — no rope mask."); continue
        ropes = split_ropes(rope_mask)
        if not ropes:
            print("  Skipping — no usable rope."); continue
        print(f"  {len(ropes)} rope component(s)")

        rust_score = np.zeros((h, w), dtype=np.float32)
        fray_score = np.zeros((h, w), dtype=np.float32)
        shape_score = np.zeros_like(fray_score)
        edge_score = np.zeros_like(fray_score)
        wisp_dbg   = np.zeros((h, w), dtype=np.uint8)
        fray_dbg   = np.zeros((h, w), dtype=np.uint8)

        for ci, single in enumerate(ropes):
            res = rectify_rope(image, single)
            if res[0] is None:
                print(f"  rope {ci+1}: rectify failed."); continue
            strip_bgr, strip_mask, xs, ys, half_w = res
            strip_bgr, strip_mask, xs, ys = trim_strip_edges(strip_bgr, strip_mask, xs, ys)
            n_cols = strip_bgr.shape[1]
            xs = xs[:n_cols]; ys = ys[:n_cols]
            if n_cols < 5: continue

            rs, _ = rust_score_for_rope(strip_bgr, xs, ys, image.shape, single, half_w)
            rust_score = np.maximum(rust_score, rs)

            if USE_FRAY:
                cue_maps = {}
                fs, wisp, ring = fray_score_for_rope(image, single, xs, ys, half_w, cue_maps)
                fray_score = np.maximum(fray_score, fs)
                shape_score = np.maximum(shape_score, cue_maps["shape"])
                edge_score = np.maximum(edge_score, cue_maps["edge"])
                wisp_dbg = np.maximum(wisp_dbg, wisp)
                fray_dbg = np.maximum(fray_dbg, (np.clip(fs, 0, 1) * 255).astype(np.uint8))

        rope_mask_u8 = cv2.resize(rope_mask,(w,h),interpolation=cv2.INTER_NEAREST)

        # independent flag/blobs for each defect
        rust_map, rust_blobs = filter_and_blobs(rust_score, rope_mask_u8,
                                                RUST_THRESHOLD, MIN_RUST_AREA, USE_RUST_AREA)
        # fray flag uses the dilated mask region, not just the rope, so wisps paint
        fray_region = cv2.dilate(rope_mask_u8, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,(FRAY_BAND_PX,FRAY_BAND_PX)))
        fray_diagnostics = []
        fray_map, fray_blobs = filter_fray_instances(
            fray_score, fray_region, 0.3, shape_score, edge_score,
            fray_diagnostics) if USE_FRAY else (np.zeros((h,w),np.uint8),[])

        print(f"  Rust blobs: {len(rust_blobs)}  sizes={[b[4] for b in rust_blobs]}  "
              f"[{'RUST' if rust_blobs else 'no rust'}]")
        if USE_FRAY:
            print(f"  Fray blobs: {len(fray_blobs)}  sizes={[b[4] for b in fray_blobs]}  "
                  f"[{'FRAY' if fray_blobs else 'no fray'}]")

        # overlay: rust = yellow, fray = red
        overlay = image.copy()
        overlay[rust_map > 0] = [0, 255, 255]      # yellow
        overlay[fray_map > 0] = [0, 0, 255]        # red
        for (x,y,bw,bh,area) in rust_blobs:
            cv2.rectangle(overlay,(x,y),(x+bw,y+bh),(0,255,255),2)
            cv2.putText(overlay,f"RUST ({area}px)",(x,max(0,y-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        for (x,y,bw,bh,area) in fray_blobs:
            cv2.rectangle(overlay,(x,y),(x+bw,y+bh),(0,0,255),2)
            cv2.putText(overlay,f"FRAY ({area}px)",(x,max(0,y+bh+16)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

        cv2.imwrite(f"{output_dir}/{name}_overlay.png", overlay)
        cv2.imwrite(f"{output_dir}/{name}_rope_mask.png", rope_mask_u8*255)
        if DEBUG_MODE and USE_FRAY:
            with open(f"{debug_dir}/{name}_fray_candidates.json", "w") as f:
                json.dump(fray_diagnostics, f, indent=2)
            cv2.imwrite(f"{debug_dir}/{name}_shape_score.png", (np.clip(shape_score, 0, 1)*255).astype(np.uint8))
            cv2.imwrite(f"{debug_dir}/{name}_bright_score.png", (np.clip(edge_score, 0, 1)*255).astype(np.uint8))
            cv2.imwrite(f"{debug_dir}/{name}_wisp.png", wisp_dbg*255)
            cv2.imwrite(f"{debug_dir}/{name}_fray_score.png", fray_dbg)
            cv2.imwrite(f"{debug_dir}/{name}_fray_instances.png", fray_map)

    print("\nDone.")

if __name__ == "__main__":
    run_inference()
