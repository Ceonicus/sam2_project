"""Preserved colleague comparison; see runtime_test_output/colleague_algorithm_equivalence.md.
No model is imported/loaded until startup. Do not substitute tuned YOLO logic.
FRAY-ONLY: rust functions remain for historical inspection but are not called.
Original fray selection is preserved with identity-safe selected-candidate membership.
"""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from skimage.morphology import skeletonize
os.environ['YOLO_AUTOINSTALL'] = 'false'
ROOT = Path(__file__).resolve().parent
ORIGINAL_COMMIT = '8fa6f4f315316c5dd1e11ea62ababf3d7945e400'
ORIGINAL_SHA256 = '4f65554d38e74824e98820d2a62d29e09f77b90a17834668c17158c841a12a05'

STRIP_STEP      = 8
TEXT_PROMPT     = ["thin wire rope"]
MIN_COMPONENT_PIXELS = 100
RUST_THRESHOLD     = 0.5
MIN_RUST_AREA      = 150
RUST_COL_FRAC      = 0.10
RUST_MIN_RUN       = 1
RUST_GAP_CLOSE     = 4
USE_GLINT_REJECT   = True
USE_RUST_COLFILTER = True
USE_RUST_AREA      = True
USE_FRAY          = True
USE_FRAY_SHAPE    = True
USE_FRAY_RING     = False
USE_FRAY_EDGE     = True
FRAY_BAND_PX      = 30
FRAY_V_MIN        = 90
FRAY_COL_MIN_FRAC = 0.08
FRAY_MIN_RUN      = 2
FRAY_GAP_CLOSE    = 3
FRAY_MIN_CLUSTER  = 18
FRAY_MIN_AREA     = 120
FRAY_MAX_AREA     = 5000
FRAY_TARGET_COUNT = 2
FRAY_INSTANCE_MIN_AREA = 50
FRAY_MIN_SEPARATION_PX = 180
FRAY_THIN_ONLY    = True
FRAY_THIN_KERNEL  = 9
FRAY_PROTRUDE_PX  = 5
FRAY_WIDTH_GAIN   = 0.35
FRAY_SCORE_PX     = 12
FRAY_WEAK_PROTRUDE_PX = 2.5
FRAY_WEAK_WIDTH_GAIN  = 0.10
FRAY_WEAK_SCORE       = 0.36
FRAY_EDGE_V_MIN       = 58
FRAY_EDGE_REL_MIN     = 8
FRAY_EDGE_MIN_AREA    = 6
FRAY_EDGE_MAX_AREA    = 220
DEBUG_MODE = True

VARIANT = 'SAM2'
MODEL_NAME = 'sam2.1_t.pt'


def create_segmenter(model_path, device):
    from ultralytics import SAM
    return SAM(str(model_path))


def acquire_raw_masks(segmenter, path, device):
    # SAM.predict -> SAM2Predictor.inference -> inherited generate(): no user
    # prompts. Default 32x32 point grid, 64 points/batch, zero extra crop layers,
    # quality >0.88, stability >0.95. Masks are class-agnostic, not labelled rope.
    results = segmenter.predict(source=str(path), device=device,
                                half=True, save=False, verbose=False)
    if not results or results[0].masks is None:
        return []
    return results[0].masks.data.cpu().numpy()


def add_timing(timings, key, started):
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + (time.perf_counter()-started)*1000


def prepare_fray_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    context = dict(gray=gray,
                   ring_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FRAY_BAND_PX, FRAY_BAND_PX)),
                   edge_kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)))
    if USE_FRAY_EDGE:
        blur = cv2.GaussianBlur(gray, (0, 0), 5)
        context['relative_brightness'] = gray.astype(np.int16) - blur.astype(np.int16)
    return context

def clean_rope_masks(sam_masks, image_shape):
    img_h, img_w = image_shape[:2]
    total_pixels = img_h * img_w
    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    accepted = 0
    for m in sam_masks:
        m_r = cv2.resize(m.astype(np.float32),(img_w,img_h),interpolation=cv2.INTER_NEAREST)
        m_b = (m_r > 0.5).astype(np.uint8)
        if m_b.sum()/total_pixels > 0.15: continue
        contours,_ = cv2.findContours(m_b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        largest = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largest)
        if max(w,h)/(min(w,h)+1e-8) < 3.0: continue
        accepted += 1
        combined = np.maximum(combined, m_b)
    if combined.sum() == 0:
        gc.collect(); return None, accepted
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k)
    gc.collect()
    return combined, accepted


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
    # A zero halo preserves the full-frame skeleton's background at the ROI edge,
    # including components touching the original image boundary.
    bounds_mask = mask if mask.dtype == np.uint8 else (mask > 0).astype(np.uint8)
    x0, y0, width, height = cv2.boundingRect(bounds_mask)
    if not width or not height: return None, None, None
    roi = mask[y0:y0+height, x0:x0+width]
    skel = skeletonize(np.pad(roi > 0, 1)).astype(np.uint8)
    ys_raw, xs_raw = np.where(skel)
    xs_raw = xs_raw + x0 - 1; ys_raw = ys_raw + y0 - 1
    if len(xs_raw) < 10:
        ys_box, xs_box = np.where(roi > 0)
        xs_box = xs_box + x0; ys_box = ys_box + y0
        order = np.argsort(xs_box); xs_raw, ys_raw = xs_box[order], ys_box[order]
    # Sort each x group by y once; indexing its two middle values gives exactly
    # the same median as repeatedly selecting ys_raw[inv == i].
    order  = np.lexsort((ys_raw, xs_raw))
    xs_raw = xs_raw[order].astype(float); ys_raw = ys_raw[order].astype(float)
    xs_u, starts, counts = np.unique(xs_raw.astype(int), return_index=True, return_counts=True)
    ys_u = (ys_raw[starts+(counts-1)//2] + ys_raw[starts+counts//2]) * 0.5
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
        y0, y1 = max(0, cy-40), min(mask.shape[0], cy+41)
        if y1 <= y0: continue
        local = np.flatnonzero(mask[y0:y1,cx] > 0) + y0
        if len(local) >= 2: widths.append(int(local[-1]-local[0]))
    if not widths: return 10
    median_w = int(np.median(widths)); half_w = max(4, median_w//2+2)
    print(f"  [Width] Rope width ~{median_w}px → half-width={half_w}px")
    return half_w


def rectify_rope(image, mask, timings=None):
    started = time.perf_counter()
    xs, ys, angles = extract_centerline(mask)
    add_timing(timings, 'centerline_ms', started)
    if xs is None: return None,None,None,None,None
    started = time.perf_counter()
    half_w = measure_rope_half_width(mask, xs, ys)
    add_timing(timings, 'width_measurement_ms', started)
    started = time.perf_counter()
    h, w = image.shape[:2]
    cx = np.rint(xs).astype(np.int64); cy = np.rint(ys).astype(np.int64)
    keep = (cy-half_w >= 0) & (cy+half_w <= h)
    cx, cy, angles = cx[keep], cy[keep], angles[keep]
    if not len(cx): return None,None,None,None,half_w
    matrices = np.array([cv2.invertAffineTransform(cv2.getRotationMatrix2D(
        (int(x), int(y)), float(angle), 1.0)) for x, y, angle in zip(cx, cy, angles)])
    rows = cy[None, :] + np.arange(-half_w, half_w)[:, None]
    # Match warpAffine INTER_LINEAR's fixed-point coordinate construction:
    # separately round the x contribution and (y + translation) at 10 bits,
    # then round to the 5-bit interpolation table. Direct floating-point normal
    # coordinates can choose different interpolation bins at their boundaries.
    xx = (np.rint((matrices[:,0,1]*rows + matrices[:,0,2])*1024).astype(np.int64)
          + np.rint(matrices[:,0,0]*cx*1024).astype(np.int64) + 16) >> 5
    yy = (np.rint((matrices[:,1,1]*rows + matrices[:,1,2])*1024).astype(np.int64)
          + np.rint(matrices[:,1,0]*cx*1024).astype(np.int64) + 16) >> 5
    mapping = np.stack((xx >> 5, yy >> 5), axis=-1).astype(np.int16)
    fractions = (((yy & 31) << 5) | (xx & 31)).astype(np.uint16)
    strip_bgr = cv2.remap(image, mapping, fractions, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
    strip_mask = cv2.remap(mask.astype(np.float32), mapping, fractions, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    add_timing(timings, 'rectification_ms', started)
    return (strip_bgr, (strip_mask > 0.3).astype(np.float32),
            cx.astype(float), cy.astype(float), half_w)


def trim_strip_edges(strip_bgr, strip_mask, xs, ys):
    n = strip_bgr.shape[1]; trim = max(4, n//33)
    if n - 2*trim < 1: return strip_bgr, strip_mask, xs, ys
    return (strip_bgr[:,trim:n-trim], strip_mask[:,trim:n-trim],
            xs[trim:len(xs)-trim], ys[trim:len(ys)-trim])


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


def fray_score_for_rope(image, single, xs, ys, half_w, image_context=None):
    """Detect broken strands sticking OUT of the rope silhouette.
    Uses two cues:
      1) silhouette protrusions already captured by SAM in the rope mask
      2) optional bright thin wisps in a ring just outside the rope mask"""
    h, w = image.shape[:2]
    context = prepare_fray_image(image) if image_context is None else image_context
    gray = context['gray']

    # ring = (dilated rope) minus (rope) — where wisps would stick out
    k = context['ring_kernel']
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
            y0 = max(0, int(np.floor(cy-win))-1)
            y1 = min(h, int(np.ceil(cy+win))+2)
            if y1 <= y0: continue
            rows = np.flatnonzero(single[y0:y1, cx] > 0) + y0
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

    if USE_FRAY_EDGE:
        rel = context['relative_brightness']
        edge = (((gray >= FRAY_EDGE_V_MIN) & (rel >= FRAY_EDGE_REL_MIN) & (ring > 0)) |
                ((gray >= FRAY_V_MIN) & (ring > 0))).astype(np.uint8) * 255
        edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN,
                                context['edge_kernel'])
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
    return out, wisp, ring


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


def filter_fray_instances(score_map, fray_region, threshold):
    """Return a display map and per-fray boxes.

    The score map can contain high-confidence primary frays and weaker secondary
    strand-tip candidates.  Select separate sites instead of splitting one site.
    """
    raw = ((score_map > threshold) & (fray_region > 0)).astype(np.uint8) * 255
    n_lbl, lbl_img, stats, _ = cv2.connectedComponentsWithStats(raw)

    candidates = []
    for lbl in range(1, n_lbl):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < FRAY_INSTANCE_MIN_AREA or area > FRAY_MAX_AREA:
            continue
        mask = ((lbl_img == lbl).astype(np.uint8) * 255)
        x, y, bw, bh = [int(v) for v in stats[lbl, :4]]
        ys, xs = np.where(mask > 0)
        candidates.append({
            "area": area,
            "mask": mask,
            "bbox": (x, y, bw, bh, area),
            "score": float(score_map[mask > 0].mean()) if area else 0.0,
            "max_score": float(score_map[mask > 0].max()) if area else 0.0,
            "cx": float(xs.mean()) if len(xs) else x + bw * 0.5,
            "cy": float(ys.mean()) if len(ys) else y + bh * 0.5,
        })

    instances = []
    if candidates:
        primary_pool = [r for r in candidates if r["area"] >= FRAY_MIN_AREA]
        ranked_primary = sorted(primary_pool or candidates,
                                key=lambda r: (r["area"], r["max_score"], r["score"]),
                                reverse=True)
        first = ranked_primary[0]
        instances.append(first)
        if FRAY_TARGET_COUNT > 1:
            ranked_rest = [r for r in candidates if r is not first]
            right = [r for r in ranked_rest
                     if r["cx"] >= first["cx"] + FRAY_MIN_SEPARATION_PX]
            pool = right if right else [
                r for r in ranked_rest
                if abs(r["cx"] - first["cx"]) >= FRAY_MIN_SEPARATION_PX
            ]
            if pool:
                instances.append(max(pool,
                                     key=lambda r: (r["area"] >= FRAY_MIN_AREA,
                                                    r["area"], r["max_score"], r["score"])))
        # Candidates contain arrays: exclude selected objects by identity, not equality.
        ranked_rest = sorted([r for r in candidates if all(r is not inst for inst in instances)],
                             key=lambda r: (r["area"], r["max_score"], r["score"]),
                             reverse=True)
        for cand in ranked_rest:
            if len(instances) >= FRAY_TARGET_COUNT:
                break
            if all(abs(cand["cx"] - inst["cx"]) >= FRAY_MIN_SEPARATION_PX
                   for inst in instances):
                instances.append(cand)

    out = np.zeros_like(raw)
    kept = []
    for inst in sorted(instances, key=lambda r: r["bbox"][0]):
        out[inst["mask"] > 0] = 255
        kept.append(inst["bbox"])
    return out, kept


TIMING_KEYS = ('image_load_ms', 'segmentation_ms', 'mask_processing_ms',
               'component_splitting_ms', 'geometry_ms', 'centerline_ms',
               'width_measurement_ms', 'rectification_ms', 'rust_scoring_ms',
               'fray_scoring_ms', 'candidate_filtering_ms', 'visualization_save_ms',
               'total_processing_ms')
CSV_COLUMNS = ('filename', 'status', 'usable_rope_count', 'raw_segmentation_mask_count',
               'accepted_rope_mask_count', 'preliminary_fray_candidates',
               'final_fray_count', 'rust_detected', 'rust_count', 'rust_evaluation', *TIMING_KEYS,
               'final_overlay_path', 'error')


def downstream(image, rope_mask, record):
    """Original decisions with timing/diagnostics; no backend-specific behavior."""
    started = time.perf_counter()
    ropes = split_ropes(rope_mask)
    add_timing(record, 'component_splitting_ms', started)
    record['component_count'] = len(ropes)
    if not ropes:
        raise ValueError('No usable rope components')
    h, w = image.shape[:2]
    # FRAY-ONLY: no rust maps, scoring, filtering or visualization work.
    # Keep compatibility placeholders; None explicitly means not evaluated.
    rust_map = rust_score = None
    record.update(rust_boxes=[], rust_count=0, rust_detected=False,
                  rust_scoring_ms=0.0, rust_evaluation='DISABLED_NOT_EVALUATED')
    fray_score = np.zeros((h, w), np.float32)
    wisp_dbg = np.zeros((h, w), np.uint8)
    context = None
    record['components'] = []
    for ci, single in enumerate(ropes):
        started = time.perf_counter()
        res = rectify_rope(image, single, record)
        if res[0] is None:
            record['components'].append(dict(id=ci+1, inspected=False, reason='rectification_failed'))
            add_timing(record, 'geometry_ms', started)
            continue
        strip_bgr, strip_mask, xs, ys, half_w = res
        samples = len(xs)
        strip_bgr, strip_mask, xs, ys = trim_strip_edges(strip_bgr, strip_mask, xs, ys)
        n_cols = strip_bgr.shape[1]
        xs = xs[:n_cols]; ys = ys[:n_cols]
        add_timing(record, 'geometry_ms', started)
        if n_cols < 5:
            record['components'].append(dict(id=ci+1, inspected=False, reason='fewer_than_five_columns'))
            continue
        # rust_score_for_rope intentionally bypassed; fray inputs are unchanged.
        if USE_FRAY:
            started = time.perf_counter()
            if context is None:
                context = prepare_fray_image(image)
            fs, wisp, ring = fray_score_for_rope(image, single, xs, ys, half_w, context)
            fray_score = np.maximum(fray_score, fs)
            wisp_dbg = np.maximum(wisp_dbg, wisp)
            add_timing(record, 'fray_scoring_ms', started)
        record['usable_rope_count'] += 1
        record['components'].append(dict(id=ci+1, inspected=True, sampled_points=samples,
                                         retained_points=n_cols, half_width=half_w))
    record['partial_inspection'] = record['usable_rope_count'] != len(ropes)
    if not record['usable_rope_count']:
        raise ValueError('Geometry failed for every rope component')
    started = time.perf_counter()
    rope_mask_u8 = cv2.resize(rope_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    # Rust filter_and_blobs intentionally bypassed in this FRAY-ONLY benchmark.
    fray_region = cv2.dilate(rope_mask_u8, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (FRAY_BAND_PX, FRAY_BAND_PX)))
    if USE_FRAY:
        raw = ((fray_score > 0.3) & (fray_region > 0)).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(raw)
        record['preliminary_fray_components'] = n-1
        record['preliminary_fray_candidates'] = int(np.count_nonzero(
            (stats[1:, cv2.CC_STAT_AREA] >= FRAY_INSTANCE_MIN_AREA) &
            (stats[1:, cv2.CC_STAT_AREA] <= FRAY_MAX_AREA)))
        fray_map, fray_blobs = filter_fray_instances(fray_score, fray_region, 0.3)
    else:
        fray_map, fray_blobs = np.zeros((h, w), np.uint8), []
    add_timing(record, 'candidate_filtering_ms', started)
    record.update(fray_boxes=fray_blobs, final_fray_count=len(fray_blobs))
    return rust_map, fray_map, rust_score, fray_score, wisp_dbg


def render_result(image, rust_map, fray_map, rust_blobs, fray_blobs):
    # Original fray colors, paint order, boxes, labels and text placement.
    # Rust parameters are unused compatibility placeholders.
    overlay = image.copy()
    overlay[fray_map > 0] = [0, 0, 255]
    for (x,y,bw,bh,area) in fray_blobs:
        cv2.rectangle(overlay,(x,y),(x+bw,y+bh),(0,0,255),2)
        cv2.putText(overlay,f"FRAY ({area}px)",(x,max(0,y+bh+16)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    return overlay


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not save {path}')


def sync_cuda(torch, device):
    if device != 'cpu':
        torch.cuda.synchronize(device)


def print_memory(torch, device):
    fields = ('MemTotal:', 'MemAvailable:', 'SwapTotal:', 'SwapFree:')
    info = {}
    try:
        info['system'] = [line.strip() for line in Path('/proc/meminfo').read_text().splitlines()
                          if line.startswith(fields)]
    except OSError:
        pass
    if device != 'cpu':
        info.update(torch_allocated_bytes=torch.cuda.memory_allocated(device),
                    torch_reserved_bytes=torch.cuda.memory_reserved(device))
    print('Memory (allocator is not total Jetson RAM):', json.dumps(info), flush=True)
    return info


def inspect_image(segmenter, path, output, device, torch, debug):
    total = time.perf_counter()
    record = dict(filename=path.name, source_path=str(path), status='INSPECTION_FAILED',
                  usable_rope_count=0, raw_segmentation_mask_count=0,
                  accepted_rope_mask_count=0, preliminary_fray_candidates=0,
                  final_fray_count=0, rust_count=0, rust_detected=False,
                  rust_scoring_ms=0.0, rust_evaluation='DISABLED_NOT_EVALUATED',
                  final_overlay_path='', error='')
    stage = 'image_load'
    try:
        started = time.perf_counter()
        image = cv2.imread(str(path))
        add_timing(record, 'image_load_ms', started)
        if image is None:
            raise ValueError('Image could not be decoded')
        stage = 'segmentation'
        sync_cuda(torch, device)
        started = time.perf_counter()
        try:
            raw = acquire_raw_masks(segmenter, path, device)
            sync_cuda(torch, device)
        finally:
            add_timing(record, 'segmentation_ms', started)
        record['raw_segmentation_mask_count'] = len(raw)
        if not len(raw):
            raise ValueError('Segmentation returned no masks')
        stage = 'mask_processing'
        started = time.perf_counter()
        rope_mask, accepted = clean_rope_masks(raw, image.shape)
        record['accepted_rope_mask_count'] = accepted
        add_timing(record, 'mask_processing_ms', started)
        del raw
        if rope_mask is None:
            raise ValueError('All segmentation masks rejected by original cleanup')
        stage = 'downstream'
        rust_map, fray_map, rust_score, fray_score, wisp = downstream(image, rope_mask, record)
        stage = 'visualization_save'
        started = time.perf_counter()
        overlay = render_result(image, rust_map, fray_map, record['rust_boxes'], record['fray_boxes'])
        dest = output / 'final_overlays' / path.name
        write_image(dest, overlay)
        artifact = output / 'images' / path.name
        artifact.mkdir(parents=True, exist_ok=True)
        write_image(artifact/'rope_mask.png', rope_mask*255)
        if debug:
            write_image(artifact/'fray_score.png', (np.clip(fray_score,0,1)*255).astype(np.uint8))
            write_image(artifact/'fray_instances.png', fray_map)
            write_image(artifact/'wisp.png', wisp*255)
        record.update(final_overlay_path=str(dest), artifacts=str(artifact),
                      status='FRAY_DETECTED' if record['final_fray_count'] else 'OK_NO_FRAY')
        add_timing(record, 'visualization_save_ms', started)
    except Exception as exc:
        record.update(error=f'{type(exc).__name__}: {exc}', failed_stage=stage)
        # Abort model/device failures instead of retrying an unfit model 161 times.
        record['stop_run'] = stage == 'segmentation' and not isinstance(exc, ValueError)
        traceback.print_exc()
    add_timing(record, 'total_processing_ms', total)
    print(json.dumps({k: record.get(k) for k in ('filename','status','usable_rope_count',
          'raw_segmentation_mask_count','accepted_rope_mask_count','final_fray_count','rust_count','error')}), flush=True)
    print('Stage timing ms:', json.dumps({k:record[k] for k in TIMING_KEYS if k in record}), flush=True)
    return record


def save_summary(output, summary):
    temp = output/'summary.json.tmp'
    temp.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    temp.replace(output/'summary.json')


def main():
    parser = argparse.ArgumentParser(description=f'{VARIANT} + original colleague FRAY-ONLY comparison; rust disabled')
    parser.add_argument('--input', type=Path, default=ROOT/'frayed/validation_images')
    parser.add_argument('--output', type=Path, default=ROOT/f'runtime_test_output/colleague_{VARIANT.lower()}_validation')
    parser.add_argument('--device', default='0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    images = sorted((p for p in source.iterdir() if p.is_file() and not p.is_symlink()
                     and not p.name.startswith('.') and p.suffix.lower() in {'.jpg','.jpeg','.png'}),
                    key=lambda p:p.name) if source.is_dir() else [source] if source.is_file() else []
    if not images:
        parser.error(f'No source images: {source}')
    if len({p.name.casefold() for p in images}) != len(images):
        parser.error('Duplicate flat-overlay basenames')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists():
        parser.error('Existing summary found; use the runner to timestamp-backup prior results')
    (output/'final_overlays').mkdir(exist_ok=True)
    (output/'selected_images.txt').write_text(''.join(p.name+'\n' for p in images))
    model_path = ROOT/MODEL_NAME
    summary = dict(benchmark='FRAY-ONLY', rust_evaluation='DISABLED_NOT_EVALUATED', variant=VARIANT, original_commit=ORIGINAL_COMMIT, original_sha256=ORIGINAL_SHA256,
                   model_path=str(model_path), input=str(source), image_count=len(images),
                   source_paths=[str(p) for p in images], run_status='STARTING', images=[],
                   timing_note='geometry_ms includes centerline, width, rectification and trim; do not sum overlapping stages')
    save_summary(output, summary)
    with (output/'review.csv').open('w', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
    print('FRAY-ONLY comparison\nRust detection disabled', flush=True)
    print(f'Variant: {VARIANT}\nColleague commit: {ORIGINAL_COMMIT}\nPython: {sys.version}\n'
          f'Model: {model_path}\nInput: {source}\nImage count: {len(images)}', flush=True)
    for p in images:
        print(p.name, flush=True)
    try:
        if not model_path.is_file():
            raise FileNotFoundError(f'Required local checkpoint missing: {model_path}; no download attempted')
        import torch
        available = torch.cuda.is_available()
        print(f'PyTorch: {torch.__version__}\nCUDA build: {torch.version.cuda}\nCUDA available: {available}\n'
              f'CUDA device count: {torch.cuda.device_count()}\n'
              f'GPU: {torch.cuda.get_device_name(0) if available else "NO CUDA"}', flush=True)
        requested = str(args.device)
        device = 'cpu' if requested == 'cpu' else f'cuda:{int(requested.split(":")[-1] if requested != "cuda" else 0)}'
        if device != 'cpu' and not available:
            raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
        summary['device'] = device
        summary['startup_memory'] = print_memory(torch, device)
        print(f'Constructing genuine {VARIANT}; requested half=True. SAM3 loads lazily in set_image().', flush=True)
        save_summary(output, summary)
        start = time.perf_counter()
        segmenter = create_segmenter(model_path, device)
        summary['constructor_ms'] = (time.perf_counter()-start)*1000
        summary['run_status'] = 'RUNNING'
        for path in images:
            # Persist the current attempt before a possible kernel OOM kill.
            summary['current_attempt'] = str(path)
            save_summary(output, summary)
            record = inspect_image(segmenter, path, output, device, torch, args.debug)
            summary['images'].append(record)
            with (output/'review.csv').open('a', newline='') as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore').writerow(record)
            save_summary(output, summary)
            print_memory(torch, device)
            if record.get('stop_run'):
                break
        summary['run_status'] = 'COMPLETE' if len(summary['images']) == len(images) else 'ABORTED'
    except Exception as exc:
        traceback.print_exc()
        summary.update(run_status='FAILED', process_error=f'{type(exc).__name__}: {exc}')
    aggregates = {}
    for key in TIMING_KEYS:
        values = [r[key] for r in summary['images'] if key in r]
        if values:
            aggregates[key] = dict(count=len(values), mean=float(np.mean(values)), median=float(np.median(values)))
    summary['timing_aggregates_ms'] = aggregates
    save_summary(output, summary)
    print('Aggregate timings:', json.dumps(aggregates, indent=2), flush=True)
    print('Run status:', summary['run_status'], 'attempted/completed records:', len(summary['images']),
          'fray sites:', sum(r['final_fray_count'] for r in summary['images']), flush=True)
    return 0 if summary['run_status']=='COMPLETE' and all(r['status']!='INSPECTION_FAILED' for r in summary['images']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
