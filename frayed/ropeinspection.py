"""Offline horizontal-rope inspection with YOLO segmentation.

Geometry preserves ropeinspect.py's calculations. Shape scoring also checks
local width-profile widening; candidate acceptance uses rope-relative area
and sustained shape evidence.
Rectification samples only the required strip pixels. Shape inspection extends
to mask-supported endpoints; strip trimming and bright-strand coverage stay
unchanged.
No segmentation model is loaded until main() runs. Statuses
refer only to the detected, successfully inspected components; an undetected
rope cannot be inferred from an expected rope count.
"""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import time

os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("ULTRALYTICS_SAFE_LOAD", "true")

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "frayed/fray_detection.pt"
DEFAULT_INPUT = ROOT / "frayed/images"
DEFAULT_OUTPUT = ROOT / "runtime_test_output/fray_yolo_pipeline"
YOLO_IMAGE_SIZE = 960
YOLO_CONFIDENCE = 0.25
ROPE_CLASS_ID = 0
ROPE_DUPLICATE_IOU_THRESHOLD = 0.80  # observed duplicates: IoU approximately 0.82-0.93
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXCLUDED_DIRECTORIES = {
    "archive", "archived_images", "runtime_test_output", "sam3_test_output",
    "predictions", "prediction", "masks", "overlays", "debug", "runs",
    "output", "outputs", "results", "labels", "annotations", "augmented",
    "augmentations", ".git", ".codex", ".agents", "__pycache__",
}
OK_NO_FRAY = "OK_NO_FRAY"
FRAY_DETECTED = "FRAY_DETECTED"
INSPECTION_FAILED = "INSPECTION_FAILED"
TIMING_LABELS = {
    'image_load': 'Image load',
    'yolo_inference': 'YOLO predict (including preprocessing/postprocessing)',
    'mask_processing_dedup': 'YOLO mask conversion/union/dedup',
    'mask_cleanup': 'Mask cleanup/component splitting',
    'geometry': 'Geometry/centerline/rectification',
    'fray_scoring': 'Fray scoring',
    'candidate_filtering': 'Candidate filtering',
    'visualization_save': 'Visualization/debug/image saving',
    'total': 'Total image',
}


@contextmanager
def timed_stage(timings, name, device=None):
    cuda = device is not None and str(device) != 'cpu'
    if cuda:
        torch.cuda.synchronize(int(device))
    started = time.perf_counter()
    try:
        yield
    finally:
        if cuda:
            torch.cuda.synchronize(int(device))
        timings[name] = (timings.get(name) or 0.0) + (time.perf_counter()-started)*1000


def print_environment():
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    info = dict(pytorch=str(torch.__version__), cuda_build=torch.version.cuda,
                cuda_available=available, cuda_device_count=count,
                gpu_names=[torch.cuda.get_device_name(i) for i in range(count)] if available else [])
    print(f"PyTorch version: {info['pytorch']}\nCUDA build: {info['cuda_build']}\n"
          f"CUDA available: {available}\nCUDA device count: {count}\n"
          f"GPU name: {', '.join(info['gpu_names']) or 'NO CUDA'}", flush=True)
    return info


def print_cuda_memory(device, label):
    if str(device) == 'cpu':
        return None
    try:
        index = int(device)
        memory = dict(allocated_bytes=torch.cuda.memory_allocated(index),
                      reserved_bytes=torch.cuda.memory_reserved(index))
        print(f"CUDA memory ({label}): allocated={memory['allocated_bytes']/2**20:.1f} MiB, "
              f"reserved={memory['reserved_bytes']/2**20:.1f} MiB "
              '(PyTorch allocator only; not total Jetson RAM)', flush=True)
        return memory
    except Exception as exc:
        print(f'CUDA memory query unavailable: {exc}', flush=True)
        return dict(error=str(exc))


def print_image_timings(record):
    print('  Stage timings (ms):', flush=True)
    for name, label in TIMING_LABELS.items():
        value = record.get('stage_timings_ms', {}).get(name)
        print(f"    {label}: {value:.2f}" if value is not None else f'    {label}: not reached', flush=True)
    if record.get('yolo_timing_ms'):
        print(f"    Ultralytics breakdown (ms): {record['yolo_timing_ms']}", flush=True)


def aggregate_timings(records):
    aggregated = {}
    for name in TIMING_LABELS:
        values = [r['stage_timings_ms'][name] for r in records
                  if r.get('stage_timings_ms', {}).get(name) is not None]
        if values:
            aggregated[name] = dict(count=len(values), mean=float(np.mean(values)),
                                    median=float(np.median(values)))
    return aggregated

# Downstream constants copied from the preserved reference (not retuned).
STRIP_STEP      = 8
MIN_COMPONENT_PIXELS = 100
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
FRAY_MAX_AREA     = 5000
FRAY_INSTANCE_MIN_AREA = 50
# The legacy area limits are reference areas, not current-image pixel limits.
# 12 px is the median measured half-width (13, 12, 11) in the original
# 1920x1080 three-rope diagnostic. Compare area / half_width**2 with
# 50/144 and 5000/144; this preserves those limits at the reference scale.
FRAY_AREA_REFERENCE_HALF_WIDTH = 12.0
# A saturated score is the shape cue's existing clipped maximum (excess
# >= FRAY_SCORE_PX). A full upper quartile must reach it to stand alone
# when the median is weak; a single high percentile spike is insufficient.
FRAY_SHAPE_SATURATED_FRACTION = 0.25
FRAY_THIN_ONLY    = True
FRAY_THIN_KERNEL  = 9
FRAY_PROTRUDE_PX  = 5
FRAY_WIDTH_GAIN   = 0.35
FRAY_SCORE_PX     = 12
FRAY_WEAK_PROTRUDE_PX = 2.5
FRAY_WEAK_WIDTH_GAIN  = 0.10
FRAY_WEAK_SCORE       = 0.36
FRAY_LOCAL_WIDTH_WINDOW = 24.0  # physical median window in reference rope widths
FRAY_LOCAL_WIDTH_RESIDUAL = 0.18
FRAY_EDGE_V_MIN       = 58
FRAY_EDGE_REL_MIN     = 8
FRAY_EDGE_MIN_AREA    = 6
FRAY_EDGE_MAX_AREA    = 220
DEBUG_MODE = True


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

def rectify_rope(image, mask):
    xs, ys, angles = extract_centerline(mask)
    if xs is None: return None,None,None,None,None
    half_w = measure_rope_half_width(mask, xs, ys)
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

def prepare_fray_image(image):
    """Read-only, image-local calculations shared by all rope components."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    context = dict(gray=gray,
                   ring_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FRAY_BAND_PX, FRAY_BAND_PX)),
                   edge_kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)))
    if USE_FRAY_EDGE:
        blur = cv2.GaussianBlur(gray, (0, 0), 5)
        context['relative_brightness'] = gray.astype(np.int16) - blur.astype(np.int16)
    return context


def local_width_shape_profile(span, valid, xs, excess, reference_width):
    """Complementary widening evidence; never promote an isolated sample.

    Round 24*reference_width/median_sample_spacing to samples, make odd
    (minimum three), then clip to the largest odd length of each contiguous
    valid segment. Nearest-edge padding matches the diagnostic experiment.
    Invalid columns split both the median window and persistence checks.
    """
    trend = np.zeros(len(span), dtype=np.float64)
    residual = np.zeros_like(trend)
    windows = np.zeros(len(span), dtype=np.int32)
    raw = np.zeros(len(span), dtype=bool)
    supported = raw.copy()
    usable = valid & np.isfinite(span) & (span > 0) & np.isfinite(xs)
    spacing = np.diff(xs)
    spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    if not len(spacing) or not np.isfinite(reference_width) or reference_width <= 0:
        return trend, residual, raw, supported, windows
    size = max(3, int(round(FRAY_LOCAL_WIDTH_WINDOW * reference_width / np.median(spacing))))
    if size % 2 == 0:
        size += 1
    bounds = np.flatnonzero(np.diff(np.r_[False, usable, False].astype(np.int8)))
    for start, end in zip(bounds[::2], bounds[1::2]):
        length = end - start
        if length < 3:
            continue
        window = min(size, length if length % 2 else length - 1)
        trend[start:end] = median_filter(span[start:end].astype(np.float64),
                                         size=window, mode='nearest')
        windows[start:end] = window
        np.divide(span[start:end] - trend[start:end], trend[start:end],
                  out=residual[start:end], where=trend[start:end] > 0)
        raw[start:end] = ((excess[start:end] >= FRAY_PROTRUDE_PX)
                          & (residual[start:end] > FRAY_LOCAL_WIDTH_RESIDUAL))
        persistent = apply_run_filter(raw[start:end].copy(),
                                      max(2, FRAY_MIN_RUN), FRAY_GAP_CLOSE)
        # Gap bridging establishes support but cannot promote a sample that
        # fails either the protrusion or residual test. Existing painting
        # still performs its original run/gap filtering afterward.
        supported[start:end] = raw[start:end] & persistent
    return trend, residual, raw, supported, windows


def endpoint_shape_samples(single, full_xs, full_ys, strip_mask, xs, profile,
                           med_upper, med_lower, half_w):
    """Extend shape coverage only; freeze all legacy interior measurements.

    full_xs/full_ys come from rectify_rope (supported skeleton x range and
    in-image vertical strip). Require center support in both original and
    rectified masks, uncensored/bracketing extents, and at least two adjacent
    valid samples. Invalid samples break runs. No new evidence threshold.
    """
    started = time.perf_counter()
    n = len(full_xs)
    interior = (full_xs >= xs[0]) & (full_xs <= xs[-1])
    endpoint = ~interior
    width = np.zeros(n, np.float32)
    upper = np.zeros(n, np.float32)
    lower = np.zeros(n, np.float32)
    valid = np.zeros(n, bool)
    reasons = np.full(n, '', dtype=object)
    old = np.flatnonzero(interior)
    width[old] = profile['width']
    valid[old] = profile['valid']
    excess = np.zeros(n, np.float32)
    excess[old] = profile['excess']
    h, w = single.shape
    win = max(FRAY_BAND_PX + half_w, half_w * 3)
    for i in np.flatnonzero(endpoint):
        x, y = full_xs[i], full_ys[i]
        if not np.isfinite(x) or not np.isfinite(y):
            reasons[i] = 'nonfinite_geometry'; continue
        cx, cy = int(round(x)), int(round(y))
        if not (0 <= cx < w and 0 <= cy < h):
            reasons[i] = 'outside_image'; continue
        if not single[cy, cx]:
            reasons[i] = 'unsupported_center'; continue
        column = strip_mask[:, i] > 0
        if not column[half_w] or np.count_nonzero(column) < 2:
            reasons[i] = 'unsupported_rectified_column'; continue
        y0 = max(0, int(np.floor(y-win))-1)
        y1 = min(h, int(np.ceil(y+win))+2)
        rows = np.flatnonzero(single[y0:y1, cx] > 0) + y0
        rows = rows[np.abs(rows-y) <= win]
        if len(rows) < 2:
            reasons[i] = 'insufficient_extent'; continue
        top, bot = float(rows[0]), float(rows[-1])
        if not (top < y < bot):
            reasons[i] = 'unbracketed_center'; continue
        # A measurement clipped by image/window boundaries is not a reliable
        # endpoint width, even if its center is supported.
        if top == 0 or bot == h-1 or y-top >= win or bot-y >= win:
            reasons[i] = 'censored_extent'; continue
        upper[i], lower[i], width[i] = y-top, bot-y, bot-top+1.0
        excess[i] = max(upper[i]-med_upper, lower[i]-med_lower)
        valid[i] = True
    contiguous = apply_run_filter(valid.copy(), max(2, FRAY_MIN_RUN), 0)
    isolated = endpoint & valid & ~contiguous
    reasons[isolated] = 'isolated_support'
    valid[endpoint] &= contiguous[endpoint]
    trend, residual, _, local, _ = local_width_shape_profile(
        width, valid, full_xs, excess, profile['reference_width'])
    strong = valid & (excess >= FRAY_PROTRUDE_PX) & (
        (width >= profile['reference_width']*(1.0+FRAY_WIDTH_GAIN)) | local)
    weak = valid & (excess >= FRAY_WEAK_PROTRUDE_PX) & (
        width >= profile['reference_width']*(1.0+FRAY_WEAK_WIDTH_GAIN))
    scores = np.where(strong, np.minimum(1.0, excess/FRAY_SCORE_PX),
                      np.where(weak, FRAY_WEAK_SCORE, 0)).astype(np.float32)
    # Interior is context only: its profile/scores are never replaced by the
    # extended median window. Preserve its original strong/weak flags too.
    scores[old] = profile['sample_scores']
    response = scores > 0
    bounds = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(np.int8)))
    filtered = np.zeros(n, bool)
    for start, end in zip(bounds[::2], bounds[1::2]):
        filtered[start:end] = apply_run_filter(response[start:end].copy(),
                                               max(2, FRAY_MIN_RUN), FRAY_GAP_CLOSE)
    admitted = endpoint & valid
    painted = admitted & filtered
    covered = interior | admitted
    diag = dict(endpoint_policy='mask_supported_shape_only',
                old_retained_samples=len(xs), new_retained_samples=int(covered.sum()),
                old_x_range=[float(xs[0]), float(xs[-1])],
                new_x_range=[float(full_xs[covered][0]), float(full_xs[covered][-1])],
                newly_admitted_samples=int(admitted.sum()),
                newly_responding_samples=int(painted.sum()),
                rejected_endpoint_reasons={str(k): int(np.count_nonzero(reasons == k))
                                           for k in np.unique(reasons[endpoint]) if k},
                samples=[dict(x=float(full_xs[i]), y=float(full_ys[i]),
                              admitted=bool(admitted[i]), reason=str(reasons[i]),
                              width=float(width[i]), excess=float(excess[i]),
                              local_residual=float(residual[i]), strong=bool(strong[i]),
                              painted=bool(painted[i]), score=float(scores[i]))
                         for i in np.flatnonzero(endpoint)])
    points = [(int(round(full_xs[i])),
               int(round(full_ys[i]+max(0.0, lower[i]-upper[i])*0.5)),
               float(max(scores[i], 0.35))) for i in np.flatnonzero(painted)]
    diag['processing_ms'] = (time.perf_counter()-started)*1000
    return points, diag


def paint_endpoint_shape(out, shape_out, points, half_w):
    """Max-composite only small endpoint ROIs; never overwrite old evidence."""
    h, w = out.shape
    radius = max(3, half_w)
    for cx, cy, score in points:
        x0, x1 = max(0, cx-radius), min(w, cx+radius+1)
        y0, y1 = max(0, cy-radius), min(h, cy+radius+1)
        if x1 <= x0 or y1 <= y0:
            continue
        blob = np.zeros((y1-y0, x1-x0), np.float32)
        cv2.circle(blob, (cx-x0, cy-y0), radius, score, -1)
        np.maximum(out[y0:y1, x0:x1], blob, out=out[y0:y1, x0:x1])
        if shape_out is not None:
            np.maximum(shape_out[y0:y1, x0:x1], blob, out=shape_out[y0:y1, x0:x1])


def fray_score_for_rope(image, single, xs, ys, half_w, cue_maps=None, image_context=None,
                        endpoint_geometry=None):
    """Detect broken strands sticking OUT of the rope silhouette.
    Uses two cues:
      1) silhouette protrusions already captured by SAM in the rope mask
      2) optional bright thin wisps in a ring just outside the rope mask"""
    endpoint_points = []
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
            # Read only a conservative local window, then retain the exact
            # original floating-point distance test (including boundary ties).
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
            excesses = np.maximum(upper - med_upper, lower - med_lower)
            trend, residual, local_raw, local_strong, windows = local_width_shape_profile(
                span, valid, np.asarray(xs), excesses, med_span)
            global_strong = valid & (excesses >= FRAY_PROTRUDE_PX) & (span >= min_span)
            for i in np.where(valid)[0]:
                excess = max(upper[i] - med_upper, lower[i] - med_lower)
                if (excess >= FRAY_PROTRUDE_PX and span[i] >= min_span) or local_strong[i]:
                    frayed[i] = True
                    fsc[i] = max(fsc[i], min(1.0, excess / FRAY_SCORE_PX))
                elif excess >= FRAY_WEAK_PROTRUDE_PX and span[i] >= weak_span:
                    frayed[i] = True
                    fsc[i] = max(fsc[i], FRAY_WEAK_SCORE)
            if cue_maps is not None:
                cue_maps['shape_profile'] = dict(
                    reference_width=med_span, xs=np.asarray(xs), width=span,
                    valid=valid, excess=excesses, local_trend=trend,
                    local_width_residual=residual, global_strong=global_strong,
                    local_width_raw=local_raw, local_width_strong=local_strong,
                    local_window_samples=windows)

    if USE_FRAY_SHAPE and endpoint_geometry is not None and valid.sum() >= 5:
        # Freeze the old interior reference and response, including local trend.
        baseline_profile = dict(width=span, valid=valid, excess=excesses,
                                reference_width=med_span, sample_scores=fsc.copy())
        endpoint_points, endpoint_diag = endpoint_shape_samples(
            single, *endpoint_geometry, np.asarray(xs), baseline_profile,
            med_upper, med_lower, half_w)
        if cue_maps is not None:
            cue_maps['endpoint_coverage'] = endpoint_diag

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
        rel = context['relative_brightness']
        edge = (((gray >= FRAY_EDGE_V_MIN) & (rel >= FRAY_EDGE_REL_MIN) & (ring > 0)) |
                ((gray >= FRAY_V_MIN) & (ring > 0))).astype(np.uint8) * 255
        edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, context['edge_kernel'])
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
    if endpoint_points:
        paint_endpoint_shape(out, cue_maps['shape'] if cue_maps is not None else None,
                             endpoint_points, half_w)
    return out, wisp, ring

def filter_fray_instances(score_map, fray_region, threshold, shape_score=None,
                          edge_score=None, diagnostics=None, *, rope_geometry):
    """Retain consistent evidence, then suppress nearby duplicate sites.

    Quality is the shape-score upper quartile, not painted area. A strong
    median or a sustained saturated upper quartile can stand alone. Moderate
    upper-quartile evidence still needs bright-strand corroboration. Assign
    each candidate to the nearest retained rope centerline at its centroid;
    that rope's measured half-width supplies its area scale. No image-size,
    coordinate or rope-identity exceptions are used. Missing geometry fails
    explicitly instead of falling back to a fixed-pixel area limit.

    A rejected full component can also qualify through a connected saturated
    shape core. Saturation (1.0) is the existing shape-score ceiling and the
    existing sustained-shape criterion, not a lowered evidence threshold.
    The core independently meets the same rope-relative area limits; weak
    attached pixels do not count against it. Reporting and suppression still
    use the original candidate, not the core box.
    """
    if not rope_geometry:
        raise ValueError('Candidate filtering requires inspected rope geometry')
    for rope in rope_geometry:
        if (not np.isfinite(rope['half_width']) or rope['half_width'] <= 0
                or len(rope['xs']) == 0 or len(rope['xs']) != len(rope['ys'])
                or not np.all(np.isfinite(rope['xs']))
                or not np.all(np.isfinite(rope['ys']))):
            raise ValueError('Invalid rope scale/centerline for candidate filtering')
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
        cx, cy = centroids[lbl]
        rope = min(rope_geometry, key=lambda r: float(np.min(
            (r['xs'] - cx)**2 + (r['ys'] - cy)**2)))
        half_width = float(rope['half_width'])
        scale_squared = (half_width / FRAY_AREA_REFERENCE_HALF_WIDTH)**2
        min_area = FRAY_INSTANCE_MIN_AREA * scale_squared
        max_area = FRAY_MAX_AREA * scale_squared
        saturated_support = int(np.count_nonzero(shape >= 1.0)) if shape is not None else 0
        rec = dict(component_id=lbl, candidate_id=None, bbox=(x, y, bw, bh, area),
                   area=area, cx=float(centroids[lbl, 0]), cy=float(centroids[lbl, 1]),
                   quality=float(np.percentile(representative, 75)),
                   representative_median=float(np.median(representative)),
                   max_score=float(values.max()), mean_score=float(values.mean()),
                   p90_score=float(np.percentile(values, 90)),
                   shape_support=shape_support, edge_support=edge_support,
                   shape_fraction=shape_support / area if shape is not None else None,
                   edge_fraction=edge_support / area if edge is not None else None,
                   rope_component_id=rope['id'], rope_half_width=half_width,
                   normalized_area=area / half_width**2,
                   min_area_pixels=min_area, max_area_pixels=max_area,
                   shape_saturated_support=saturated_support,
                   shape_saturated_fraction=saturated_support / area,
                   strong_floor=strong_floor, retained=False)
        # Analyze only this candidate's shape pixels. No cue regeneration,
        # morphology, or merging of disconnected strong islands is involved.
        cores = []
        if shape is not None:
            core_mask = (mask & (shape_score[y:y+bh, x:x+bw] >= 1.0)).astype(np.uint8)
            core_count, core_labels, core_stats, core_centroids = cv2.connectedComponentsWithStats(
                core_mask, connectivity=8)
            for core_id in range(1, core_count):
                kx, ky, kw, kh, ka = [int(v) for v in core_stats[core_id]]
                core_values = shape_score[y+ky:y+ky+kh, x+kx:x+kx+kw][
                    core_labels[ky:ky+kh, kx:kx+kw] == core_id]
                # At least three connected pixels, even at extremely small
                # scales; at normal rope scales the normalized minimum is
                # already much larger. Disconnected spikes never pool support.
                core_reason = ('too_small_normalized' if ka < max(3, min_area)
                               else 'too_large_normalized' if ka > max_area
                               else 'size_valid')
                cores.append(dict(core_id=core_id, area=ka,
                                  bbox=(x+kx, y+ky, kw, kh),
                                  centroid=(x+float(core_centroids[core_id, 0]),
                                            y+float(core_centroids[core_id, 1])),
                                  fraction=ka / area, rope_half_width=half_width,
                                  normalized_area=ka / half_width**2,
                                  median_score=float(np.median(core_values)),
                                  mean_score=float(np.mean(core_values)),
                                  p75_score=float(np.percentile(core_values, 75)),
                                  max_score=float(np.max(core_values)),
                                  eligible=core_reason == 'size_valid', reason=core_reason))
        valid_cores = [c for c in cores if c['eligible']]
        best_core = max(valid_cores or cores, key=lambda c: (
            c['median_score'], c['p75_score'], c['area'], -c['core_id']), default=None)
        rec.update(strong_core_threshold=1.0, strong_core_count=len(cores),
                   strong_core_eligible_count=len(valid_cores), strong_cores=cores,
                   strong_core_best_area=best_core['area'] if best_core else 0,
                   strong_core_best_bbox=best_core['bbox'] if best_core else None,
                   strong_core_best_fraction=best_core['fraction'] if best_core else 0.0,
                   strong_core_normalized_area=best_core['normalized_area'] if best_core else None,
                   strong_core_representative_score=best_core['median_score'] if best_core else None)
        if area < min_area:
            rec["reason"] = "too_small_normalized"
        elif area > max_area:
            rec["reason"] = "too_large_normalized"
        else:
            candidate_id += 1
            rec["candidate_id"] = candidate_id
            if rec["representative_median"] >= strong_floor:
                rec["reason"] = "strong_shape" if shape is not None else "strong_combined"
                rec["acceptance_test"] = "strong_median"
            elif (shape is not None
                  and rec["shape_saturated_fraction"] >= FRAY_SHAPE_SATURATED_FRACTION
                  and saturated_support >= max(2, min_area)):
                rec["reason"] = "strong_shape"
                rec["acceptance_test"] = "sustained_saturated_upper_quartile"
            elif (shape is not None and edge is not None
                  and rec["quality"] >= strong_floor
                  and shape_support >= min_area
                  and edge_support >= min_area):
                rec["reason"] = "shape_plus_bright"
                rec["acceptance_test"] = "corroborated_upper_quartile"
            else:
                rec["reason"] = "weak_shape" if rec["quality"] < strong_floor else "insufficient_corroboration"
        # Fallback only: existing acceptance paths keep their decisions/tests.
        # A too-large full blob may qualify on a compact core without changing
        # the maximum area limit itself or the original reporting box.
        if (rec['reason'] in {'too_large_normalized', 'weak_shape', 'insufficient_corroboration'}
                and valid_cores):
            rec['pre_core_reason'] = rec['reason']
            rec['reason'] = 'strong_shape'
            rec['acceptance_test'] = 'strong_core_shape'
            if rec['candidate_id'] is None:
                candidate_id += 1
                rec['candidate_id'] = candidate_id
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

    eligible = [r for r in records if r["reason"] in
                {"strong_shape", "strong_combined", "shape_plus_bright"}]
    ranked = sorted(eligible, key=lambda r: (-r["quality"],
                    -r["representative_median"], -r["mean_score"], r["component_id"]))
    selected = []
    for rec in ranked:
        duplicate = next((s for s in selected if nearby(rec, s)), None)
        if duplicate is not None:
            rec["reason"] = "duplicate"
            rec["duplicate_of"] = duplicate["candidate_id"]
        else:
            rec["retained"] = True
            selected.append(rec)
    # Weak satellites do not become independent sites or inflate a kept box.
    for rec in records:
        if rec["reason"] in {"weak_shape", "insufficient_corroboration"}:
            duplicate = next((s for s in selected if nearby(rec, s)), None)
            if duplicate is not None:
                rec["reason"] = "duplicate"
                rec["duplicate_of"] = duplicate["candidate_id"]

    out = np.zeros_like(raw)
    kept = []
    for rec in sorted(selected, key=lambda r: r["bbox"][:2]):
        out[lbl_img == rec["component_id"]] = 255
        kept.append(rec["bbox"])
    if diagnostics is not None:
        diagnostics.extend(records)
    if DEBUG_MODE:
        print(f"  Fray selection: {candidate_id} full/core area-valid candidates -> {len(kept)} sites")
        for rec in records:
            if rec["retained"]:
                print(f"    C{rec['candidate_id']} bbox={rec['bbox'][:4]} area={rec['area']} "
                      f"quality={rec['quality']:.3f} max={rec['max_score']:.3f} "
                      f"mean={rec['mean_score']:.3f} shape={rec['shape_support']} "
                      f"bright={rec['edge_support']} half_width={rec['rope_half_width']:.1f} "
                      f"normalized_area={rec['normalized_area']:.3f} reason={rec['reason']} "
                      f"test={rec.get('acceptance_test')} core_area={rec['strong_core_best_area']}")
    return out, kept

def normalize_rope_mask(mask, image_shape):
    """Match the reference's nearest-neighbor resize and >0.5 binarization."""
    h, w = image_shape[:2]
    resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0.5).astype(np.uint8)


def deduplicate_rope_masks(instances):
    """Confidence-ordered mask IoU suppression; IDs never compare array values."""
    kept, decisions = [], []
    for item in sorted(instances, key=lambda r: (-r['confidence'], r['id'])):
        duplicate = None
        for prior in kept:
            intersection = int(np.count_nonzero(item['mask'] & prior['mask']))
            union = int(np.count_nonzero(item['mask'] | prior['mask']))
            iou = intersection / union if union else 0.0
            if iou > ROPE_DUPLICATE_IOU_THRESHOLD:
                duplicate = (prior['id'], iou)
                break
        record = dict(id=item['id'], confidence=item['confidence'], retained=duplicate is None)
        if duplicate is None:
            kept.append(item)
            record['reason'] = 'distinct_mask'
        else:
            record.update(reason='duplicate_mask', duplicate_of=duplicate[0], iou=duplicate[1])
        decisions.append(record)
    return kept, decisions


def clean_rope_masks(instances, image_shape):
    """Reference get_rope_mask post-prediction behavior, with filter diagnostics."""
    img_h, img_w = image_shape[:2]
    total_pixels = img_h * img_w
    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    decisions = []
    for item in instances:
        m_b = normalize_rope_mask(item['mask'], image_shape)
        record = dict(id=item['id'], area=int(m_b.sum()), retained=False)
        decisions.append(record)
        if m_b.sum()/total_pixels > 0.15:
            record['reason'] = 'area_exceeds_15_percent'
            continue
        contours, _ = cv2.findContours(m_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            record['reason'] = 'empty_mask'
            continue
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        aspect = max(w, h)/(min(w, h)+1e-8)
        record.update(bbox=[x, y, w, h], aspect_ratio=aspect)
        if aspect < 3.0:
            record['reason'] = 'aspect_below_3'
            continue
        combined = np.maximum(combined, m_b)
        record.update(retained=True, reason='accepted')
    if combined.sum() == 0:
        gc.collect()
        return None, decisions
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k)
    gc.collect()
    return combined, decisions


def acquire_rope_masks(model, image, device, half, conf, imgsz, stage_timings=None):
    stage_timings = {} if stage_timings is None else stage_timings
    with timed_stage(stage_timings, 'yolo_inference', device):
        results = model.predict(source=image, imgsz=imgsz, conf=conf, device=device,
                                half=half, retina_masks=True, classes=[ROPE_CLASS_ID],
                                save=False, verbose=False, batch=1)
    with timed_stage(stage_timings, 'mask_processing_dedup'):
        if not results:
            return [], {}
        result = results[0]
        if result.masks is None or result.boxes is None:
            return [], dict(result.speed)
        if len(result.masks.data) != len(result.boxes):
            raise ValueError('YOLO mask/box counts disagree')
        instances = []
        for i, (mask, cls, confidence) in enumerate(zip(
                result.masks.data.cpu().numpy(), result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist())):
            if int(cls) == ROPE_CLASS_ID:
                instances.append(dict(id=i, confidence=float(confidence),
                                      mask=normalize_rope_mask(mask, image.shape)))
        return instances, dict(result.speed)


def inspect_image(model, path, artifact_dir, device, half, conf, imgsz, debug):
    started = time.perf_counter()
    timings = dict.fromkeys(TIMING_LABELS)
    record = dict(filename=str(path), status=INSPECTION_FAILED, raw_yolo_rope_count=0,
                  deduped_rope_count=0, cleaned_component_count=0, usable_rope_count=0,
                  preliminary_fray_candidates=0, final_fray_count=0,
                  components=[], candidates=[], warnings=[], yolo_timing_ms={},
                  stage_timings_ms=timings)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with timed_stage(timings, 'image_load'):
        image = cv2.imread(str(path))
    if image is None:
        record.update(error='Image could not be decoded', total_processing_ms=(time.perf_counter()-started)*1000)
        timings['total'] = record['total_processing_ms']
        (artifact_dir/'diagnostics.json').write_text(json.dumps(record, indent=2))
        return record
    h, w = image.shape[:2]
    raw = np.zeros((h, w), np.uint8)
    cleaned = raw.copy()
    fray_map = raw.copy()
    score = np.zeros((h, w), np.float32)
    shape_score, edge_score = np.zeros_like(score), np.zeros_like(score)
    centerlines = image.copy() if debug else None
    blobs = []
    fray_context = None
    rope_geometry = []
    try:
        instances, timing = acquire_rope_masks(model, image, device, half, conf, imgsz, timings)
        record.update(raw_yolo_rope_count=len(instances), yolo_timing_ms=timing)
        with timed_stage(timings, 'mask_processing_dedup'):
            for item in instances:
                raw = np.maximum(raw, item['mask'])
            if not instances:
                raise ValueError('YOLO returned no rope masks')
            instances, record['rope_deduplication'] = deduplicate_rope_masks(instances)
            record['deduped_rope_count'] = len(instances)
        with timed_stage(timings, 'mask_cleanup'):
            result, record['mask_cleanup'] = clean_rope_masks(instances, image.shape)
            if result is None:
                raise ValueError('All rope masks rejected by cleanup')
            cleaned = result
            ropes = split_ropes(cleaned)
            record['cleaned_component_count'] = len(ropes)
            if not ropes:
                raise ValueError('No usable rope components after cleanup')
        for i, single in enumerate(ropes, 1):
            component = dict(id=i, area=int(single.sum()), bbox=list(cv2.boundingRect(single)),
                             geometry_success=False, inspected=False)
            record['components'].append(component)
            try:
                with timed_stage(timings, 'geometry'):
                    res = rectify_rope(image, single)
                    if res[0] is None:
                        raise ValueError('Centerline/rectification failed')
                    strip_bgr, strip_mask, xs, ys, half_w = res
                    endpoint_geometry = (xs.copy(), ys.copy(), strip_mask)
                    component['sampled_points'] = len(xs)
                    strip_bgr, strip_mask, xs, ys = trim_strip_edges(strip_bgr, strip_mask, xs, ys)
                    n_cols = strip_bgr.shape[1]
                    xs = xs[:n_cols]; ys = ys[:n_cols]
                    if n_cols < 5:
                        raise ValueError('Fewer than five retained centerline samples')
                    component.update(geometry_success=True, retained_points=n_cols, half_width=half_w)
            except Exception as exc:
                component['error'] = f'{type(exc).__name__}: {exc}'
                record['warnings'].append(f'Rope component {i} geometry failed: {exc}')
                continue
            with timed_stage(timings, 'fray_scoring'):
                if fray_context is None:
                    fray_context = prepare_fray_image(image)
                cues = {}
                fs, wisp, ring = fray_score_for_rope(image, single, xs, ys, half_w, cues, fray_context,
                                                    endpoint_geometry=endpoint_geometry)
                if 'endpoint_coverage' in cues:
                    component['endpoint_coverage'] = {
                        k: v for k, v in cues['endpoint_coverage'].items()
                        if debug or k != 'samples'}
                score = np.maximum(score, fs)
                shape_score = np.maximum(shape_score, cues['shape'])
                edge_score = np.maximum(edge_score, cues['edge'])
                profile = cues.get('shape_profile')
                if profile is not None:
                    component['shape_profile_summary'] = dict(
                        reference_width=profile['reference_width'],
                        peak_local_width_residual=float(np.max(profile['local_width_residual'])),
                        global_strong_samples=int(np.count_nonzero(profile['global_strong'])),
                        local_width_strong_samples=int(np.count_nonzero(profile['local_width_strong'])),
                        added_local_strong_samples=int(np.count_nonzero(
                            profile['local_width_strong'] & ~profile['global_strong'])))
            component['inspected'] = True
            rope_geometry.append(dict(id=i, xs=xs, ys=ys, half_width=half_w))
            record['usable_rope_count'] += 1
            if debug:
                with timed_stage(timings, 'visualization_save'):
                    for x, y in zip(xs, ys):
                        cv2.circle(centerlines, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
                    save_image(artifact_dir/f'rope_{i}_mask.png', single*255)
                    save_image(artifact_dir/f'rope_{i}_strip.png', strip_bgr)
                    np.savez(artifact_dir/f'rope_{i}_geometry.npz', xs=xs, ys=ys, half_width=half_w)
                    if profile is not None:
                        np.savez_compressed(artifact_dir/f'rope_{i}_shape_profile.npz', **profile)
            print(f'    rope {i}: geometry OK, {n_cols} retained samples', flush=True)
        if not record['usable_rope_count']:
            raise ValueError('Geometry failed for every rope component')
        record['partial_inspection'] = any(not r['inspected'] for r in record['components'])
        with timed_stage(timings, 'candidate_filtering'):
            region = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                               (FRAY_BAND_PX, FRAY_BAND_PX)))
            fray_map, blobs = filter_fray_instances(score, region, 0.3, shape_score,
                                                    edge_score, record['candidates'],
                                                    rope_geometry=rope_geometry)
        record.update(preliminary_fray_candidates=sum(r['candidate_id'] is not None for r in record['candidates']),
                      preliminary_fray_components=len(record['candidates']),
                      final_fray_count=len(blobs), fray_boxes=blobs,
                      status=FRAY_DETECTED if blobs else OK_NO_FRAY)
    except Exception as exc:
        record.update(status=INSPECTION_FAILED, error=f'{type(exc).__name__}: {exc}')
    save_started = time.perf_counter()
    overlay = image.copy()
    overlay[fray_map > 0] = (0, 0, 255)
    for x, y, bw, bh, area in blobs:
        cv2.rectangle(overlay, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
        cv2.putText(overlay, f'FRAY ({area}px)', (x, max(0, y+bh+16)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
    label = record['status'] + (' - PARTIAL INSPECTION' if record.get('partial_inspection') else '')
    cv2.putText(overlay, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if record['status'] != OK_NO_FRAY else (0, 180, 0), 2)
    try:
        save_image(artifact_dir/'final_overlay.jpg', overlay)
        save_image(artifact_dir/'raw_yolo_rope_union.png', raw*255)
        save_image(artifact_dir/'cleaned_rope_union.png', cleaned*255)
        save_image(artifact_dir/'fray_score.png', (np.clip(score, 0, 1)*255).astype(np.uint8))
        if debug:
            save_image(artifact_dir/'fray_instances.png', fray_map)
            save_image(artifact_dir/'centerlines.jpg', centerlines)
            save_image(artifact_dir/'shape_score.png', (np.clip(shape_score, 0, 1)*255).astype(np.uint8))
            save_image(artifact_dir/'bright_score.png', (np.clip(edge_score, 0, 1)*255).astype(np.uint8))
            np.savez_compressed(artifact_dir/'scores.npz', fray=score, shape=shape_score, edge=edge_score)
    except Exception as exc:
        record.update(status=INSPECTION_FAILED, error=f'Output failure: {exc}')
    timings['visualization_save'] = (timings['visualization_save'] or 0.0) + (time.perf_counter()-save_started)*1000
    record['total_processing_ms'] = (time.perf_counter()-started)*1000
    timings['total'] = record['total_processing_ms']
    record['artifacts'] = str(artifact_dir)
    (artifact_dir/'diagnostics.json').write_text(json.dumps(record, indent=2))
    return record


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write {path}')


def discover_images(inputs, output):
    """Recurse source directories, excluding archived/generated content."""
    found = set()
    generated_suffixes = ('_mask', '_masks', '_overlay', '_prediction', '_fray_score', '_fray_instances')
    for value in inputs:
        source = Path(value).expanduser().resolve()
        allow_rusted = 'rusted' in source.parts
        def allowed(path):
            return (path.suffix.lower() in IMAGE_SUFFIXES
                    and not any(p.lower() in EXCLUDED_DIRECTORIES or p.endswith('.inspection') for p in path.parts)
                    and (allow_rusted or 'rusted' not in path.parts)
                    and not path.stem.lower().endswith(generated_suffixes)
                    and not path.stem.lower().startswith(('mask_', 'overlay_'))
                    and path != output and output not in path.parents)
        if source.is_file():
            if allowed(source):
                found.add(source)
        elif source.is_dir():
            for directory, dirs, files in os.walk(source):
                dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIRECTORIES
                           and not d.endswith('.inspection') and (allow_rusted or d != 'rusted')
                           and (Path(directory)/d).resolve() != output]
                for name in files:
                    path = (Path(directory)/name).resolve()
                    if allowed(path):
                        found.add(path)
        else:
            raise FileNotFoundError(source)
    return sorted(found)


def choose_device(requested):
    if requested == 'cpu':
        return 'cpu', False
    if not torch.cuda.is_available():
        if requested != 'auto':
            raise RuntimeError(f'CUDA unavailable for requested device {requested}')
        print('CUDA unavailable: using CPU FP32 for development/testing.', flush=True)
        return 'cpu', False
    index = 0 if requested in ('auto', 'cuda') else int(requested.removeprefix('cuda:'))
    capability = torch.cuda.get_device_capability(index)
    return str(index), capability >= (5, 3)


def main(argv=None):
    global DEBUG_MODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--input', nargs='+', default=[str(DEFAULT_INPUT)], help='Source directory or one or more source images')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='auto', help='auto, cpu, or CUDA index (e.g. 0)')
    parser.add_argument('--conf', type=float, default=YOLO_CONFIDENCE)
    parser.add_argument('--imgsz', type=int, default=YOLO_IMAGE_SIZE)
    parser.add_argument('--debug', action='store_true', help='Save candidate masks, cue scores and geometry diagnostics')
    args = parser.parse_args(argv)
    if not 0 <= args.conf <= 1 or args.imgsz <= 0:
        parser.error('conf must be between 0 and 1, and imgsz must be positive')
    DEBUG_MODE = args.debug
    output = args.output.expanduser().resolve()
    paths = discover_images(args.input, output)
    if not paths:
        parser.error('No permitted source images found')
    if any(output == p.parent or output in p.parents for p in paths):
        parser.error('Output must not be a source directory or its ancestor')
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(configuration=dict(model=str(args.model.resolve()), imgsz=args.imgsz,
                   confidence=args.conf, duplicate_iou_threshold=ROPE_DUPLICATE_IOU_THRESHOLD,
                   debug=args.debug), images=[],
                   timing_scope='Stages use perf_counter in ms. YOLO predict includes preprocessing/postprocessing; '
                                'Ultralytics sub-stage timings are also recorded. Total includes image/debug saves '
                                'and bookkeeping, excluding final diagnostics/summary JSON serialization. '
                                'Unreached stages are null; aggregate counts include reached stages on failed images.')
    try:
        summary['environment'] = print_environment()
        device, half = choose_device(args.device)
        summary['configuration'].update(device=device, fp16=half)
        print(f'YOLO device: {device}; FP16: {half}. Geometry and fray processing use CPU.', flush=True)
        if not args.model.is_file():
            raise FileNotFoundError(f'Local checkpoint missing: {args.model}')
        model = YOLO(str(args.model.resolve()))
        if model.task != 'segment' or str(model.names.get(ROPE_CLASS_ID, '')).lower() != 'rope':
            raise ValueError('Expected a segmentation checkpoint with class 0 named rope')
        summary['startup_cuda_memory'] = print_cuda_memory(device, 'startup, before first inference')
    except Exception as exc:
        summary['images'] = [dict(filename=str(p), status=INSPECTION_FAILED,
                                  error=f'Model startup failed: {type(exc).__name__}: {exc}') for p in paths]
        write_summary(output, summary)
        print(summary['images'][0]['error'], flush=True)
        return 1
    for i, path in enumerate(paths, 1):
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            relative = Path('external')/Path(*path.parts[1:])
        destination = output/'images'/relative.parent/(relative.name+'.inspection')
        print(f'[{i}/{len(paths)}] {path.name}', flush=True)
        try:
            result = inspect_image(model, path, destination, device, half, args.conf, args.imgsz, args.debug)
        except Exception as exc:
            result = dict(filename=str(path), status=INSPECTION_FAILED,
                          error=f'Inspection/output failure: {type(exc).__name__}: {exc}')
        result['cuda_memory_after_image'] = print_cuda_memory(device, path.name)
        summary['images'].append(result)
        write_summary(output, summary)
        print(f"  {result['status']}: {result.get('final_fray_count', 0)} fray sites", flush=True)
        print_image_timings(result)
    print('\nAggregate timings (ms; reached stages, including failed attempts):', flush=True)
    for name, values in summary['timing_aggregates_ms'].items():
        print(f"  {TIMING_LABELS[name]}: mean={values['mean']:.2f}, "
              f"median={values['median']:.2f}, n={values['count']}", flush=True)
    print(f'Summary: {output / "summary.json"}', flush=True)
    return int(any(r['status'] == INSPECTION_FAILED for r in summary['images']))


def write_summary(output, summary):
    summary['timing_aggregates_ms'] = aggregate_timings(summary['images'])
    temporary = output/'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2))
    temporary.replace(output/'summary.json')


if __name__ == '__main__':
    raise SystemExit(main())
