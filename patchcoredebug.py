import os
import gc
import cv2
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics.models.sam import SAM3SemanticPredictor
from skimage.morphology import skeletonize
from scipy.interpolate import UnivariateSpline
from sklearn.neighbors import NearestNeighbors

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
test_dir        = "/opt/sam2_project/input_images2"
normal_dir      = "/opt/sam2_project/input_images"
model_path      = "/opt/sam2_project/patchcore_rope.pkl"
output_dir      = "/opt/sam2_project/pipeline_b_output"
device          = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE      = 224
STRIP_STEP      = 8
STRIP_HALF_W    = 28
CORESET_RATIO   = 0.10
HSV_WEIGHT      = 0.7          # unused while rust-only; kept for the future fray path
PC_WEIGHT       = 0.3          # unused while rust-only
RUST_THRESHOLD  = 0.5
MIN_RUST_AREA   = 150          # image-level backstop (painted-blob pixels)
TEXT_PROMPT     = ["thin wire rope"]
MIN_COMPONENT_PIXELS = 100     # ignore connected-component fragments smaller than this

# ── Rust column-filter knobs (the real tuning levers) ──
COL_FRAC   = 0.10   # a column counts as rusty if >= this fraction of it is rust
MIN_RUN    = 1      # keep only runs of >= this many consecutive rusty columns
GAP_CLOSE  = 4      # bridge gaps of <= this many columns inside a run

# ── FILTER TOGGLES (set False to disable a filter for A/B testing) ──
#   All False  = "before filters" baseline (HSV-only, every rusty column kept).
#   Turn ON one at a time and watch the photo10/11/12 blob lines to see which
#   filter is costing you rust recall.
USE_GLINT_REJECT  = True    # the "not brighter than rope median" test in HSV
USE_COLUMN_FILTER = True    # the per-column / run-length / gap logic
USE_AREA_FILTER   = True   # the MIN_RUST_AREA blob filter

# ── DEBUG CONFIG ──────────────────────────────
DEBUG_MODE      = True
PERCENTILE_FLAG = 90
# ─────────────────────────────────────────────

os.makedirs(output_dir, exist_ok=True)
if DEBUG_MODE:
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

print(f"Loading SAM3 on {device}...")
overrides = dict(
    conf=0.5, task="segment", mode="predict",
    model="sam3.pt", device=device, half=(device=="cuda"), save=False,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

resnet = models.resnet18(pretrained=True).to(device).eval()
layer0 = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
layer1 = resnet.layer1
layer2 = resnet.layer2
layer3 = resnet.layer3

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def extract_features(x):
    x = x.to(device)
    with torch.no_grad():
        x = layer0(x); f1 = layer1(x); f2 = layer2(f1); f3 = layer3(f2)
    base = f2.shape[-2:]
    f3r  = F.interpolate(f3, size=base, mode="bilinear", align_corners=False)
    fused = torch.cat([f2, f3r], dim=1)[0]
    C,H,W = fused.shape
    return fused.permute(1,2,0).reshape(H*W,C).cpu().numpy(), H, W

def get_rope_mask(img_path, image):
    img_h, img_w = image.shape[:2]
    total_pixels = img_h * img_w
    predictor.set_image(img_path)
    try:
        results = predictor(text=TEXT_PROMPT)
    except RuntimeError as e:
        if "CUDA" in str(e) or "NVML" in str(e):
            print("  CUDA OOM — retrying...")
            torch.cuda.empty_cache(); gc.collect(); torch.cuda.synchronize()
            results = predictor(text=TEXT_PROMPT)
        else:
            raise
    if not results or results[0].masks is None:
        print("  SAM3 found no ropes.")
        torch.cuda.empty_cache(); gc.collect(); return None
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
        torch.cuda.empty_cache(); gc.collect(); return None
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k)
    torch.cuda.empty_cache(); gc.collect()
    return combined

def split_ropes(rope_mask):
    """Split a combined rope mask into individual ropes (connected components)."""
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
        order = np.argsort(xs_box)
        xs_raw, ys_raw = xs_box[order], ys_box[order]
    order  = np.argsort(xs_raw)
    xs_raw = xs_raw[order].astype(float)
    ys_raw = ys_raw[order].astype(float)
    xs_u, inv = np.unique(xs_raw.astype(int), return_inverse=True)
    ys_u = np.array([np.median(ys_raw[inv==i]) for i in range(len(xs_u))])
    if len(xs_u) < 4:
        return None, None, None
    spline = UnivariateSpline(xs_u, ys_u, k=3, s=len(xs_u)*5)
    deriv  = spline.derivative()
    xs = np.arange(xs_u[0], xs_u[-1], STRIP_STEP, dtype=float)
    if len(xs) < 2:
        return None, None, None
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
    median_w = int(np.median(widths))
    half_w   = max(4, median_w//2+2)
    print(f"  [Width] Rope width ~{median_w}px → half-width={half_w}px")
    return half_w

def rectify_rope(image, mask):
    xs, ys, angles = extract_centerline(mask)
    if xs is None:
        return None,None,None,None,None
    half_w = measure_rope_half_width(mask, xs, ys)
    h, w   = image.shape[:2]
    strip_h = half_w*2
    strips_bgr, strips_mask = [], []
    kept_xs, kept_ys = [], []
    for cx,cy,ang in zip(xs,ys,angles):
        cx,cy = int(round(cx)), int(round(cy))
        M = cv2.getRotationMatrix2D((cx,cy),ang,1.0)
        ri = cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        rm = cv2.warpAffine(mask.astype(np.float32),M,(w,h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
        rm = (rm > 0.3).astype(np.float32)
        y0,y1 = cy-half_w, cy+half_w
        if y0<0 or y1>h: continue
        cb = ri[y0:y1,cx:cx+1]; cm = rm[y0:y1,cx:cx+1]
        if cb.shape[0] != strip_h: continue
        strips_bgr.append(cb); strips_mask.append(cm)
        kept_xs.append(cx); kept_ys.append(cy)
    if not strips_bgr:
        return None,None,None,None,half_w
    return (np.concatenate(strips_bgr,axis=1),
            np.concatenate(strips_mask,axis=1),
            np.array(kept_xs, dtype=float),
            np.array(kept_ys, dtype=float),
            half_w)

def trim_strip_edges(strip_bgr, strip_mask, xs, ys):
    n = strip_bgr.shape[1]
    trim = max(4, n//33)
    if n - 2*trim < 1:
        return strip_bgr, strip_mask, xs, ys
    return (strip_bgr[:,trim:n-trim], strip_mask[:,trim:n-trim],
            xs[trim:len(xs)-trim], ys[trim:len(ys)-trim])

def hsv_rust_map(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1,1), dtype=np.float32)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Warm + saturated (orange rust). Bounds tuned from sampled pixels.
    color = (H >= 8) & (H <= 22) & (S >= 55) & (V >= 100)

    # Rust is matte, NOT brighter than the surrounding clean wire — glints are.
    if USE_GLINT_REJECT:
        valid = V[V > 30]
        rope_median_v = float(np.median(valid)) if valid.size else 128.0
        # only treat a pixel as a glint if it's bright AND desaturated.
        # bright-but-saturated pixels are real rust, so keep them.
        not_glint = ~((V > rope_median_v + 15) & (S < 90))
    else:
        not_glint = np.ones_like(V, dtype=bool)

    rust = (color & not_glint).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    rust = cv2.morphologyEx(rust, cv2.MORPH_OPEN, k)
    rust = cv2.morphologyEx(rust, cv2.MORPH_DILATE, k, iterations=1)
    return rust.astype(np.float32) / 255.0

class PatchCore:
    def __init__(self):
        self.memory_bank = None; self.nn_index = None
        self.pc_ref_lo = 0.0; self.pc_ref_hi = 1.0
    def fit(self, strips):
        print(f"  [PatchCore] Building from {len(strips)} strips...")
        all_emb = []
        for s in strips:
            rgb = cv2.cvtColor(s,cv2.COLOR_BGR2RGB)
            x   = transform(rgb).unsqueeze(0).to(device)
            emb,_,_ = extract_features(x)
            all_emb.append(emb)
        bank = np.concatenate(all_emb,axis=0)
        bank = self._coreset(bank)
        self.memory_bank = bank.astype(np.float32)
        self.nn_index = NearestNeighbors(n_neighbors=1,algorithm="ball_tree",n_jobs=-1)
        self.nn_index.fit(self.memory_bank)
        print(f"  [PatchCore] Bank: {self.memory_bank.shape[0]:,} x {self.memory_bank.shape[1]}")
        # NOTE: bank-vs-itself reference (≈0) is a placeholder, unused in the rust-only
        # path. Fix when building the fray path: score held-out clean strips, not the bank.
        train_dists, _ = self.nn_index.kneighbors(self.memory_bank)
        d = train_dists[:, 0]
        self.pc_ref_lo = float(np.percentile(d, 50))
        self.pc_ref_hi = float(np.percentile(d, 99))
        print(f"  [PatchCore] clean PC ref (placeholder): lo={self.pc_ref_lo:.3f} hi={self.pc_ref_hi:.3f}")
    def _coreset(self, bank):
        n_keep = max(1,int(len(bank)*CORESET_RATIO))
        if n_keep >= len(bank): return bank
        rng = np.random.default_rng(42)
        idx = [int(rng.integers(len(bank)))]
        min_dist = np.full(len(bank),np.inf)
        for _ in range(n_keep-1):
            d = np.linalg.norm(bank-bank[idx[-1]],axis=1)
            min_dist = np.minimum(min_dist,d)
            idx.append(int(np.argmax(min_dist)))
        return bank[idx]
    def score(self, strip_bgr):
        rgb = cv2.cvtColor(strip_bgr,cv2.COLOR_BGR2RGB)
        x   = transform(rgb).unsqueeze(0).to(device)
        emb,H,W = extract_features(x)
        dists,_ = self.nn_index.kneighbors(emb)
        return dists[:,0].reshape(H,W).astype(np.float32)
    def save(self,path):
        with open(path,"wb") as f:
            pickle.dump({"memory_bank": self.memory_bank,
                         "pc_ref_lo": self.pc_ref_lo,
                         "pc_ref_hi": self.pc_ref_hi}, f)
        print(f"  [PatchCore] Saved -> {path}")
    def load(self,path):
        with open(path,"rb") as f: data=pickle.load(f)
        self.memory_bank = data["memory_bank"]
        self.pc_ref_lo   = data.get("pc_ref_lo", 0.0)
        self.pc_ref_hi   = data.get("pc_ref_hi", 1.0)
        self.nn_index = NearestNeighbors(n_neighbors=1,algorithm="ball_tree",n_jobs=-1)
        self.nn_index.fit(self.memory_bank)
        print(f"  [PatchCore] Loaded {self.memory_bank.shape[0]:,} vectors  "
              f"(PC ref lo={self.pc_ref_lo:.3f} hi={self.pc_ref_hi:.3f})")

def train(patchcore):
    files = [f for f in os.listdir(normal_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]
    print(f"\n[Train] Found {len(files)} normal images")
    strips = []
    for fname in files:
        img_path = os.path.join(normal_dir,fname)
        image = cv2.imread(img_path)
        if image is None: continue
        print(f"  Processing {fname}")
        rope_mask = get_rope_mask(img_path,image)
        if rope_mask is None: continue
        ropes = split_ropes(rope_mask)
        print(f"    {len(ropes)} rope component(s)")
        for ri, single in enumerate(ropes):
            s,sm,xs,ys,hw = rectify_rope(image,single)
            if s is None: continue
            s,sm,xs,ys = trim_strip_edges(s,sm,xs,ys)
            if s.shape[1] < 5: continue
            strips.append(s)
    if not strips: raise RuntimeError("No strips found.")
    patchcore.fit(strips)
    patchcore.save(model_path)
    print("\nTraining complete.")

def fuse_scores(hsv_map, pc_map):
    # RUST PATH ONLY. PatchCore is parked until fray test images exist.
    # (When frays are added, OR a normalized PC term back in via np.maximum.)
    return hsv_map.astype(np.float32)

def map_strip_score_to_image(fused_strip, xs, ys, image_shape, rope_mask, half_w,
                             min_run=MIN_RUN, col_frac=COL_FRAC, gap_close=GAP_CLOSE):
    """Paint per-column rust scores onto the image.

    When USE_COLUMN_FILTER is True, only columns belonging to a real rust RUN
    survive (stray single columns dropped, small internal gaps bridged).
    When False, every rusty column is painted (the "before filters" baseline)."""
    h, w = image_shape[:2]
    out = np.zeros((h, w), dtype=np.float32)
    n = min(fused_strip.shape[1], len(xs))

    # 1. per-column rusty flag + score
    rusty = np.zeros(n, dtype=bool)
    csc   = np.zeros(n, dtype=np.float32)
    for i in range(n):
        col = fused_strip[:, i]
        if float((col > 0.5).mean()) >= col_frac:
            rusty[i] = True
            csc[i]   = float(col.max())

    if USE_COLUMN_FILTER:
        # 2. bridge tiny gaps so one weak column doesn't split a real patch
        i = 0
        while i < n:
            if not rusty[i]:
                j = i
                while j < n and not rusty[j]:
                    j += 1
                if i > 0 and j < n and (j - i) <= gap_close:   # gap flanked by rust
                    rusty[i:j] = True
                i = j
            else:
                i += 1

        # 3. drop runs shorter than min_run
        i = 0
        while i < n:
            if rusty[i]:
                j = i
                while j < n and rusty[j]:
                    j += 1
                if (j - i) < min_run:
                    rusty[i:j] = False
                i = j
            else:
                i += 1
    # else: leave rusty[] untouched — every rusty column is kept

    # 4. paint surviving columns
    r = max(2, half_w // 2)
    for i in range(n):
        if not rusty[i]:
            continue
        cx, cy = int(round(xs[i])), int(round(ys[i]))
        if 0 <= cx < w and 0 <= cy < h:
            out[max(0,cy-r):min(h,cy+r+1), max(0,cx-r):min(w,cx+r+1)] = np.maximum(
                out[max(0,cy-r):min(h,cy+r+1), max(0,cx-r):min(w,cx+r+1)], csc[i])
    return out * (rope_mask > 0).astype(np.float32)

# ─────────────────────────────────────────────
#  DEBUG HELPERS
# ─────────────────────────────────────────────
def norm_to_u8(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)

def save_debug_strip(name, strip_bgr, hsv_strip, pc_strip, fused_strip):
    """4-row composite: raw / HSV / PatchCore / fused.
    NOTE: each row is contrast-stretched, so a faint signal still looks bright.
    Trust the printed score table for absolute levels, not this image."""
    TARGET_H = 48
    def make_row(arr_float, colormap=None):
        u8 = norm_to_u8(arr_float)
        if u8.ndim == 2:
            u8 = cv2.applyColorMap(u8, colormap) if colormap else cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        return cv2.resize(u8, (strip_bgr.shape[1], TARGET_H))
    raw_row    = cv2.resize(strip_bgr, (strip_bgr.shape[1], TARGET_H))
    hsv_avg    = hsv_strip.mean(axis=0, keepdims=True)
    hsv_row    = make_row(np.tile(hsv_avg, (TARGET_H, 1)), cv2.COLORMAP_HOT)
    pc_resized = cv2.resize(pc_strip, (strip_bgr.shape[1], TARGET_H))
    pc_row     = cv2.applyColorMap(norm_to_u8(pc_resized), cv2.COLORMAP_JET)
    fused_avg  = fused_strip.mean(axis=0, keepdims=True)
    fused_row  = make_row(np.tile(fused_avg, (TARGET_H, 1)), cv2.COLORMAP_HOT)
    def label(img, text):
        out = img.copy()
        cv2.putText(out, text, (4, TARGET_H-6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
        return out
    composite = np.vstack([
        label(raw_row,  "1. Raw strip"),
        label(hsv_row,  "2. HSV rust map"),
        label(pc_row,   "3. PatchCore map"),
        label(fused_row,"4. Fused score"),
    ])
    cv2.imwrite(os.path.join(debug_dir, f"{name}_DEBUG_strip.png"), composite)

def print_score_table(name, hsv_strip, pc_strip, fused_strip, rope_scores):
    def pct(arr, p):
        if arr is None or arr.size == 0: return 0.0
        return float(np.percentile(arr, p))
    print(f"\n  ┌─ Score table for {name} ─────────────────────────────")
    print(f"  │           {'min':>7}  {'p50':>7}  {'p90':>7}  {'p95':>7}  {'max':>7}")
    print(f"  │  HSV    {pct(hsv_strip,0):>7.4f}  {pct(hsv_strip,50):>7.4f}  {pct(hsv_strip,90):>7.4f}  {pct(hsv_strip,95):>7.4f}  {pct(hsv_strip,100):>7.4f}")
    print(f"  │  PC     {pct(pc_strip,0):>7.4f}  {pct(pc_strip,50):>7.4f}  {pct(pc_strip,90):>7.4f}  {pct(pc_strip,95):>7.4f}  {pct(pc_strip,100):>7.4f}")
    print(f"  │  Fused  {pct(fused_strip,0):>7.4f}  {pct(fused_strip,50):>7.4f}  {pct(fused_strip,90):>7.4f}  {pct(fused_strip,95):>7.4f}  {pct(fused_strip,100):>7.4f}")
    print(f"  │  Rope   {pct(rope_scores,0):>7.4f}  {pct(rope_scores,50):>7.4f}  {pct(rope_scores,90):>7.4f}  {pct(rope_scores,95):>7.4f}  {pct(rope_scores,100):>7.4f}")
    print(f"  └──────────────────────────────────────────────────────")

# ─────────────────────────────────────────────
#  MAIN INFERENCE LOOP
# ─────────────────────────────────────────────
def run_inference(patchcore):
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".png",".jpg",".jpeg"))]
    print(f"\n[Inference] Found {len(test_files)} test images.")
    print(f"[Filters] glint_reject={USE_GLINT_REJECT}  column_filter={USE_COLUMN_FILTER}  "
          f"area_filter={USE_AREA_FILTER}  (MIN_RUN={MIN_RUN} GAP_CLOSE={GAP_CLOSE} "
          f"COL_FRAC={COL_FRAC} MIN_RUST_AREA={MIN_RUST_AREA})")
    score_log = []

    for i, fname in enumerate(test_files):
        img_path = os.path.join(test_dir, fname)
        image    = cv2.imread(img_path)
        if image is None: continue
        h_orig, w_orig = image.shape[:2]
        name = os.path.splitext(fname)[0]
        print(f"\n[{i+1}/{len(test_files)}] {fname}")

        rope_mask = get_rope_mask(img_path, image)
        if rope_mask is None:
            print("  Skipping — no rope mask."); continue

        ropes = split_ropes(rope_mask)
        if not ropes:
            print("  Skipping — no usable rope component."); continue
        print(f"  {len(ropes)} rope component(s)")

        score_img = np.zeros((h_orig, w_orig), dtype=np.float32)
        any_rope  = False
        last_vis  = None

        for comp_i, single in enumerate(ropes):
            print(f"  ── Rope component {comp_i+1}/{len(ropes)} ──")
            strip_bgr, strip_mask, xs, ys, half_w = rectify_rope(image, single)
            if strip_bgr is None:
                print("    rectification failed — skipping component."); continue
            print(f"    Strip before trim: {strip_bgr.shape[1]} cols x {strip_bgr.shape[0]}px")
            print(f"    strip_mask nonzero: {(strip_mask > 0).sum()} / {strip_mask.size} pixels")

            strip_bgr, strip_mask, xs, ys = trim_strip_edges(strip_bgr, strip_mask, xs, ys)
            print(f"    Strip after trim:  {strip_bgr.shape[1]} cols")

            MAX_COLS = 2000
            if strip_bgr.shape[1] > MAX_COLS:
                idx = np.linspace(0, strip_bgr.shape[1]-1, MAX_COLS, dtype=int)
                strip_bgr  = strip_bgr[:,idx]; strip_mask = strip_mask[:,idx]
                xs = xs[idx]; ys = ys[idx]
                print(f"    [Cap] Subsampled to {MAX_COLS} cols")

            n_cols = strip_bgr.shape[1]
            xs = xs[:n_cols]; ys = ys[:n_cols]
            if n_cols < 5:
                print("    Too few columns — skipping component."); continue

            hsv_strip  = hsv_rust_map(strip_bgr)
            pc_heatmap = patchcore.score(strip_bgr)
            pc_strip   = cv2.resize(pc_heatmap,(strip_bgr.shape[1],strip_bgr.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)

            fused_strip = fuse_scores(hsv_strip, pc_strip)
            comp_score  = map_strip_score_to_image(fused_strip, xs, ys, image.shape, single, half_w)
            score_img   = np.maximum(score_img, comp_score)
            any_rope    = True

            if DEBUG_MODE:
                comp_rope_scores = comp_score[single > 0]
                print_score_table(f"{name} (rope {comp_i+1})",
                                   hsv_strip, pc_strip, fused_strip, comp_rope_scores)
                save_debug_strip(f"{name}_rope{comp_i+1}", strip_bgr, hsv_strip, pc_strip, fused_strip)

            fused_u8 = (np.clip(fused_strip, 0, 1) * 255).astype(np.uint8)
            last_vis = np.concatenate([
                cv2.resize(strip_bgr, (strip_bgr.shape[1], 64)),
                cv2.resize(cv2.applyColorMap(fused_u8, cv2.COLORMAP_HOT),
                           (strip_bgr.shape[1], 64))
            ], axis=0)

        if not any_rope:
            print("  Skipping — no component produced a strip."); continue

        rope_mask_u8 = cv2.resize(rope_mask,(w_orig,h_orig),interpolation=cv2.INTER_NEAREST)
        rope_scores  = score_img[rope_mask_u8 > 0]

        # ── THRESHOLD + (optional) AREA FILTER on the SHARP map (no pre-blur) ──
        threshold = RUST_THRESHOLD
        rust_map = ((score_img > threshold) & (rope_mask_u8 > 0)).astype(np.uint8) * 255
        n_lbl, lbl_img, stats, _ = cv2.connectedComponentsWithStats(rust_map)
        if USE_AREA_FILTER:
            for lbl in range(1, n_lbl):
                if stats[lbl, cv2.CC_STAT_AREA] < MIN_RUST_AREA:
                    rust_map[lbl_img == lbl] = 0

        # report exactly what survived — the real flag/no-flag truth
        area_floor = MIN_RUST_AREA if USE_AREA_FILTER else 1
        kept = [int(stats[l, cv2.CC_STAT_AREA]) for l in range(1, n_lbl)
                if stats[l, cv2.CC_STAT_AREA] >= area_floor]
        print(f"  Surviving anomaly blobs: {len(kept)}  sizes={kept}")

        max_score = float(rope_scores.max()) if rope_scores.size > 0 else 0.0
        score_log.append((fname, max_score))

        # ── visualization only: blur a separate copy for a smooth heatmap ──
        score_vis_src = cv2.GaussianBlur(score_img, (9,9), 0)
        rope_vals = score_vis_src[rope_mask_u8 > 0]
        if rope_vals.size and rope_vals.max() > 0:
            vmin, vmax = rope_vals.min(), rope_vals.max()
            score_vis = np.clip((score_vis_src-vmin)/(vmax-vmin+1e-8),0,1)
        else:
            score_vis = score_vis_src
        score_u8  = (score_vis * (rope_mask_u8 > 0) * 255).astype(np.uint8)
        heatmap_c = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)

        overlay = image.copy()
        overlay[rope_mask_u8 > 0] = cv2.addWeighted(image,0.5,heatmap_c,0.5,0)[rope_mask_u8 > 0]
        overlay[rust_map > 0] = [0, 255, 255]
        for lbl in range(1, n_lbl):
            x,y,bw,bh,area = stats[lbl]
            if area >= area_floor and (rust_map[lbl_img == lbl].max() if (lbl_img == lbl).any() else 0) > 0:
                cv2.rectangle(overlay,(x,y),(x+bw,y+bh),(0,255,255),2)
                cv2.putText(overlay,f"ANOMALY ({area}px)",(x,max(0,y-6)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

        cv2.imwrite(f"{output_dir}/{name}_rope_mask.png", rope_mask_u8*255)
        cv2.imwrite(f"{output_dir}/{name}_heatmap.png",   score_u8)
        cv2.imwrite(f"{output_dir}/{name}_overlay.png",   overlay)
        if last_vis is not None:
            cv2.imwrite(f"{output_dir}/{name}_strip.png", last_vis)

        status = "FLAGGED" if len(kept) > 0 else "CLEAN"
        print(f"  Max score: {max_score:.4f}  threshold: {threshold:.4f}  [{status}]")

    if score_log:
        scores_only = [s for _,s in score_log]
        print(f"\nScore summary:")
        print(f"   Min  : {min(scores_only):.4f}")
        print(f"   Mean : {np.mean(scores_only):.4f}")
        print(f"   Max  : {max(scores_only):.4f}")
    print("\nDone.")

if __name__ == "__main__":
    patchcore = PatchCore()
    if not os.path.exists(model_path):
        print("No model found — training first.")
        train(patchcore)
    else:
        patchcore.load(model_path)
    run_inference(patchcore)