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
#  (kept your original keys, added new ones)
# ─────────────────────────────────────────────
test_dir        = "/opt/sam2_project/input_images2"
normal_dir      = "/opt/sam2_project/input_images"   # NEW: rust-free training images
model_path      = "/opt/sam2_project/patchcore_rope.pkl"  # NEW: replaces padim .pth
output_dir      = "/opt/sam2_project/pipeline_b_output"
device          = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE        = 256
PATCH_SIZE      = 224    # fed into PatchCore feature extractor
STRIP_STEP      = 2      # sample a strip every N pixels along centerline
STRIP_HALF_W    = 28     # fallback only — overridden dynamically per image (see measure_rope_half_width)
CORESET_RATIO   = 0.10   # fraction of embeddings kept in memory bank
HSV_WEIGHT      = 0.35   # contribution of colour prior to fused score
PC_WEIGHT       = 0.65   # contribution of PatchCore to fused score
RUST_THRESHOLD  = 0.50   # fused score above this -> rust
MIN_RUST_AREA   = 50     # ignore blobs smaller than this (pixels)
TEXT_PROMPT     = ["thin wire rope"]

os.makedirs(output_dir, exist_ok=True)

# ─────────────────────────────────────────────
#  SAM3  (unchanged from your original)
# ─────────────────────────────────────────────
print("Loading SAM3...")
overrides = dict(
    conf=0.5, task="segment", mode="predict",
    model="sam3.pt", device="cuda", half=True, save=False,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# ─────────────────────────────────────────────
#  RESNET BACKBONE
#  CHANGED: wide_resnet50_2 + layer2+layer3 only
#  (PatchCore works best on mid-level features)
# ─────────────────────────────────────────────
resnet = models.resnet18(pretrained=True).to("cpu").eval()
layer0 = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
layer1 = resnet.layer1
layer2 = resnet.layer2
layer3 = resnet.layer3

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_features(x: torch.Tensor):
    """Returns fused layer2+layer3 features as (H*W, C) numpy array."""
    x = x.to("cpu")   # force CPU
    with torch.no_grad():
        x  = layer0(x)
        f1 = layer1(x)
        f2 = layer2(f1)
        f3 = layer3(f2)
    base = f2.shape[-2:]
    f3r  = F.interpolate(f3, size=base, mode="bilinear", align_corners=False)
    fused = torch.cat([f2, f3r], dim=1)[0]          # (C, H, W)
    C, H, W = fused.shape
    return fused.permute(1, 2, 0).reshape(H * W, C).cpu().numpy(), H, W


# ═══════════════════════════════════════════
#  STAGE 1 - SEGMENTATION  (your original)
# ═══════════════════════════════════════════
def get_rope_mask(img_path: str, image: np.ndarray):
    """Returns combined binary mask (0/1) for all rope strands."""
    img_h, img_w = image.shape[:2]
    total_pixels = img_h * img_w

    predictor.set_image(img_path)
    results = predictor(text=TEXT_PROMPT)

    if not results or results[0].masks is None:
        print("  SAM3 found no ropes.")
        return None

    sam_masks = results[0].masks.data.cpu().numpy()
    print(f"  SAM3 found {len(sam_masks)} mask(s).")

    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    for m in sam_masks:
        m_resized = cv2.resize(m.astype(np.float32), (img_w, img_h),
                               interpolation=cv2.INTER_NEAREST)
        m_binary  = (m_resized > 0.5).astype(np.uint8)

        area_ratio = m_binary.sum() / total_pixels
        if area_ratio > 0.15:
            print(f"  Rejected: area {area_ratio:.2%} (too large)")
            continue

        contours, _ = cv2.findContours(m_binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        aspect = max(w, h) / (min(w, h) + 1e-8)
        if aspect < 3.0:
            print(f"  Rejected: aspect {aspect:.1f} (too blob-like)")
            continue

        combined = np.maximum(combined, m_binary)

    if combined.sum() == 0:
        print("  All masks rejected.")
        return None

    # replace the last few lines of get_rope_mask, before return combined
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)

    # FREE GPU memory immediately after SAM3 is done
    torch.cuda.empty_cache()
    gc.collect()

    return combined


# ═══════════════════════════════════════════
#  STAGE 2 - CENTERLINE EXTRACTION  (NEW)
# ═══════════════════════════════════════════
def extract_centerline(mask: np.ndarray):
    """
    Skeletonizes the rope mask then fits a smoothing spline so we get
    a clean parametric centerline free of skeleton noise.

    Returns xs, ys (centerline coords) and tangent_angles (degrees).
    """
    skel = skeletonize(mask > 0).astype(np.uint8)
    ys_raw, xs_raw = np.where(skel)

    if len(xs_raw) < 10:
        # fallback: bounding-box midline
        ys_box, xs_box = np.where(mask > 0)
        order = np.argsort(xs_box)
        xs_raw, ys_raw = xs_box[order], ys_box[order]

    order  = np.argsort(xs_raw)
    xs_raw = xs_raw[order].astype(float)
    ys_raw = ys_raw[order].astype(float)

    # One y per unique x (median to handle skeleton branches)
    xs_u, inv = np.unique(xs_raw.astype(int), return_inverse=True)
    ys_u = np.array([np.median(ys_raw[inv == i]) for i in range(len(xs_u))])

    spline = UnivariateSpline(xs_u, ys_u, k=3, s=len(xs_u) * 5)
    deriv  = spline.derivative()

    xs = np.arange(xs_u[0], xs_u[-1], STRIP_STEP, dtype=float)
    ys = spline(xs)
    tangent_angles = np.degrees(np.arctan(deriv(xs)))

    return xs, ys, tangent_angles


# ═══════════════════════════════════════════
#  STAGE 2b — DYNAMIC ROPE WIDTH MEASUREMENT  (NEW FIX)
# ═══════════════════════════════════════════
def measure_rope_half_width(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> int:
    widths = []
    step = max(1, len(xs) // 30)
    for cx, cy in zip(xs[::step], ys[::step]):
        cx, cy = int(round(cx)), int(round(cy))
        if cx < 0 or cx >= mask.shape[1]:
            continue
        col = mask[:, cx]
        rope_rows = np.where(col > 0)[0]
        if len(rope_rows) < 2:
            continue
        # FIX: only look at rows within 40px of this centreline point
        # prevents measuring across both ropes when combined mask is used
        local = rope_rows[np.abs(rope_rows - cy) <= 40]
        if len(local) >= 2:
            widths.append(int(local[-1] - local[0]))

    if not widths:
        print("  [Width] Fallback to 10px half-width")
        return 10

    median_w = int(np.median(widths))
    half_w   = max(4, median_w // 2 + 2)
    print(f"  [Width] Rope width ~{median_w}px → using half-width={half_w}px")
    return half_w


# ═══════════════════════════════════════════
#  STAGE 3 - ROPE RECTIFICATION  (NEW)
# ═══════════════════════════════════════════
def rectify_rope(image: np.ndarray, mask: np.ndarray):
    """
    Samples perpendicular cross-sections along the centreline and stacks
    them into a flat strip.  Eliminates helical-twist drift.

    FIX: half_w is now measured dynamically so we only sample actual rope
    pixels, not background above/below the rope.

    Returns strip_bgr (H, N, 3), strip_mask (H, N), xs, ys, half_w.
    """
    xs, ys, angles = extract_centerline(mask)
    half_w  = measure_rope_half_width(mask, xs, ys)   # dynamic width
    h, w    = image.shape[:2]
    strip_h = half_w * 2

    strips_bgr, strips_mask = [], []

    for cx, cy, ang in zip(xs, ys, angles):
        cx, cy = int(round(cx)), int(round(cy))
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rot_img  = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        rot_mask = cv2.warpAffine(mask.astype(np.float32), M, (w, h),
                                  flags=cv2.INTER_NEAREST)
        y0, y1 = cy - half_w, cy + half_w
        if y0 < 0 or y1 > h:
            continue
        col_bgr  = rot_img [y0:y1, cx:cx+1]
        col_mask = rot_mask[y0:y1, cx:cx+1]
        if col_bgr.shape[0] != strip_h:
            continue
        strips_bgr .append(col_bgr)
        strips_mask.append(col_mask)

    if not strips_bgr:
        return None, None, xs, ys, half_w

    return (np.concatenate(strips_bgr,  axis=1),
            np.concatenate(strips_mask, axis=1),
            xs, ys, half_w)


# ═══════════════════════════════════════════
#  STAGE 4 - HSV / LAB RUST FILTER  (NEW)
# ═══════════════════════════════════════════
def hsv_rust_map(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Float32 heatmap [0..1] where 1 = strong rust colour signal.
    Covers orange/red rust, brown oxidation, and grey desaturated scale.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

    m1 = cv2.inRange(hsv, (0,  40,  30), (20,  255, 200))  # orange-red
    m2 = cv2.inRange(hsv, (5,  25,  20), (25,  180, 150))  # brown
    m3 = cv2.inRange(hsv, (0,   0,  30), (180,  30, 140))  # grey oxide
    m4 = cv2.inRange(lab, (0, 133, 128), (255, 200, 200))  # LAB red-yellow

    rust = cv2.bitwise_or(m1, cv2.bitwise_or(m2, cv2.bitwise_or(m3, m4)))
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rust = cv2.morphologyEx(rust, cv2.MORPH_OPEN,   k)
    rust = cv2.morphologyEx(rust, cv2.MORPH_DILATE, k, iterations=2)

    return rust.astype(np.float32) / 255.0


# ═══════════════════════════════════════════
#  STAGE 5 - PATCHCORE  (replaces PaDiM)
# ═══════════════════════════════════════════
class PatchCore:
    """
    Lightweight PatchCore.

    Key difference from PaDiM:
      - No spatial correspondence assumed.
      - Anomaly score = distance to nearest-neighbour in memory bank.
      - Trained on NORMAL images only — no rust labels needed.
    """

    def __init__(self):
        self.memory_bank = None
        self.nn_index    = None

    def fit(self, strips: list):
        print(f"  [PatchCore] Building memory bank from {len(strips)} strips...")
        all_embeddings = []
        for strip in strips:
            rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
            x   = transform(rgb).unsqueeze(0).to(device)
            emb, _, _ = extract_features(x)
            all_embeddings.append(emb)

        bank = np.concatenate(all_embeddings, axis=0)
        bank = self._coreset_subsample(bank)
        self.memory_bank = bank.astype(np.float32)
        self.nn_index    = NearestNeighbors(n_neighbors=1,
                                            algorithm="ball_tree",
                                            n_jobs=-1)
        self.nn_index.fit(self.memory_bank)
        print(f"  [PatchCore] Memory bank: {self.memory_bank.shape[0]:,} "
              f"vectors (dim={self.memory_bank.shape[1]})")

    def _coreset_subsample(self, bank: np.ndarray) -> np.ndarray:
        """Greedy k-centre coreset — keeps CORESET_RATIO fraction."""
        n_keep = max(1, int(len(bank) * CORESET_RATIO))
        if n_keep >= len(bank):
            return bank
        print(f"  [PatchCore] Coreset: {len(bank):,} -> {n_keep:,} vectors...")
        rng      = np.random.default_rng(42)
        idx      = [int(rng.integers(len(bank)))]
        min_dist = np.full(len(bank), np.inf)
        for _ in range(n_keep - 1):
            d        = np.linalg.norm(bank - bank[idx[-1]], axis=1)
            min_dist = np.minimum(min_dist, d)
            idx.append(int(np.argmax(min_dist)))
        return bank[idx]

    def score(self, strip_bgr: np.ndarray):
        """Anomaly heatmap (H_feat, W_feat) float32. Higher = more anomalous."""
        if self.memory_bank is None:
            raise RuntimeError("PatchCore not trained. Call fit() first.")
        rgb = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB)
        x   = transform(rgb).unsqueeze(0).to(device)
        emb, H, W = extract_features(x)
        dists, _  = self.nn_index.kneighbors(emb)
        return dists[:, 0].reshape(H, W).astype(np.float32)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"memory_bank": self.memory_bank}, f)
        print(f"  [PatchCore] Saved -> {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.memory_bank = data["memory_bank"]
        self.nn_index    = NearestNeighbors(n_neighbors=1,
                                            algorithm="ball_tree",
                                            n_jobs=-1)
        self.nn_index.fit(self.memory_bank)
        print(f"  [PatchCore] Loaded {self.memory_bank.shape[0]:,} vectors from {path}")


# ─────────────────────────────────────────────
#  TRAINING  (run once on normal images)
# ─────────────────────────────────────────────
def train(patchcore: PatchCore):
    normal_files = [
        f for f in os.listdir(normal_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"\n[Train] Found {len(normal_files)} normal images in {normal_dir}")
    normal_strips = []

    for fname in normal_files:
        img_path = os.path.join(normal_dir, fname)
        image    = cv2.imread(img_path)
        if image is None:
            continue
        print(f"  Processing {fname}")
        rope_mask = get_rope_mask(img_path, image)
        if rope_mask is None:
            continue
        strip_bgr, _, _, _, _ = rectify_rope(image, rope_mask)   # unpack 5-tuple
        if strip_bgr is None:
            print("  Rectification failed, skipping.")
            continue
        normal_strips.append(strip_bgr)

    if not normal_strips:
        raise RuntimeError(
            "No usable normal strips found. Check normal_dir and SAM3 detection."
        )
    patchcore.fit(normal_strips)
    patchcore.save(model_path)
    print("\n Training complete.")


# ─────────────────────────────────────────────
#  STAGE 6 - SCORE FUSION  (NEW)
# ─────────────────────────────────────────────
def fuse_scores(hsv_map: np.ndarray, pc_map: np.ndarray) -> np.ndarray:
    """
    Normalise both maps to [0,1] then weighted sum.
    HSV catches obvious rust colour.
    PatchCore catches early-stage texture change before colour shifts.
    """
    pc_min, pc_max = pc_map.min(), pc_map.max()
    pc_norm = (pc_map - pc_min) / (pc_max - pc_min + 1e-8)
    return HSV_WEIGHT * hsv_map.astype(np.float32) + PC_WEIGHT * pc_norm


# ─────────────────────────────────────────────
#  MAP STRIP SCORES BACK TO ORIGINAL IMAGE
# ─────────────────────────────────────────────
def map_strip_score_to_image(fused_strip: np.ndarray,
                              xs: np.ndarray,
                              ys: np.ndarray,
                              image_shape: tuple,
                              rope_mask: np.ndarray,
                              half_w: int) -> np.ndarray:
    """
    Maps each strip column score back to its centreline point (xs[i], ys[i]).

    FIX 1: splat radius is now half_w (matches actual rope thickness) instead
            of the old hardcoded STRIP_HALF_W // 2.
    FIX 2: scores are masked to rope_mask pixels only — stops scores bleeding
            into the background and causing false detections off the rope.
    """
    h, w   = image_shape[:2]
    out    = np.zeros((h, w), dtype=np.float32)
    n_cols = min(fused_strip.shape[1], len(xs))
    r      = max(2, half_w // 2)   # splat radius scales with rope thickness

    for i in range(n_cols):
        col_score = float(fused_strip[:, i].max())
        cx = int(round(xs[i]))
        cy = int(round(ys[i]))
        if 0 <= cx < w and 0 <= cy < h:
            x0, x1 = max(0, cx - r), min(w, cx + r + 1)
            y0, y1 = max(0, cy - r), min(h, cy + r + 1)
            out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], col_score)

    # FIX 2: zero out anything outside the rope mask
    out = out * (rope_mask > 0).astype(np.float32)
    return out


# ─────────────────────────────────────────────
#  MAIN INFERENCE LOOP
# ─────────────────────────────────────────────
def run_inference(patchcore: PatchCore):
    test_files = [
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"\n[Inference] Found {len(test_files)} test images.")
    score_log = []

    for i, fname in enumerate(test_files):
        img_path = os.path.join(test_dir, fname)
        image    = cv2.imread(img_path)
        if image is None:
            continue
        h_orig, w_orig = image.shape[:2]
        print(f"\n[{i+1}/{len(test_files)}] {fname}")

        # Stage 1 — Segment
        rope_mask = get_rope_mask(img_path, image)
        if rope_mask is None:
            print("  Skipping — no rope mask.")
            continue
        print(f"  Rope pixels: {rope_mask.sum():,}")

        # Stage 2+3 — Centerline + Rectify
        strip_bgr, strip_mask, xs, ys, half_w = rectify_rope(image, rope_mask)
        if strip_bgr is None:
            print("  Skipping — rectification failed.")
            continue
        print(f"  Rectified strip: {strip_bgr.shape[1]} cols x {strip_bgr.shape[0]}px")

        # Stage 4 — HSV colour filter
        hsv_strip = hsv_rust_map(strip_bgr)          # (strip_h, N)

        # Stage 5 — PatchCore
        pc_heatmap = patchcore.score(strip_bgr)      # (H_feat, W_feat)
        pc_strip   = cv2.resize(pc_heatmap,
                                (strip_bgr.shape[1], strip_bgr.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

        # Stage 6 — Fuse
        fused_strip = fuse_scores(hsv_strip, pc_strip)   # (strip_h, N) [0,1]
        score_img   = map_strip_score_to_image(
                          fused_strip, xs, ys, image.shape,
                          rope_mask, half_w)              # FIX: pass mask + half_w
        score_img   = cv2.GaussianBlur(score_img, (9, 9), 0)

        # Threshold + remove tiny blobs
        rust_map = (score_img > RUST_THRESHOLD).astype(np.uint8) * 255
        n_lbl, lbl_img, stats, _ = cv2.connectedComponentsWithStats(rust_map)
        for lbl in range(1, n_lbl):
            if stats[lbl, cv2.CC_STAT_AREA] < MIN_RUST_AREA:
                rust_map[lbl_img == lbl] = 0

        max_score = float(score_img[rope_mask > 0].max()) if rope_mask.sum() > 0 else 0.0
        score_log.append((fname, max_score))

        # Visualise
        rope_mask_u8 = cv2.resize(rope_mask, (w_orig, h_orig),
                                  interpolation=cv2.INTER_NEAREST)
        rope_vals = score_img[rope_mask_u8 > 0]
        if rope_vals.size:
            vmin, vmax = rope_vals.min(), rope_vals.max()
            score_vis  = np.clip((score_img - vmin) / (vmax - vmin + 1e-8), 0, 1)
        else:
            score_vis = score_img.copy()
        score_u8  = (score_vis * rope_mask_u8 * 255).astype(np.uint8)
        heatmap_c = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)

        overlay = image.copy()
        overlay[rope_mask_u8 > 0] = cv2.addWeighted(
            image, 0.5, heatmap_c, 0.5, 0
        )[rope_mask_u8 > 0]
        overlay[rust_map > 0] = [0, 0, 255]

        # Bounding boxes on rust regions
        n_lbl2, lbl_img2, stats2, _ = cv2.connectedComponentsWithStats(rust_map)
        for lbl in range(1, n_lbl2):
            x, y, bw, bh, area = stats2[lbl]
            if area >= MIN_RUST_AREA:
                cv2.rectangle(overlay, (x, y), (x+bw, y+bh), (0, 255, 255), 2)
                cv2.putText(overlay, f"RUST ({area}px)",
                            (x, max(0, y-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Debug strip (raw + fused heatmap stacked)
        strip_vis = np.concatenate([
            cv2.resize(strip_bgr, (strip_bgr.shape[1], 64)),
            cv2.applyColorMap(
                (fused_strip * 255).astype(np.uint8), cv2.COLORMAP_HOT
            )[:64]
        ], axis=0)

        name = os.path.splitext(fname)[0]
        cv2.imwrite(f"{output_dir}/{name}_rope_mask.png", rope_mask_u8 * 255)
        cv2.imwrite(f"{output_dir}/{name}_heatmap.png",   score_u8)
        cv2.imwrite(f"{output_dir}/{name}_overlay.png",   overlay)
        cv2.imwrite(f"{output_dir}/{name}_strip.png",     strip_vis)

        status = "RUST DETECTED" if rust_map.sum() > 0 else "CLEAN"
        print(f"  Max fused score: {max_score:.3f}  (threshold: {RUST_THRESHOLD})  [{status}]")

    # Summary
    if score_log:
        scores_only = [s for _, s in score_log]
        rust_count  = sum(1 for _, s in score_log if s > RUST_THRESHOLD)
        print(f"\n Score summary (fused 0-1):")
        print(f"   Min  : {min(scores_only):.3f}")
        print(f"   Mean : {np.mean(scores_only):.3f}")
        print(f"   Max  : {max(scores_only):.3f}")
        print(f"   Threshold : {RUST_THRESHOLD}")
        print(f"   Flagged   : {rust_count}/{len(score_log)} images")

    print("\nDone.")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    patchcore = PatchCore()

    if not os.path.exists(model_path):
        print("No trained model found — running training phase first.")
        train(patchcore)
    else:
        patchcore.load(model_path)

    run_inference(patchcore)