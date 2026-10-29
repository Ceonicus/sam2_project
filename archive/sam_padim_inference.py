import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics.models.sam import SAM3SemanticPredictor

# -----------------------------
# CONFIG
# -----------------------------
test_dir   = "/opt/sam2_project/input_images2"
model_path = "/opt/sam2_project/padim_multilayer.pth"
output_dir = "/opt/sam2_project/padim_test_output2"
device     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 256
THRESHOLD  = 4000
TEXT_PROMPT = ["thin wire rope"]  # tweak if SAM3 misses ropes

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD PADIM MODEL
# -----------------------------
ckpt    = torch.load(model_path, map_location=device)
mean    = ckpt["mean"]
inv_cov = ckpt["inv_cov"]

# -----------------------------
# LOAD SAM3
# -----------------------------
print("Loading SAM3...")
overrides = dict(
    conf=0.5,
    task="segment",
    mode="predict",
    model="sam3.pt",
    device="cuda",
    half=True,
    save=False,   # we handle saving ourselves
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# -----------------------------
# LOAD RESNET BACKBONE
# -----------------------------
resnet = models.resnet18(pretrained=True).to(device).eval()
layer0 = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
layer1 = resnet.layer1
layer2 = resnet.layer2
layer3 = resnet.layer3
layer4 = resnet.layer4

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_features(x):
    x  = layer0(x)
    f1 = layer1(x)
    f2 = layer2(f1)
    f3 = layer3(f2)
    f4 = layer4(f3)
    return [f1, f2, f3, f4]

# -----------------------------
# STEP 1: SEGMENT ROPE WITH SAM3
# -----------------------------
def get_rope_mask(img_path, image):
    img_h, img_w = image.shape[:2]
    total_pixels = img_h * img_w

    predictor.set_image(img_path)
    results = predictor(text=TEXT_PROMPT)

    if not results or results[0].masks is None:
        print("  ⚠️  SAM3 found no ropes.")
        return None

    sam_masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
    print(f"  SAM3 found {len(sam_masks)} mask(s).")

    combined = np.zeros((img_h, img_w), dtype=np.uint8)

    for m in sam_masks:
        m_resized = cv2.resize(
            m.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST
        )
        m_binary = (m_resized > 0.5).astype(np.uint8)

        # Filter 1: reject masks covering too much of the image
        area_ratio = m_binary.sum() / total_pixels
        if area_ratio > 0.15:
            print(f"  🚫 Rejected mask: area ratio {area_ratio:.2%} (too large)")
            continue

        # Filter 2: reject blob-shaped masks — ropes must be elongated
        contours, _ = cv2.findContours(m_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        aspect = max(w, h) / (min(w, h) + 1e-8)
        if aspect < 3.0:
            print(f"  🚫 Rejected mask: aspect ratio {aspect:.1f} (too blob-like)")
            continue

        combined = np.maximum(combined, m_binary)

    if combined.sum() == 0:
        print("  ⚠️  All SAM3 masks rejected by filters.")
        return None

    # Clean up noise — wide flat kernel suits rope shape
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)

    return combined  # binary 0/1

# -----------------------------
# STEP 2: PADIM SCORING
# -----------------------------
def padim_score(image, rope_mask):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x   = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        feats     = extract_features(x)
        base_size = feats[0].shape[-2:]
        resized   = [F.interpolate(f, size=base_size, mode="bilinear", align_corners=False) for f in feats]
        fused     = torch.cat(resized, dim=1)[0]

    C, H_feat, W_feat = fused.shape
    feat    = fused.view(C, H_feat * W_feat).T.cpu().numpy()
    diff    = feat - mean
    scores  = np.sum(diff @ inv_cov * diff, axis=1)
    heatmap = scores.reshape(H_feat, W_feat)

    # Resize mask to feature map size and gate background
    mask_feat     = cv2.resize(rope_mask, (W_feat, H_feat), interpolation=cv2.INTER_NEAREST).astype(bool)
    heatmap_gated = heatmap * mask_feat.astype(np.float32)

    return heatmap_gated, H_feat, W_feat

# -----------------------------
# PROCESS TEST IMAGES
# -----------------------------
test_files = [
    f for f in os.listdir(test_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
print(f"Found {len(test_files)} test images.")

score_log = []

for i, fname in enumerate(test_files):
    img_path = os.path.join(test_dir, fname)
    image    = cv2.imread(img_path)
    if image is None:
        continue

    print(f"\n[{i+1}/{len(test_files)}] {fname}")

    # Segment rope with SAM3
    rope_mask = get_rope_mask(img_path, image)
    if rope_mask is None:
        print("  ⚠️  Could not segment rope. Skipping.")
        continue

    rope_pixel_count = rope_mask.sum()
    print(f"  Rope pixels: {rope_pixel_count:,}")

    # PaDiM scoring
    heatmap_gated, H_feat, W_feat = padim_score(image, rope_mask)

    h_orig, w_orig = image.shape[:2]

    # Anomaly mask at original resolution
    anomaly_map      = (heatmap_gated > THRESHOLD).astype(np.uint8) * 255
    anomaly_map_full = cv2.resize(anomaly_map, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # Heatmap visualization — only rope region, normalized within rope
    mask_feat   = cv2.resize(rope_mask, (W_feat, H_feat), interpolation=cv2.INTER_NEAREST).astype(bool)
    rope_scores = heatmap_gated[mask_feat]
    if rope_scores.size > 0:
        vmin, vmax  = rope_scores.min(), rope_scores.max()
        heatmap_vis = np.clip((heatmap_gated - vmin) / (vmax - vmin + 1e-8), 0, 1)
        heatmap_vis = (heatmap_vis * mask_feat) * 255
    else:
        heatmap_vis = np.zeros_like(heatmap_gated)

    heatmap_vis_full = cv2.resize(heatmap_vis.astype(np.uint8), (w_orig, h_orig))

    # JET colormap overlay on rope region
    heatmap_color = cv2.applyColorMap(heatmap_vis_full, cv2.COLORMAP_JET)
    rope_mask_full = cv2.resize(rope_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy()
    overlay[rope_mask_full > 0] = cv2.addWeighted(
        image, 0.5, heatmap_color, 0.5, 0
    )[rope_mask_full > 0]

    # Hard red on confirmed anomalies
    overlay[anomaly_map_full > 0] = [0, 0, 255]

    max_score = rope_scores.max() if rope_scores.size > 0 else 0
    score_log.append((fname, max_score))

    name = os.path.splitext(fname)[0]
    cv2.imwrite(f"{output_dir}/{name}_rope_mask.png", rope_mask_full * 255)
    cv2.imwrite(f"{output_dir}/{name}_heatmap.png",   heatmap_vis_full)
    cv2.imwrite(f"{output_dir}/{name}_overlay.png",   overlay)

    print(f"  Max rope score: {max_score:.1f}  (threshold: {THRESHOLD})")

# -----------------------------
# SCORE SUMMARY
# -----------------------------
if score_log:
    scores_only = [s for _, s in score_log]
    print(f"\n📊 Score summary:")
    print(f"   Min : {min(scores_only):.1f}")
    print(f"   Mean: {np.mean(scores_only):.1f}")
    print(f"   Max : {max(scores_only):.1f}")
    print(f"   Threshold: {THRESHOLD}")

print("\n✅ Done.")
