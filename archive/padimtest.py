import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms

# -----------------------------
# CONFIG
# -----------------------------
images_dir = "/opt/sam2_project/dataset2/images"   # original images
masks_dir  = "/opt/sam2_project/dataset2/masks"    # binary masks
model_path = "/opt/sam2_project/padim_multilayer.pth"
output_dir = "/opt/sam2_project/padim_test_output2"
device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

# This is a raw Mahalanobis distance threshold — tune this after
# looking at score distributions. Start high and lower until you
# get meaningful detections without too many false positives.
THRESHOLD = 4500

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
ckpt    = torch.load(model_path, map_location=device)
mean    = ckpt["mean"]
inv_cov = ckpt["inv_cov"]

# -----------------------------
# BACKBONE (ResNet18)
# -----------------------------
resnet = models.resnet18(pretrained=True).to(device).eval()
layer0 = torch.nn.Sequential(
    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
)
layer1 = resnet.layer1
layer2 = resnet.layer2
layer3 = resnet.layer3
layer4 = resnet.layer4

# -----------------------------
# TRANSFORM — must match training exactly
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ← was missing in your inference
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(x):
    x  = layer0(x)
    f1 = layer1(x)
    f2 = layer2(f1)
    f3 = layer3(f2)
    f4 = layer4(f3)
    return [f1, f2, f3, f4]

# -----------------------------
# PADIM INFERENCE
# Returns RAW Mahalanobis scores (not normalized per-image)
# -----------------------------
def padim_predict(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x   = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        feats    = extract_features(x)
        base_size = feats[0].shape[-2:]
        resized  = [
            F.interpolate(f, size=base_size, mode="bilinear", align_corners=False)
            for f in feats
        ]
        fused = torch.cat(resized, dim=1)[0]  # (C, H, W)

    C, H, W = fused.shape
    feat    = fused.view(C, H * W).T.cpu().numpy()  # (H*W, C)
    diff    = feat - mean                            # (H*W, C)
    scores  = np.sum(diff @ inv_cov * diff, axis=1) # (H*W,) Mahalanobis²
    heatmap = scores.reshape(H, W)

    return heatmap, H, W

# -----------------------------
# RUN TEST
# -----------------------------
mask_files = [
    f for f in os.listdir(masks_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
print(f"Testing on {len(mask_files)} images")

score_log = []  # collect per-image max scores to help you tune THRESHOLD

for i, mask_name in enumerate(mask_files):
    base_name = mask_name.replace("_mask", "")
    img_path  = os.path.join(images_dir, base_name)
    mask_path = os.path.join(masks_dir, mask_name)

    if not os.path.exists(img_path):
        print(f"⚠️  No matching image for {mask_name}, skipping.")
        continue

    image = cv2.imread(img_path)
    mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        continue

    # Raw Mahalanobis heatmap
    heatmap, H_feat, W_feat = padim_predict(image)

    # Resize mask to feature map size
    mask_resized = cv2.resize(mask, (W_feat, H_feat), interpolation=cv2.INTER_NEAREST)
    mask_bool    = mask_resized > 127

    # Zero out background — only score rope pixels
    heatmap_gated = heatmap * mask_bool.astype(np.float32)

    # Threshold on raw scores
    anomaly_map = (heatmap_gated > THRESHOLD).astype(np.uint8) * 255

    # Resize outputs back to original image size
    h_orig, w_orig = image.shape[:2]
    anomaly_map_full = cv2.resize(anomaly_map, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # For visualization: normalize only the rope region for the heatmap image
    rope_scores = heatmap[mask_bool]
    if rope_scores.size > 0:
        vmin, vmax = rope_scores.min(), rope_scores.max()
        heatmap_vis = np.clip((heatmap - vmin) / (vmax - vmin + 1e-8), 0, 1)
        heatmap_vis = (heatmap_vis * mask_bool) * 255  # black out background
    else:
        heatmap_vis = np.zeros_like(heatmap)

    heatmap_vis_full = cv2.resize(
        heatmap_vis.astype(np.uint8), (w_orig, h_orig)
    )

    # Overlay anomalies in red on original image
    overlay = image.copy()
    overlay[anomaly_map_full > 0] = [0, 0, 255]

    # Log max rope score to help calibrate THRESHOLD
    max_score = rope_scores.max() if rope_scores.size > 0 else 0
    score_log.append((base_name, max_score))

    # Save
    name = os.path.splitext(base_name)[0]
    cv2.imwrite(f"{output_dir}/{name}_heatmap.png", heatmap_vis_full)
    cv2.imwrite(f"{output_dir}/{name}_anomaly_mask.png", anomaly_map_full)
    cv2.imwrite(f"{output_dir}/{name}_overlay.png", overlay)

    print(f"[{i+1}/{len(mask_files)}] {base_name} — max rope score: {max_score:.1f}")

# -----------------------------
# THRESHOLD CALIBRATION HELPER
# -----------------------------
print("\n📊 Score summary (use this to tune THRESHOLD):")
scores_only = [s for _, s in score_log]
print(f"   Min max-score across images : {min(scores_only):.1f}")
print(f"   Mean max-score across images: {np.mean(scores_only):.1f}")
print(f"   Max max-score across images : {max(scores_only):.1f}")
print(f"\n   Current THRESHOLD = {THRESHOLD}")
print(f"   Suggestion: set THRESHOLD between mean and max if images are normal.")
print(f"               set THRESHOLD below mean if images contain defects.")

print("\n✅ Done testing PaDiM")
