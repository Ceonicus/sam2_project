import os
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
import torch.nn.functional as F
from ultralytics import YOLO, SAM
from torchvision import models, transforms

# -----------------------------
# CONFIG
# -----------------------------
yolo_weights = "/workspace/accordwireseg3.pt"
sam_weights = "/workspace/sam3.pt"
padim_weights = "/workspace/padim_multilayer.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD = 0.5

LOWER_RUST = np.array([10, 80, 50])
UPPER_RUST = np.array([25, 255, 255])

output_dir = "/workspace/output"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading models...")

yolo_model = YOLO(yolo_weights)
sam_model = SAM(sam_weights)

ckpt = torch.load(padim_weights, map_location=device)
mean = ckpt["mean"]
inv_cov = ckpt["inv_cov"]

resnet = models.resnet18(pretrained=True).to(device).eval()

layer0 = torch.nn.Sequential(
    resnet.conv1,
    resnet.bn1,
    resnet.relu,
    resnet.maxpool
)
layer1 = resnet.layer1
layer2 = resnet.layer2
layer3 = resnet.layer3
layer4 = resnet.layer4

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(x):
    x = layer0(x)
    f1 = layer1(x)
    f2 = layer2(f1)
    f3 = layer3(f2)
    f4 = layer4(f3)
    return [f1, f2, f3, f4]

# -----------------------------
# PADIM
# -----------------------------
def padim_predict(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = extract_features(x)

        base_size = feats[0].shape[-2:]
        resized = [
            F.interpolate(f, size=base_size, mode="bilinear", align_corners=False)
            for f in feats
        ]

        fused = torch.cat(resized, dim=1)[0]

    C, H, W = fused.shape
    feat = fused.view(C, H * W).T.cpu().numpy()

    diff = feat - mean
    scores = np.sum(diff @ inv_cov * diff, axis=1)

    heatmap = scores.reshape(H, W)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap

# -----------------------------
# REALSENSE INIT
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

print("Camera started. Press 'q' to quit, 's' to save frame.")

# -----------------------------
# MAIN LOOP
# -----------------------------
frame_id = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        display = image.copy()

        # -----------------------------
        # YOLO
        # -----------------------------
        yolo_results = yolo_model.predict(image, device=device, conf=0.2, iou=0.2, verbose=False)
        boxes = yolo_results[0].boxes.xyxy.tolist()

        if boxes:
            # -----------------------------
            # SAM
            # -----------------------------
            sam_results = sam_model.predict(
                image,
                bboxes=boxes,
                device=device,
                retina_masks=False,
                verbose=False
            )

            result = sam_results[0]

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                rope_mask = np.any(masks, axis=0).astype(np.uint8) * 255

                # -----------------------------
                # APPLY MASK
                # -----------------------------
                masked = cv2.bitwise_and(image, image, mask=rope_mask)

                # -----------------------------
                # PADIM
                # -----------------------------
                heatmap = padim_predict(masked)

                anomaly_mask = (heatmap > THRESHOLD).astype(np.uint8) * 255
                anomaly_mask = cv2.resize(anomaly_mask, (image.shape[1], image.shape[0]))

                # -----------------------------
                # RUST FILTER
                # -----------------------------
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                rust_mask = cv2.inRange(hsv, LOWER_RUST, UPPER_RUST)

                final_rust = cv2.bitwise_and(anomaly_mask, rust_mask)

                # -----------------------------
                # CLEANUP
                # -----------------------------
                kernel = np.ones((5,5), np.uint8)
                final_rust = cv2.morphologyEx(final_rust, cv2.MORPH_OPEN, kernel)
                final_rust = cv2.morphologyEx(final_rust, cv2.MORPH_CLOSE, kernel)

                # -----------------------------
                # OVERLAY
                # -----------------------------
                display[final_rust > 0] = [0, 0, 255]

        # -----------------------------
        # DISPLAY
        # -----------------------------
        cv2.imshow("Rust Detection", display)

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            break

        # Save frame
        if key == ord('s'):
            cv2.imwrite(f"{output_dir}/frame_{frame_id}.png", display)
            cv2.imwrite(f"{output_dir}/debug_rope_mask.png", rope_mask)
            cv2.imwrite(f"{output_dir}/debug_anomaly.png", anomaly_mask)
            cv2.imwrite(f"{output_dir}/debug_rust.png", final_rust)
            print(f"Saved frame {frame_id}")
            frame_id += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
