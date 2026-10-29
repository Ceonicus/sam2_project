# import os
# import cv2
# import torch
# import numpy as np
# from torchvision import models, transforms
# from torch.utils.data import Dataset, DataLoader

# # -----------------------------
# # CONFIG
# # -----------------------------
# train_dir = "/workspace/dataset/masked"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# IMG_SIZE = 256

# # -----------------------------
# # TRANSFORM
# # -----------------------------
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
# ])

# # -----------------------------
# # DATASET
# # -----------------------------
# class RopeDataset(Dataset):
#     def __init__(self, folder):
#         self.files = [
#             os.path.join(folder, f)
#             for f in os.listdir(folder)
#             if f.lower().endswith((".png", ".jpg", ".jpeg"))
#         ]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         img = cv2.imread(self.files[idx])
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = transform(img)
#         return img

# dataset = RopeDataset(train_dir)
# loader = DataLoader(dataset, batch_size=8, shuffle=False)

# print(f"Training samples: {len(dataset)}")

# # -----------------------------
# # BACKBONE (ResNet18)
# # -----------------------------
# model = models.resnet18(pretrained=True)
# model.eval().to(device)

# feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])

# # -----------------------------
# # FEATURE EXTRACTION
# # -----------------------------
# features = []

# with torch.no_grad():
#     for batch in loader:
#         batch = batch.to(device)
#         f = feature_extractor(batch)  # (B, C, H, W)
#         features.append(f.cpu().numpy())

# features = np.concatenate(features, axis=0)

# print("Feature shape:", features.shape)

# # -----------------------------
# # PADIM STATISTICS
# # -----------------------------
# N, C, H, W = features.shape

# features = features.reshape(N, C, H * W)
# features = np.transpose(features, (0, 2, 1))  # (N, HW, C)

# pixel_features = features.reshape(-1, C)

# mean = np.mean(pixel_features, axis=0)
# cov = np.cov(pixel_features, rowvar=False)

# cov += np.eye(C) * 0.01  # stability
# inv_cov = np.linalg.inv(cov)

# # -----------------------------
# # SAVE MODEL
# # -----------------------------
# padim_model = {
#     "mean": mean,
#     "inv_cov": inv_cov,
# }

# torch.save(padim_model, "/workspace/padim_model.pth")

# print("✅ PaDiM training complete")

import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms

# -----------------------------
# CONFIG
# -----------------------------
images_dir = "/opt/sam2_project/dataset/images"   # original unmasked images
masks_dir  = "/opt/sam2_project/dataset/masks"    # binary masks
device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
MIN_VALID_PIXELS = 500  # minimum masked pixels to bother processing an image

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# LOAD BACKBONE (ResNet18)
# -----------------------------
resnet = models.resnet18(pretrained=True)
resnet.eval().to(device)

layer0 = torch.nn.Sequential(
    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
)
layer1 = resnet.layer1
layer2 = resnet.layer2
layer3 = resnet.layer3
layer4 = resnet.layer4

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
# LOAD FILE LIST
# -----------------------------
mask_files = [
    f for f in os.listdir(masks_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
print(f"Found {len(mask_files)} masks.")

all_pixel_features = []  # collect masked pixel features across all images

# -----------------------------
# FEATURE EXTRACTION LOOP
# -----------------------------
with torch.no_grad():
    for mask_name in mask_files:
        base_name = mask_name.replace("_mask", "")
        img_path  = os.path.join(images_dir, base_name)
        mask_path = os.path.join(masks_dir, mask_name)

        if not os.path.exists(img_path):
            print(f"⚠️  No matching image for {mask_name}, skipping.")
            continue

        image = cv2.imread(img_path)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"⚠️  Failed to read files for {mask_name}, skipping.")
            continue

        # Sanity check — enough rope pixels in this mask?
        if np.sum(mask > 127) < MIN_VALID_PIXELS:
            print(f"⚠️  Mask too sparse for {mask_name}, skipping.")
            continue

        # -----------------------------
        # FEED ORIGINAL IMAGE TO RESNET
        # -----------------------------
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor  = transform(img_rgb).unsqueeze(0).to(device)  # (1, 3, IMG_SIZE, IMG_SIZE)

        feats = extract_features(tensor)

        # Resize all feature maps to layer1 spatial size
        base_size = feats[0].shape[-2:]  # (H_feat, W_feat)
        fused_list = []
        for f in feats:
            f_resized = torch.nn.functional.interpolate(
                f, size=base_size, mode="bilinear", align_corners=False
            )
            fused_list.append(f_resized)

        fused = torch.cat(fused_list, dim=1)  # (1, C, H_feat, W_feat)
        _, C, H_feat, W_feat = fused.shape

        # -----------------------------
        # RESIZE MASK TO FEATURE MAP SIZE
        # -----------------------------
        # Mask needs to match feature spatial dims to index correctly
        mask_resized = cv2.resize(
            mask, (W_feat, H_feat), interpolation=cv2.INTER_NEAREST
        )
        mask_bool = mask_resized > 127  # (H_feat, W_feat) boolean

        if mask_bool.sum() == 0:
            print(f"⚠️  Mask vanished after resize for {mask_name}, skipping.")
            continue

        # -----------------------------
        # EXTRACT ONLY MASKED PIXEL FEATURES
        # -----------------------------
        # fused: (1, C, H_feat, W_feat) → (C, H_feat, W_feat)
        feat_map = fused.squeeze(0).cpu().numpy()

        # Reshape to (H_feat*W_feat, C) then select rope pixels only
        feat_flat  = feat_map.reshape(C, -1).T  # (H_feat*W_feat, C)
        mask_flat  = mask_bool.reshape(-1)       # (H_feat*W_feat,)
        rope_feats = feat_flat[mask_flat]        # (N_rope_pixels, C)

        all_pixel_features.append(rope_feats)

        print(f"   {base_name}: {rope_feats.shape[0]} rope feature pixels extracted.")

# -----------------------------
# STACK ALL ROPE PIXEL FEATURES
# -----------------------------
if not all_pixel_features:
    print("❌ No features collected — check your paths and masks.")
    exit()

pixel_features = np.concatenate(all_pixel_features, axis=0)  # (Total_rope_pixels, C)
C = pixel_features.shape[1]
print(f"\nTotal rope pixels for PaDiM: {pixel_features.shape[0]:,} across all images.")

# -----------------------------
# PADIM STATISTICS
# (computed only over rope pixels, not background)
# -----------------------------
mean    = np.mean(pixel_features, axis=0)          # (C,)
cov     = np.cov(pixel_features, rowvar=False)      # (C, C)
cov    += np.eye(C) * 0.01                          # numerical stability
inv_cov = np.linalg.inv(cov)                        # (C, C)

# -----------------------------
# SAVE MODEL
# -----------------------------
padim_model = {
    "mean":    mean,
    "inv_cov": inv_cov,
    "C": C,
    "H": H_feat,
    "W": W_feat,
    "IMG_SIZE": IMG_SIZE,
}
torch.save(padim_model, "/opt/sam2_project/padim_multilayer.pth")
print("✅ PaDiM trained on masked ROI pixels — saved to padim_multilayer.pth")
