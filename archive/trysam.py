# from ultralytics import YOLO, SAM

# # Correct internal container path
# img_path = "/workspace/photo0002.png" 

# # 1. Load YOLO 
# # We don't call .half() here; we let the predictor handle it
# yolo_model = YOLO("/workspace/accordwireseg.pt") 

# # Run YOLO with half=True - this fixes the dtype mismatch
# yolo_results = yolo_model.predict(img_path, device='cuda', half=True)

# # 2. Get the boxes
# boxes = yolo_results[0].boxes.xyxy.tolist()

# if not boxes:
#     print("No objects detected by YOLO. Check your model/image!")
# else:
#     # 3. Load SAM 2.1 Tiny
#     sam_model = SAM("sam2.1_t.pt")
    
#     # Run SAM with half=True to save memory on the Orin Nano
#     final_results = sam_model.predict(
#         img_path, 
#         bboxes=boxes, 
#         save=True, 
#         device='cuda', 
#         half=True
#     )
#     print("Success! Results saved to /workspace/runs/")

import torch
import gc
import os
from ultralytics import YOLO, SAM

# 1. Setup Paths
img_path = "/workspace/photo0002.png"
yolo_weights = "/workspace/accordwireseg.pt"

# Ensure output directory exists
os.makedirs("/workspace/runs", exist_ok=True)

def run_pipeline():
    # --- STEP 1: YOLO DETECTION ---
    print("--- Starting YOLO Detection ---")
    yolo_model = YOLO(yolo_weights)
    
    # Tuning parameters to fix overlapping boxes
    yolo_results = yolo_model.predict(
        img_path, 
        device='cuda', 
        half=True, 
        conf=0.2, 
        iou=0.2
    )
    
    # Extract bounding boxes
    boxes = yolo_results[0].boxes.xyxy.tolist()
    print(f"YOLO found {len(boxes)} objects.")

    # --- CRITICAL: GPU MEMORY CLEANUP ---
    del yolo_model
    del yolo_results
    gc.collect()
    torch.cuda.empty_cache() 
    print("GPU Memory Cleared of YOLO.")

    if not boxes:
        print("No wire ropes detected.")
        return

    # --- STEP 2: SAM SEGMENTATION ---
    print(f"--- Starting SAM Segmentation on {len(boxes)} boxes ---")
    
    # Load SAM 2.1 Tiny
    # sam_model = SAM("sam2.1_t.pt")
    sam_model = SAM("sam3.pt")
    
    # Run prediction
    # Note: Removed 'multimask_output' to fix the SyntaxError
    sam_results = sam_model.predict(
        img_path, 
        bboxes=boxes, 
        save=True, 
        device='cuda', 
        half=True,
        retina_masks=False
    )

    print(f"Done! Results saved to: {sam_results[0].save_dir}")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"An error occurred: {e}")