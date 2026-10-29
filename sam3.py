from ultralytics.models.sam import SAM3SemanticPredictor

# 1. Initialize with overrides (this is the correct interface for text prompts)
overrides = dict(
    conf=0.5,
    task="segment",
    mode="predict",
    model="sam3.pt",
    device="cuda",
    half=True,          # Essential for Orin Nano
    save=True,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# 2. Set image once (efficient if you want to run multiple queries)
predictor.set_image("/opt/sam2_project/input_images/photo0033.png")

# 3. Query with text prompt — pass as a LIST, not a bare string
results = predictor(text=["thin wire rope"])

# 4. Count results
if results and results[0].masks is not None:
    print(f"SAM 3 found {len(results[0].masks)} ropes using just a text prompt.")
else:
    print("No wire ropes detected — try lowering conf or rephrasing the prompt.")
