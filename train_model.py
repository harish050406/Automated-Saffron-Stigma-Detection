from ultralytics import YOLO

# 1. Start with a fresh Nano segmentation brain
model = YOLO('yolov8n-seg.pt')

print("Starting High-Accuracy Training Run...")

# 2. Run the upgraded training command
results = model.train(
    data='data.yaml',
    
    # --- TIME & SAFETY ---
    epochs=200,          # Give the AI 200 chances to learn instead of 50
    patience=30,         # If accuracy doesn't improve for 30 rounds, stop early!
    
    # --- HARDWARE LIMITS ---
    batch=4,             # Keep this low so your GPU doesn't run out of memory
    workers=0,           # Prevents Windows pagefile/thread crashing
    
    # --- AGGRESSIVE DATA AUGMENTATION ---
    degrees=90.0,        # Rotates images up to 90 degrees to simulate new angles
    fliplr=0.5,          # Flips images left-to-right 50% of the time
    hsv_s=0.5,           # Randomly changes color saturation by 50%
    hsv_v=0.4,           # Randomly changes lighting/brightness by 40%
    
    # --- PROJECT FOLDER ---
    project='runs/segment',
    name='saffron_stigma_upgraded' # Saves this new brain in a separate folder!
)

print("Training Complete! Check the 'saffron_stigma_upgraded' folder for your new best.pt")