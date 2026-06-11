from ultralytics import YOLO
import cv2
import os

# --- Configuration ---
# This path points to where your model saves its best weights after training
MODEL_PATH = "runs/segment/saffron_stigma_model/weights/best.pt"

# Pick a random image from your validation set to test
# (Change this filename to an actual image in your val folder later)
TEST_IMAGE_PATH = "data/dataset/images/val/your_test_image.jpg" 

def test_model():
    print(f"1. Loading trained model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model weights not found. Did you finish training?")
        return
        
    model = YOLO(MODEL_PATH)

    print(f"2. Running inference on {TEST_IMAGE_PATH}...")
    # Run the model. 'save=True' tells YOLO to automatically draw the masks and save a copy
    results = model(TEST_IMAGE_PATH, save=True, conf=0.5)

    print("\n✅ Inference Complete!")
    print("Check the 'runs/segment/predict' folder to see the image with the red stigmas highlighted!")

if __name__ == "__main__":
    test_model()