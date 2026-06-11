from ultralytics import YOLO
import cv2
import os

# 1. Load the new 52MB Medium "Brain"
# Make sure 'best.pt' is in your saffron_project folder
model = YOLO('best.pt') 

# 2. Point it at the validation image
image_path = r'dataset/images/val/TdC1_118.jpg'

if not os.path.exists(image_path):
    print(f"❌ Error: File not found at {image_path}")
else:
    print(f"🔍 Scanning with Medium Model...")

    # 3. Run the detection
    # Change conf to 0.10 to make it find more stigmas
    results = model(image_path, device='cpu', conf=0.10)
    
    # 4. Load image for custom drawing
    img = cv2.imread(image_path)
    boxes = results[0].boxes

    print(f"🎯 Found {len(boxes)} stigmas.")

    # 5. Draw the neon dots and red numbers
    for i, box in enumerate(boxes):
        # Get center coordinates
        x_center, y_center, _, _ = box.xywh[0]
        x, y = int(x_center), int(y_center)
        
        # Draw neon green center dot
        cv2.circle(img, (x, y), radius=15, color=(0, 255, 0), thickness=-1)
        
        # Draw red number (1, 2, 3...)
        cv2.putText(img, str(i+1), (x + 25, y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 6. Setup Display Window
    cv2.namedWindow("Saffron Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Saffron Detection", 1000, 700)
    cv2.imshow("Saffron Detection", img)

    print("Press ANY KEY to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()