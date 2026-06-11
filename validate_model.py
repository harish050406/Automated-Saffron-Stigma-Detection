from ultralytics import YOLO

def validate():
    # Load the Large model from the lab
    model = YOLO('best.pt')

    # Run validation with memory-saving settings
    metrics = model.val(
        data='data.yaml', 
        batch=1,         # Load only 1 image at a time to prevent OOM
        imgsz=640,       # Ensure we aren't using a higher resolution
        workers=1,       # Reduce CPU overhead
        device=0         # Use the MX150
    )

    # Print results
    results = metrics.mean_results()
    print(f"Precision: {results[0]:.4f}")
    print(f"Recall: {results[1]:.4f}")
    print(f"mAP@50: {results[2]:.4f}")

if __name__ == '__main__':
    validate()