# Automated Saffron Stigma Detection

**Problem:** Harvesting saffron by hand is exhausting, costly, and easily ruins delicate flowers.  
**Solution:** We used YOLOv8-segmentation to automate this, letting AI instantly pinpoint the exact plucking spots for robotic tools.  
**How it works:** Saffron photos are processed on a GPU where the AI maps out thin stigmas to find precise cutting coordinates.

---

## Project Architecture & Workflow

The pipeline takes raw field imagery of saffron flowers, processes the frames using a computer vision backbone, and outputs pixel-accurate instance segmentations to isolate the target stigmas.

1. **Data Preprocessing:** Input images are resized to 640x640 pixels to prepare for model tensor dimensions.
2. **Feature Extraction:** A deep convolutional neural network backbone extracts spatial features from the floral structures.
3. **Segmentation & Localization:** The YOLOv8-seg head predicts boundary coordinates along with binary mask layers for individual threads.
4. **Coordinate Mapping:** The central centroids of the masks are computed to dictate precise robotic picking vectors.

---

## Tech Stack & Environment

* **Framework:** Ultralytics YOLOv8, PyTorch
* **Languages:** Python 3.11
* **Hardware Acceleration:** NVIDIA CUDA 
* **Tested Baselines:** * High-End Workstation (NVIDIA RTX 2000 Ada) -> Fast inference (~14.1ms)
  * Edge/Mobile Laptop (NVIDIA GeForce MX150) -> Memory-optimized execution

---

## Quick Start (Inference & Validation)

To validate the trained model locally on your setup, ensure your virtual environment is active and execute:

```bash
python validate_model.py
