import sys
import os
from pathlib import Path
import cv2
import numpy as np
import torch

# Add module path
sys.path.append(os.getcwd())

from modules.infrastructure.vision.sam_engine import SamEngine

def test_segmentation():
    print("Testing segmentation logic...")
    
    # Setup paths
    base_path = Path(os.getcwd())
    # Try vit_b first as it is smaller
    model_name = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    model_path = base_path / "models" / model_name
    img_path = base_path / "assets" / "Coffee.png"
    
    if not model_path.exists():
        print(f"Model {model_name} not found. Trying vit_h...")
        model_name = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        model_path = base_path / "models" / model_name
    
    if not model_path.exists():
        print("No models found. Cannot test.")
        return

    if not img_path.exists():
        print(f"Image not found at {img_path}")
        return

    print(f"Loading model {model_type} from {model_path}...")
    try:
        sam = SamEngine(model_path, model_type=model_type, device="cpu") # Use CPU to be safe
        sam.load()
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    except SystemExit:
        print("SystemExit caught during model loading")
        return

    print(f"Processing image {img_path}...")
    try:
        # Check image size
        img = cv2.imread(str(img_path))
        if img is not None:
            print(f"Image shape: {img.shape}")
        
        # Use fewer points to test stability
        bgr, masks, scores = sam.auto_masks_from_image(img_path, points_per_side=32)
        print(f"Success! Generated {len(masks)} masks.")
    except Exception as e:
        print(f"Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
    except SystemExit:
        print("SystemExit caught during segmentation")

if __name__ == "__main__":
    test_segmentation()
