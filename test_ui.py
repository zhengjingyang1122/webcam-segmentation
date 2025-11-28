import sys
import os
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication

# Add module path
sys.path.append(os.getcwd())

from modules.presentation.qt.segmentation.segmentation_viewer import SegmentationViewer

def mock_compute_masks(path, pps, iou):
    # Return dummy data
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    masks = [np.zeros((512, 512), dtype=np.uint8)]
    scores = [0.99]
    return img, masks, scores

def test_ui():
    app = QApplication(sys.argv)
    
    print("Creating SegmentationViewer...")
    try:
        viewer = SegmentationViewer(
            None,
            [Path("dummy.jpg")],
            mock_compute_masks,
            title="Test Viewer"
        )
        viewer.show()
        print("Viewer shown.")
        
        # Check if central widget is set
        if viewer.centralWidget():
            print(f"Central widget: {viewer.centralWidget()}")
        else:
            print("WARNING: No central widget set!")
            
        # Check docks
        docks = viewer.findChildren(type(viewer.dock_objects)) # QDockWidget
        print(f"Found {len(docks)} docks.")
        for dock in docks:
            print(f"Dock: {dock.windowTitle()}, Visible: {dock.isVisible()}")
            
    except Exception as e:
        print(f"Failed to create viewer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ui()
