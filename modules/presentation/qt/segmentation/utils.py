from __future__ import annotations
from typing import Tuple, Optional
import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

def np_bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Convert a BGR numpy array to a QPixmap."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute the bounding box (x, y, w, h) for a binary mask."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)

def compute_polygon(mask: np.ndarray) -> Optional[np.ndarray]:
    """回傳最大連通域的外輪廓座標，形狀為 (N,2)，整數像素座標。"""
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return c.reshape(-1, 2)  # (N,2)
