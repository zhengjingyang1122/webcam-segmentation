from __future__ import annotations
import logging
from PySide6.QtCore import QThread, Signal
import numpy as np

logger = logging.getLogger(__name__)

class SegmentationWorker(QThread):
    finished = Signal(object, object, object)  # bgr, masks, scores
    error = Signal(str)

    def __init__(self, compute_fn, path, pps, iou):
        super().__init__()
        self.compute_fn = compute_fn
        self.path = path
        self.pps = pps
        self.iou = iou

    def run(self):
        try:
            bgr, masks, scores = self.compute_fn(self.path, self.pps, self.iou)
            self.finished.emit(bgr, masks, scores)
        except Exception as e:
            logger.error(f"SegmentationWorker error processing {self.path}: {e}", exc_info=True)
            self.error.emit(str(e))


class BatchSegmentationWorker(QThread):
    progress = Signal(int, int, str)
    finished = Signal()
    
    def __init__(self, compute_fn, paths, pps, iou):
        super().__init__()
        self.compute_fn = compute_fn
        self.paths = paths
        self.pps = pps
        self.iou = iou
        self._is_running = True

    def run(self):
        total = len(self.paths)
        for i, path in enumerate(self.paths):
            if not self._is_running:
                break
            
            cache_file = path.parent / f"{path.stem}.sam_cache.npz"
            if cache_file.exists():
                self.progress.emit(i + 1, total, f"已快取: {path.name}")
                continue

            self.progress.emit(i + 1, total, f"處理中: {path.name}")
            try:
                bgr, masks, scores = self.compute_fn(path, self.pps, self.iou)
                
                # Save to cache
                cache_data = {'scores': np.array(scores)}
                for k, m in enumerate(masks):
                    cache_data[f'mask_{k}'] = m
                np.savez_compressed(cache_file, **cache_data)
                
            except Exception as e:
                logger.error(f"BatchSegmentationWorker error processing {path}: {e}", exc_info=True)
        
        self.finished.emit()

    def stop(self):
        self._is_running = False
