# modules/photo.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer
from PySide6.QtMultimedia import QImageCapture
import cv2
import numpy as np

from utils.utils import build_burst_path, build_snapshot_path, ensure_dir

logger = logging.getLogger(__name__)


class PhotoCapture(QObject):
    def __init__(self, image_capture: QImageCapture, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._cap = image_capture

    def _ready(self) -> bool:
        """檢查相機是否準備好進行擷取。"""
        try:
            return self._cap.isReadyForCapture()
        except Exception:
            return True

    def capture_single(self, save_dir: Path, on_saved: Optional[Callable[[Path], None]] = None):
        """擷取單張照片。"""
        path = build_snapshot_path(save_dir)
        self._capture_with_retry(path, on_saved=on_saved)

    def capture_burst_one(
        self,
        save_dir: Path,
        series_id: str,
        index: int,
        on_saved: Optional[Callable[[Path], None]] = None,
    ):
        """作為連拍序列的一部分擷取單張照片。"""
        path = build_burst_path(save_dir, series_id, index)
        self._capture_with_retry(path, on_saved=on_saved)

    def _capture_with_retry(
        self, path: Path, on_saved: Optional[Callable[[Path], None]] = None, retry_ms: int = 50
    ):
        """嘗試擷取到檔案，如果相機未就緒則重試。"""
        if self._ready():
            try:
                self._cap.captureToFile(str(path))
                if on_saved:
                    on_saved(path)
            except Exception:
                logger.exception("captureToFile 失敗: %s", path)
        else:
            logger.warning("相機未就緒，%d ms 後重試：%s", retry_ms, path)
            QTimer.singleShot(retry_ms, lambda: self._capture_with_retry(path, on_saved, retry_ms))
