# modules/burst.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer
from PySide6.QtMultimedia import QImageCapture

from modules.infrastructure.io.photo import PhotoCapture
from utils.utils import ensure_dir, ts

logger = logging.getLogger(__name__)


@dataclass
class BurstCallbacks:
    on_progress: Optional[Callable[[int], None]] = None  # remaining
    on_done: Optional[Callable[[], None]] = None


class BurstShooter(QObject):
    def __init__(self, image_capture: QImageCapture, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._photo = PhotoCapture(image_capture, parent=self)
        self._total = 0
        self._remaining = 0
        self._interval_ms = 500
        self._series_id = ""
        self._save_dir = Path(".")
        self._cbs = BurstCallbacks()

    def start(
        self,
        count: int,
        interval_ms: int,
        save_dir: Path,
        callbacks: Optional[BurstCallbacks] = None,
    ):
        """開始連拍序列。"""
        if self._timer.isActive():
            logger.warning("嘗試在連拍進行中再次 start，已忽略")
            return
        ensure_dir(save_dir)
        self._total = int(count)
        self._remaining = int(count)
        self._interval_ms = int(interval_ms)
        self._series_id = ts()
        self._save_dir = save_dir
        self._cbs = callbacks or BurstCallbacks()

        # 先拍第一張
        self._tick(initial=True)
        # 啟動之後的節奏
        self._timer.start(self._interval_ms)

    def stop(self):
        """停止連拍序列。"""
        if self._timer.isActive():
            self._timer.stop()
        self._remaining = 0
        self._series_id = ""

    def is_active(self) -> bool:
        """檢查連拍序列目前是否處於活動狀態。"""
        return self._timer.isActive()

    def _tick(self, initial: bool = False):
        if self._remaining <= 0:
            self.stop()
            logger.info(
                "連拍完成: series=%s, 最後張序號=%d, 共 %d 張",
                self._series_id,
                self._total,
                self._total,
            )

            if self._cbs.on_done:
                self._cbs.on_done()
            return

        shot_index = self._total - self._remaining + 1
        self._photo.capture_burst_one(self._save_dir, self._series_id, shot_index)
        self._remaining -= 1

        if self._remaining > 0 and self._cbs.on_progress:
            self._cbs.on_progress(self._remaining)
