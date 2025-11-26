# modules/infrastructure/io/path_manager.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


class PathManager:
    """
    管理所有與新輸出目錄結構相關的路徑生成。

    結構範例:
    <base_dir>/
    └── <timestamp>/
        ├── source/
        │   ├── burst_000.jpg
        │   └── burst_001.jpg
        │
        ├── embedding_burst_000.npz
        ├── masks_burst_000.npz
        ├── objects_burst_000/
        │   ├── 0.png
        │   └── 0.txt
        │
        └── embedding_burst_001.npz
            ...
    """

    def __init__(self, base_dir: str | Path, timestamp: Optional[str] = None):
        if timestamp:
            ts_str = timestamp
        else:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.base_dir = Path(base_dir)
        self.timestamp = ts_str
        self._capture_dir = self.base_dir / self.timestamp
        self._source_dir = self._capture_dir / "source"

        # 確保基礎目錄存在
        self._source_dir.mkdir(parents=True, exist_ok=True)

    def get_capture_dir(self) -> Path:
        """取得此擷取 Session 的根目錄。"""
        return self._capture_dir

    def get_source_dir(self) -> Path:
        """取得儲存來源影像/影片的目錄。"""
        return self._source_dir

    @staticmethod
    def get_source_name(path: Path) -> str:
        """從 source 檔案路徑中提取不含副檔名的基本名稱 (例如 'burst_001')"""
        return path.stem

    def get_photo_path(self) -> Path:
        """用於單張拍照模式"""
        return self._source_dir / "photo.jpg"

    def get_burst_path(self, index: int) -> Path:
        """用於連拍模式"""
        return self._source_dir / f"burst_{index:03d}.jpg"

    def get_video_path(self) -> Path:
        """用於錄影模式"""
        return self._source_dir / "video.mp4"

    def get_embedding_path(self, source_name: str) -> Path:
        """取得指定 source 的 embedding 檔案路徑"""
        return self._capture_dir / f"embedding_{source_name}.npz"

    def get_masks_path(self, source_name: str) -> Path:
        """取得指定 source 的 masks NPZ 檔案路徑"""
        return self._capture_dir / f"masks_{source_name}.npz"

    def get_objects_dir(self, source_name: str) -> Path:
        """取得指定 source 的物件儲存資料夾路徑"""
        obj_dir = self._capture_dir / f"objects_{source_name}"
        obj_dir.mkdir(exist_ok=True)
        return obj_dir

    def get_object_path(
        self, source_name: str, object_index: int, label: bool = False, label_ext: str = "txt"
    ) -> Path:
        """取得單一物件的影像或標註檔案路徑"""
        obj_dir = self.get_objects_dir(source_name)
        ext = label_ext if label else "png"
        return obj_dir / f"{object_index}.{ext}"
