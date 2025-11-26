# utils/utils.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Union, Optional

from modules.infrastructure.io.path_manager import PathManager

try:
    from PySide6.QtCore import QUrl
except Exception:
    QUrl = None

PathLike = Union[str, Path]

# 全域變數，用於在同一個 capture session 中共享 PathManager 實例
_current_path_manager: Optional[PathManager] = None

def get_path_manager(base_dir: PathLike, timestamp: Optional[str] = None) -> PathManager:
    """取得或建立一個共享的 PathManager 實例"""
    global _current_path_manager
    if _current_path_manager is None or (_current_path_manager.timestamp != timestamp and timestamp is not None):
        _current_path_manager = PathManager(base_dir, timestamp)
    return _current_path_manager

def clear_current_path_manager():
    """清除當前的共享 PathManager 實例"""
    global _current_path_manager
    _current_path_manager = None

def ensure_dir(p: PathLike) -> Path:
    pth = Path(p).expanduser()
    pth.mkdir(parents=True, exist_ok=True)
    return pth

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Removed unused ts_ms() function to simplify the utilities module.

def build_snapshot_path(save_dir: PathLike) -> Path:
    """使用 PathManager 建立單張照片的路徑"""
    pm = get_path_manager(save_dir, timestamp=ts())
    return pm.get_photo_path()

def build_burst_path(save_dir: PathLike, series_id: str, index: int) -> Path:
    """使用 PathManager 建立連拍照片的路徑"""
    # series_id 就是 timestamp
    pm = get_path_manager(save_dir, timestamp=series_id)
    return pm.get_burst_path(index)

def build_record_path(save_dir: PathLike) -> Path:
    """使用 PathManager 建立影片的路徑"""
    pm = get_path_manager(save_dir, timestamp=ts())
    return pm.get_video_path()

def to_qurl_or_str(path: Path) -> object:
    if QUrl is not None:
        try:
            return QUrl.fromLocalFile(str(path))
        except Exception:
            pass
    return str(path)