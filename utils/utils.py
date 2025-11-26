# utils/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def ensure_dir(p: PathLike) -> Path:
    """確保目錄存在，如果不存在則建立。"""
    pth = Path(p).expanduser()
    pth.mkdir(parents=True, exist_ok=True)
    return pth