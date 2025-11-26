from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QDir, QModelIndex, QPoint, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QFileSystemModel,
    QInputDialog,
    QMenu,
    QMessageBox,
    QToolBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class MediaExplorer(QDockWidget):
    """左側可收放的檔案導航欄
    - 顯示指定根目錄下的影像與影片檔
    - 支援刪除、重新命名
    - 可透過 QDockWidget 收放/關閉；可浮動為獨立視窗
    """

    file_deleted = Signal(str)  # 送出被刪除檔案的絕對路徑
    file_renamed = Signal(str, str)  # (old_abs_path, new_abs_path)
    files_segment_requested = Signal(list)  # 多選影像欲執行自動分割時發出

    def __init__(self, parent=None, name_filters: Optional[List[str]] = None):
        super().__init__("媒體檔案", parent)
        self.setObjectName("MediaExplorerDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )

        self._root_dir = Path.home()
        self._filters = name_filters or ["*.jpg", "*.jpeg", "*.png", "*.mp4", "*.mov", "*.avi"]

        # 內部主體
        body = QWidget(self)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(4, 4, 4, 4)

        # 工具列
        self.toolbar = QToolBar(body)
        self.action_segment = QAction("自動分割", self.toolbar)

        layout.addWidget(self.toolbar)

        # 檔案樹
        self.model = QFileSystemModel(self)
        self.model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)
        self.model.setNameFilters(self._filters)
        self.model.setNameFilterDisables(False)  # 僅顯示符合篩選
        self.model.setReadOnly(True)  # 重新命名我們自行處理

        self.tree = QTreeView(body)
        self.tree.setModel(self.model)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.AscendingOrder)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setColumnWidth(0, 260)  # 0-Name, 1-Size, 2-Type, 3-Date Modified

        layout.addWidget(self.tree)
        self.setWidget(body)

        # 綁定工具列動作
        self.action_segment.triggered.connect(self._emit_segment_selected)

    # ----------------------
    # 公開 API
    # ----------------------
    def set_root_dir(self, path: Path | str):
        """Set the root directory for the file explorer."""
        p = Path(path).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        self._root_dir = p
        root_index = self.model.setRootPath(str(p))
        self.tree.setRootIndex(root_index)

    def refresh(self):
        """Refresh the file system model."""
        # 某些 PySide6 版本未暴露 QFileSystemModel.refresh，改以 reset rootPath
        root = str(self._root_dir)
        root_index = self.model.setRootPath(root)
        self.tree.setRootIndex(root_index)

    # ----------------------
    # 內部輔助
    # ----------------------
    def _selected_indexes(self) -> List[QModelIndex]:
        sel = self.tree.selectionModel()
        return sel.selectedRows() if sel else []

    def _indexes_to_paths(self, indexes: List[QModelIndex]) -> List[Path]:
        return [Path(self.model.filePath(idx)) for idx in indexes]

    def _select_first_file_index(self) -> Optional[QModelIndex]:
        idxs = self._selected_indexes()
        return idxs[0] if idxs else None

    # ----------------------
    # 操作: 刪除 / 重新命名
    # ----------------------
    def delete_selected(self):
        """Delete the selected files."""
        indexes = self._selected_indexes()
        if not indexes:
            return
        paths = self._indexes_to_paths(indexes)

        # 僅刪除檔案，資料夾略過(避免誤刪)
        files = [p for p in paths if p.is_file()]
        if not files:
            return

        msg = "確定要刪除以下檔案？\n\n" + "\n".join([f"- {p.name}" for p in files])
        if (
            QMessageBox.question(self, "刪除檔案", msg, QMessageBox.Yes | QMessageBox.No)
            != QMessageBox.Yes
        ):
            return

        errors = []
        for p in files:
            try:
                p.unlink()
                self.file_deleted.emit(str(p))
            except Exception as e:
                logger.error("刪除檔案失敗: %s, error=%s", p, e)
                errors.append(f"{p.name}: {e}")

        if errors:
            QMessageBox.warning(self, "刪除部份失敗", "\n".join(errors))
        self.refresh()

    def rename_selected(self):
        """Rename the selected file."""
        idx = self._select_first_file_index()
        if idx is None:
            return
        p = Path(self.model.filePath(idx))
        if not p.is_file():
            return

        new_name, ok = QInputDialog.getText(self, "重新命名", f"新檔名（含副檔名）:\n{p.name}")
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()

        target = p.with_name(new_name)
        if target.exists():
            QMessageBox.warning(self, "命名衝突", f"檔案已存在：{target.name}")
            return

        try:
            p.rename(target)
            self.file_renamed.emit(str(p), str(target))
        except Exception as e:
            logger.exception("重新命名失敗: %s -> %s", p, target)
            QMessageBox.critical(self, "重新命名失敗", str(e))
            return
        self.refresh()

    # ----------------------
    # 右鍵選單
    # ----------------------
    # 修改右鍵選單：在選到影像時出現「自動分割」
    def _on_context_menu(self, pos: QPoint):
        """Show context menu for selected items."""
        menu = QMenu(self)
        img_files = self._selected_image_files()
        if img_files:
            menu.addAction(self.action_segment)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def last_image_path(self) -> Optional[str]:
        """Get the path of the most recently modified image file in the root directory."""
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        files = [
            p for p in Path(self._root_dir).glob("*") if p.is_file() and p.suffix.lower() in exts
        ]
        if not files:
            return None
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(files[0])

    def last_video_path(self) -> Optional[str]:
        """Get the path of the most recently modified video file in the root directory."""
        exts = {".mp4", ".mov", ".avi", ".mkv"}
        files = [
            p for p in Path(self._root_dir).glob("*") if p.is_file() and p.suffix.lower() in exts
        ]
        if not files:
            return None
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(files[0])

    def _selected_image_files(self) -> List[Path]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
        sel = self.tree.selectionModel().selectedIndexes()
        rows = [idx for idx in sel if idx.column() == 0]  # 只取檔名欄
        out = []
        for idx in rows:
            p = Path(self.model.filePath(idx))
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)
        return out

    # 新增：觸發 Signal
    def _emit_segment_selected(self):
        """Emit a signal to request segmentation for selected image files."""
        files = self._selected_image_files()
        if files:
            self.files_segment_requested.emit([str(p) for p in files])
