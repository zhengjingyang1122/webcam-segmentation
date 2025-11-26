# modules/explorer_controller.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit, QMainWindow, QPushButton

from modules.presentation.qt.explorer.explorer import MediaExplorer

logger = logging.getLogger(__name__)


class ExplorerController:
    """封裝 Dock 建立、按鈕切換、可視狀態同步與重新指定根目錄"""

    def __init__(self, main_window: QMainWindow, toggle_btn: QPushButton, dir_edit: QLineEdit):
        self._win = main_window
        self._btn = toggle_btn
        self._dir_edit = dir_edit

        self.explorer = MediaExplorer(self._win)
        self._win.addDockWidget(Qt.LeftDockWidgetArea, self.explorer)
        self.set_root_dir_from_edit()

        self._btn.setCheckable(True)
        self._btn.setChecked(True)
        self._btn.toggled.connect(self._on_toggle)

        self.explorer.visibilityChanged.connect(self._on_visibility_changed)

    def _on_toggle(self, checked: bool):
        """Handle toggle button state change to show/hide the explorer dock."""
        try:
            self.explorer.setVisible(checked)
            if checked:
                try:
                    self.explorer.setFloating(False)
                except Exception:
                    logger.warning("將 Dock 取消浮動時發生例外", exc_info=True)
        except Exception:
            logger.warning("顯示/隱藏檔案瀏覽 Dock 例外", exc_info=True)

    def _on_visibility_changed(self, visible: bool):
        """Sync the toggle button state when the dock visibility changes."""
        self._btn.blockSignals(True)
        self._btn.setChecked(visible)
        self._btn.blockSignals(False)

    def set_root_dir_from_edit(self):
        """Update the explorer's root directory from the path in the line edit."""
        path = Path(self._dir_edit.text()).expanduser()
        try:
            self.explorer.set_root_dir(path)
        except Exception:
            logger.error("設定檔案瀏覽根目錄失敗: %s", path, exc_info=True)

    def refresh(self):
        """Refresh the explorer view."""
        self.explorer.refresh()
