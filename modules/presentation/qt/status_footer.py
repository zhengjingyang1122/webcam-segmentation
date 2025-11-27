# modules/status_footer.py
from __future__ import annotations

from typing import Optional

# 既有 import 區塊若尚未匯入 Qt, 請補上
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


class StatusFooter(QStatusBar):
    """統一美化的底部狀態列：訊息 + 進度（支援忙碌/不定進度 與 定量進度）"""

    def __init__(self, parent: Optional[QMainWindow] = None):
        super().__init__(parent)
        self.setObjectName("UnifiedStatusFooter")

        # 文字 + 進度
        self._msg = QLabel("準備就緒", self)
        self._msg.setObjectName("StatusMessageLabel")
        self._msg.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self._progress = QProgressBar(self)
        self._progress.setObjectName("StatusProgressBar")
        self._progress.setFixedHeight(14)
        self._progress.setVisible(False)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)

        wrap = QWidget(self)
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(10)
        lay.addWidget(self._msg, 1)
        lay.addWidget(self._progress, 0)
        self.addPermanentWidget(wrap, 1)

        self._focus = QLabel("-", self)
        self._focus.setMinimumWidth(120)
        self.addPermanentWidget(self._focus)

        # style
        self.setStyleSheet(
            """
        #UnifiedStatusFooter {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #2b2f36, stop:1 #22252b);
            color: #e8eaed;
            border-top: 1px solid #3a3f47;
            font-size: 12px;
        }
        #StatusMessageLabel {
            color: #e8eaed;
        }
        #StatusProgressBar {
            background: #1b1e23;
            border: 1px solid #3a3f47;
            border-radius: 7px;
        }
        #StatusProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 #00b894, stop:1 #00cec9);
            border-radius: 6px;
        }
        """
        )

        # 暫時訊息
        self._last_persist_msg = "準備就緒"
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_temp_timeout)

        # 科幻彈窗指標
        self._scifi: Optional[SciFiProgressDialog] = None

        # 資訊顯示區 (Resolution, Cursor, Mode, Count)
        self._info_widget = QWidget(self)
        self._info_layout = QHBoxLayout(self._info_widget)
        self._info_layout.setContentsMargins(0, 0, 0, 0)
        self._info_layout.setSpacing(15)

        # 1. 解析度
        self._lbl_res = QLabel("", self)
        self._lbl_res.setMinimumWidth(80)
        self._lbl_res.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_res.setStyleSheet("color: #aaaaaa;")
        
        # 2. 游標座標
        self._lbl_cursor = QLabel("", self)
        self._lbl_cursor.setMinimumWidth(100)
        self._lbl_cursor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_cursor.setStyleSheet("color: #aaaaaa;")

        # 3. 模式與數量
        self._lbl_status = QLabel("", self)
        self._lbl_status.setMinimumWidth(120)
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_status.setStyleSheet("font-weight: bold; color: #e8eaed;")

        self._info_layout.addWidget(self._lbl_res)
        self._info_layout.addWidget(self._lbl_cursor)
        self._info_layout.addWidget(self._lbl_status)
        
        self.addPermanentWidget(self._info_widget, 0)

        self._img_wh = None
        self._cur_xy = None

        self._display_mode = None  # "遮罩" 或 "BBox"
        self._is_union = False  # True=聯集, False=個別
        self._selected_count = 0  # 已選目標數

        self._sim_timer = QTimer(self)
        self._sim_timer.setInterval(60)
        self._sim_timer.timeout.connect(self._on_sim_tick)
        self._sim_value = 0
        self._sim_target = 0
        self._sim_step = 1

    # ---------- 公開 API ----------
    def set_focus_quality(self, score: float, ok: bool) -> None:
        """更新對焦分數與狀態"""
        icon = "✔" if ok else "✖"
        self._focus.setText(f"清晰度 {icon} {score:.0f}")
        if ok:
            self._focus.setStyleSheet("color: #00dd00;")
        else:
            self._focus.setStyleSheet("color: #ff4444;")

    def message(self, text: str) -> None:
        """Display a persistent message in the status bar."""
        self._last_persist_msg = text
        self._msg.setText(text)

    def message_temp(self, text: str, ms: int = 2500) -> None:
        """Display a temporary message for a specified duration."""
        self._msg.setText(text)
        self._timer.start(max(1, int(ms)))

    def start_busy(self, text: Optional[str] = None) -> None:
        """Show an indeterminate progress bar (busy state)."""
        if text:
            self.message(text)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

    def stop_busy(self, text: Optional[str] = None) -> None:
        """Stop the busy state and hide the progress bar."""
        self._progress.setVisible(False)
        self._progress.setRange(0, 100)
        if text is not None:
            self.message(text)

    def set_progress(self, value: int, text: Optional[str] = None, maximum: int = 100) -> None:
        """Set the progress bar to a specific value."""
        self._progress.setVisible(True)
        self._progress.setRange(0, max(1, int(maximum)))
        v = max(0, min(int(value), self._progress.maximum()))
        self._progress.setValue(v)
        if text:
            self.message(text)

    # [新增] 放在「公開 API」區塊其他方法旁
    def set_display_info(self, mode: str, is_union: bool, selected_count: int) -> None:
        """Update the display information in the status bar (mode, union state, selection count)."""
        self._display_mode = mode
        self._is_union = bool(is_union)
        try:
            self._selected_count = max(0, int(selected_count))
        except Exception:
            self._selected_count = 0
        self._update_meta()

    # ---------- 靜態安裝器 ----------
    @staticmethod
    def install(win: QMainWindow) -> "StatusFooter":
        """Install the StatusFooter onto the given QMainWindow."""
        bar = StatusFooter(win)
        win.setStatusBar(bar)
        return bar

    # ---------- 內部 ----------
    def _on_temp_timeout(self):
        self._msg.setText(self._last_persist_msg)

    # ---------- 科幻進度條 API ----------
    def start_scifi(self, text: str = "處理中...") -> None:
        """Start a sci-fi style progress dialog."""
        try:
            if self._scifi is None:
                self._scifi = SciFiProgressDialog(parent=self.parent(), title=text)
            else:
                self._scifi.set_title(text)
            self._scifi.center_to_parent()
            self._scifi.show()
            self._scifi.raise_()

            # ★ 新增：顯示後立刻處理一次事件，確保視窗真的畫出來
            from PySide6.QtGui import QGuiApplication

            QGuiApplication.processEvents()
        except Exception:
            self.start_busy(text)

    def set_scifi_progress(self, value: int, text: Optional[str] = None) -> None:
        """Update the sci-fi progress dialog value."""
        if self._scifi is None:
            self.start_scifi(text or "處理中...")
        if text:
            self._scifi.set_title(text)
        self._scifi.set_determinate(value)

        # ★ 新增：更新進度後也刷新一次，避免卡在重計算時看不到變化
        from PySide6.QtGui import QGuiApplication

        QGuiApplication.processEvents()

    def stop_scifi(self, text: Optional[str] = None) -> None:
        """Stop and close the sci-fi progress dialog."""
        try:
            if self._sim_timer.isActive():
                self._sim_timer.stop()
            if self._scifi:
                try:
                    self._scifi.set_determinate(100)
                except Exception:
                    pass
                from PySide6.QtGui import QGuiApplication

                QGuiApplication.processEvents()
                self._scifi.close()
        finally:
            self._scifi = None
            if text is not None:
                self.message(text)

    # 顯示影像寬高與游標座標（供 SegmentationViewer 呼叫）
    def set_image_resolution(self, w: int, h: int) -> None:
        """Set the image resolution to be displayed in the status bar."""
        self._img_wh = (w, h)
        self._update_meta()

    def set_cursor_xy(self, x: int | None, y: int | None) -> None:
        """Set the cursor coordinates to be displayed in the status bar."""
        self._cur_xy = None if x is None or y is None else (x, y)
        self._update_meta()

    # [修改] 以下方法以新版本覆蓋
    def _update_meta(self) -> None:
        # 1. 解析度
        if self._img_wh:
            self._lbl_res.setText(f"{self._img_wh[0]} x {self._img_wh[1]}")
        else:
            self._lbl_res.setText("")
            
        # 2. 游標
        if self._cur_xy:
            self._lbl_cursor.setText(f"XY: {self._cur_xy[0]}, {self._cur_xy[1]}")
        else:
            self._lbl_cursor.setText("")
            
        # 3. 狀態 (模式 + 數量)
        if getattr(self, "_display_mode", None):
            mode_str = "聯集" if self._is_union else "個別"
            self._lbl_status.setText(f"[{mode_str}] 已選: {self._selected_count}")
        else:
            self._lbl_status.setText("")

    # 模擬載入進度：從 25% 慢慢跑到 99%，等待實際完成後才補 100%
    def start_scifi_simulated(
        self,
        text: str = "載入中...",
        start: int = 25,
        stop_at: int = 99,
        interval_ms: int = 60,
        step: int = 1,
    ) -> None:
        """Start a simulated progress animation."""
        self.start_scifi(text)
        self._sim_value = max(0, min(100, int(start)))
        self._sim_target = max(0, min(100, int(stop_at)))
        self._sim_step = max(1, int(step))
        self._sim_timer.setInterval(max(15, int(interval_ms)))
        if self._scifi:
            self._scifi.set_determinate(self._sim_value)
        QGuiApplication.processEvents()
        self._sim_timer.start()

    def _on_sim_tick(self) -> None:
        if self._scifi is None:
            self._sim_timer.stop()
            return
        if self._sim_value >= self._sim_target:
            self._sim_timer.stop()
            return
        self._sim_value = min(self._sim_target, self._sim_value + self._sim_step)
        self._scifi.set_determinate(self._sim_value)
        QGuiApplication.processEvents()


# ========== 科幻進度條對話框 ==========
class SciFiProgressDialog(QDialog):
    """半透明霓虹風掃描條 + 光暈"""

    def __init__(self, parent=None, title: str = "處理中..."):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setModal(True)

        self._title_label = QLabel(title, self)
        self._bar = QProgressBar(self)
        self._bar.setRange(0, 0)  # 預設不定進度
        self._bar.setTextVisible(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.addWidget(self._title_label)
        lay.addWidget(self._bar)

        # 霓虹樣式
        self.setStyleSheet(
            """
QDialog {
    background: rgba(7, 12, 20, 180);
    border: 1px solid rgba(0, 200, 255, 160);
    border-radius: 12px;
}
QLabel {
    color: #bffaff;
    font-size: 14px;
    letter-spacing: 1px;
}
QProgressBar {
    border: 1px solid rgba(0, 200, 255, 140);
    border-radius: 8px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 rgba(0,40,60,220), stop:1 rgba(0,25,45,220));
    height: 16px;
}
QProgressBar::chunk {
    border-radius: 7px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #00e5ff, stop:0.5 #00ffd5, stop:1 #00e5ff);
}
"""
        )

        glow = QGraphicsDropShadowEffect(self)
        glow.setBlurRadius(40)
        glow.setOffset(0, 0)
        glow.setColor(Qt.cyan)
        self._bar.setGraphicsEffect(glow)

        # 不定進度掃描動畫（來回 0↔100）
        self._indeterminate = True
        self._t = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)  # 約 60 fps

    def _tick(self):
        if self._indeterminate:
            self._t = (self._t + 2) % 200
            v = self._t if self._t <= 100 else 200 - self._t
            self._bar.setValue(v)

    def center_to_parent(self):
        if self.parent() and self.parent().isVisible():
            pw, ph = self.parent().width(), self.parent().height()
            px, py = self.parent().x(), self.parent().y()
            self.resize(max(360, int(pw * 0.38)), 110)
            self.move(px + (pw - self.width()) // 2, py + (ph - self.height()) // 2)
        else:
            screen = QGuiApplication.primaryScreen().geometry()
            self.resize(420, 110)
            self.move(screen.center() - self.rect().center())

    def set_title(self, text: str):
        self._title_label.setText(text)

    def set_determinate(self, value: int):
        self._indeterminate = False
        self._bar.setRange(0, 100)
        self._bar.setValue(max(0, min(100, int(value))))
