from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPoint, Qt, QRectF, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QFont, QPainter, QPen, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QColorDialog, QDockWidget, QMainWindow, QMessageBox, QWidget, QVBoxLayout
)

from modules.presentation.qt.status_footer import StatusFooter
from .utils import compute_bbox
from .workers import SegmentationWorker, BatchSegmentationWorker
from .image_view import ImageView
from .exporter import Exporter
from .state_manager import StateManager
from .widgets.control_panel import ControlPanelWidget
from .widgets.object_list import ObjectListWidget

logger = logging.getLogger(__name__)

class SegmentationViewer(QMainWindow):
    """Main window for the segmentation viewer, allowing interactive mask selection and saving."""
    
    closed = Signal()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def __init__(
        self,
        parent: Optional[QWidget],
        image_paths: List[Path],
        compute_masks_fn: Callable[[Path, int, float], Tuple[np.ndarray, List[np.ndarray], List[float]]],
        params_defaults: Optional[Dict[str, float]] = None,
        title: str = "分割檢視",
        path_manager: Optional["PathManager"] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowModality(Qt.NonModal)
        self.showMaximized()

        self.image_paths: List[Path] = list(image_paths)
        self.idx: int = 0
        self.compute_masks_fn = compute_masks_fn
        self.pm = path_manager
        self.params = {
            "points_per_side": int((params_defaults or {}).get("points_per_side", 16)),
            "pred_iou_thresh": float((params_defaults or {}).get("pred_iou_thresh", 0.88)),
        }
        self.cache: Dict[Path, Tuple[np.ndarray, List[np.ndarray], List[float]]] = {}
        
        # State Manager
        self.state_manager = StateManager()
        
        # UI Components
        self.view = ImageView(self)
        self.view.viewport().installEventFilter(self)
        self.setCentralWidget(self.view)
        
        self.control_panel = ControlPanelWidget()
        self.object_list = ObjectListWidget()
        self.object_list.set_color_getter(self._get_class_color)
        
        self._setup_docks()
        self._create_menu_bar()
        self._connect_signals()
        
        self.status = StatusFooter.install(self)
        self.status.message("準備就緒")
        
        self.mask_color = [0, 255, 0]
        self.bbox_color = [0, 255, 0]
        self.mask_alpha = 0.4
        self._hover_idx: Optional[int] = None
        self._list_hover_idx: Optional[int] = None
        
        self._setup_shortcuts()
        self._start_batch_processing()

    def _setup_docks(self):
        # Left Dock: Object List
        left_widget = QWidget()
        left_box = QVBoxLayout()
        left_box.addWidget(self.object_list)
        left_box.setContentsMargins(0, 0, 0, 0)
        left_widget.setLayout(left_box)
        
        self.dock_objects = QDockWidget("標註物件", self)
        self.dock_objects.setWidget(left_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_objects)
        
        # Right Dock: Controls
        self.dock_controls = QDockWidget("控制面板", self)
        self.dock_controls.setWidget(self.control_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_controls)

    def _connect_signals(self):
        # View
        self.view.drawing_started.connect(self._on_drawing_started)
        self.view.drawing_moved.connect(self._on_drawing_moved)
        self.view.drawing_finished.connect(self._on_drawing_finished)
        
        # Control Panel
        cp = self.control_panel
        cp.display_mode_changed.connect(self._update_canvas)
        cp.nav_prev_clicked.connect(self._prev_image)
        cp.nav_next_clicked.connect(self._next_image)
        cp.reset_view_clicked.connect(self._reset_view_and_selections)
        cp.show_candidates_toggled.connect(lambda _: self._update_canvas())
        cp.output_mode_changed.connect(self._on_mode_changed)
        cp.tool_changed.connect(self._on_tool_changed)
        cp.brush_size_changed.connect(lambda _: self._update_cursor_visual(self.view.mapFromGlobal(QCursor.pos())))
        cp.undo_clicked.connect(self._undo_annotation)
        cp.redo_clicked.connect(self._redo_annotation)
        cp.save_selected_clicked.connect(self._save_selected)
        cp.save_all_clicked.connect(self._save_all)
        
        # Object List
        ol = self.object_list
        ol.hover_changed.connect(self._on_list_hover)
        ol.class_changed.connect(self._on_class_changed)
        ol.delete_requested.connect(self._on_delete_object)

    def _on_list_hover(self, mask_idx: int):
        self._list_hover_idx = mask_idx if mask_idx >= 0 else None
        self._update_canvas()

    def _on_class_changed(self, mask_idx: int, new_class_id: int):
        if mask_idx in self.state_manager.selected_indices:
            self.state_manager.annotations[mask_idx] = new_class_id
            self._update_canvas()
            self._update_object_list() # To update color

    def _on_delete_object(self, mask_idx: int):
        if mask_idx in self.state_manager.selected_indices:
            self.state_manager.save_history()
            self.state_manager.selected_indices.remove(mask_idx)
            if mask_idx in self.state_manager.annotations:
                del self.state_manager.annotations[mask_idx]
            self._update_ui_state()

    def _update_ui_state(self):
        self.control_panel.update_selected_count(len(self.state_manager.selected_indices))
        self.control_panel.update_undo_redo(self.state_manager.can_undo(), self.state_manager.can_redo())
        self._update_object_list()
        self._update_canvas()

    def _update_object_list(self):
        is_union = self.control_panel.mode_group.checkedId() == 1
        self.object_list.update_list(
            self.state_manager.selected_indices, 
            self.state_manager.annotations, 
            is_union
        )

    def _save_selected(self):
        if not self.state_manager.selected_indices and self._hover_idx is not None:
            ret = QMessageBox.question(self, "未選擇目標", "是否儲存目前滑鼠指向的目標？")
            if ret == QMessageBox.StandardButton.Yes:
                self._save_indices([self._hover_idx])
            return
            
        if not self.state_manager.selected_indices:
            QMessageBox.information(self, "提示", "尚未選擇任何目標")
            return

        path = self.image_paths[self.idx]
        bgr, masks, _ = self.cache[path]
        
        cp = self.control_panel
        is_union = cp.mode_group.checkedId() == 1
        
        common_args = {
            "parent_widget": self,
            "image_path": path,
            "bgr": bgr,
            "masks": masks,
            "selected_indices": self.state_manager.selected_indices,
            "annotations": self.state_manager.annotations,
            "output_path_str": cp.output_path_edit.text(),
            "format_str": cp.format_combo.currentText(),
            "crop_bbox": cp.crop_group.checkedId() == 1,
            "chk_coco": cp.chk_coco.isChecked(),
            "chk_voc": cp.chk_voc.isChecked(),
            "chk_yolo_det": cp.chk_yolo_det.isChecked(),
            "chk_yolo_seg": cp.chk_yolo_seg.isChecked(),
        }

        if is_union:
            Exporter.save_union(**common_args)
        else:
            Exporter.save_indices(**common_args)

    def _save_all(self):
        if not self.image_paths: return
        path = self.image_paths[self.idx]
        if path not in self.cache: return
        _, masks, _ = self.cache[path]
        if not masks:
            QMessageBox.information(self, "提示", "目前影像沒有任何分割目標")
            return
            
        ret = QMessageBox.question(self, "確認儲存", f"確定要儲存全部 {len(masks)} 個目標嗎？", QMessageBox.Yes | QMessageBox.No)
        if ret != QMessageBox.Yes: return

        self._save_indices(list(range(len(masks))))

    def _save_indices(self, indices: List[int]):
        # Helper to call Exporter.save_indices with subset
        path = self.image_paths[self.idx]
        bgr, masks, _ = self.cache[path]
        cp = self.control_panel
        
        Exporter.save_indices(
            parent_widget=self,
            image_path=path,
            bgr=bgr,
            masks=masks,
            selected_indices=set(indices),
            annotations=self.state_manager.annotations, # Pass all annotations, Exporter filters by indices
            output_path_str=cp.output_path_edit.text(),
            format_str=cp.format_combo.currentText(),
            crop_bbox=cp.crop_group.checkedId() == 1,
            chk_coco=cp.chk_coco.isChecked(),
            chk_voc=cp.chk_voc.isChecked(),
            chk_yolo_det=cp.chk_yolo_det.isChecked(),
            chk_yolo_seg=cp.chk_yolo_seg.isChecked(),
        )

    def _undo_annotation(self):
        if self.state_manager.undo():
            self._update_ui_state()
            self.status.message_temp("已復原", 1000)

    def _redo_annotation(self):
        if self.state_manager.redo():
            self._update_ui_state()
            self.status.message_temp("已重做", 1000)

    def _on_mode_changed(self, mode_id: int):
        is_union = mode_id == 1
        if is_union:
            for mask_idx in self.state_manager.selected_indices:
                self.state_manager.annotations[mask_idx] = 0
        self._update_ui_state()

    def _generate_class_color(self, class_id: int) -> list:
        import colorsys
        golden_ratio = 0.618033988749895
        hue = (class_id * golden_ratio) % 1.0
        saturation = 0.9
        value = 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return [int(b * 255), int(g * 255), int(r * 255)]

    def _get_class_color(self, class_id: int) -> list:
        return self._generate_class_color(class_id)

    def _get_mask_color(self, mask_idx: int) -> list:
        class_id = self.state_manager.annotations.get(mask_idx, 0)
        return self._get_class_color(class_id)

    def _update_canvas(self):
        if not self.image_paths: return
        path = self.image_paths[self.idx]
        if path not in self.cache: return
        bgr, masks, _ = self.cache[path]
        base = bgr.copy()
        
        # Draw candidates
        if self.control_panel.chk_show_candidates.isChecked():
            self._draw_candidates(base, masks)

        use_bbox = self.control_panel.display_group.checkedId() == 1
        is_union = self.control_panel.mode_group.checkedId() == 1
        
        union_color = None
        if is_union and self.state_manager.selected_indices:
            first_idx = sorted(list(self.state_manager.selected_indices))[0]
            union_color = np.array(self._get_mask_color(first_idx), dtype=np.uint8)

        if not use_bbox:
            self._draw_masks(base, masks, is_union, union_color)
        else:
            self._draw_bboxes(base, masks, is_union, union_color)

        self.status.set_display_info("BBox" if use_bbox else "遮罩", is_union, len(self.state_manager.selected_indices))
        self.view.set_image_bgr(base)

    def _draw_candidates(self, base, masks):
        candidates_overlay = np.zeros_like(base)
        alpha_map = np.zeros(base.shape[:2], dtype=np.float32)
        for i, m in enumerate(masks):
            if i in self.state_manager.selected_indices: continue
            mask_bool = m > 0
            color = np.array(self._generate_class_color(i), dtype=np.uint8)
            candidates_overlay[mask_bool] = color
            alpha_map[mask_bool] = 0.3
        
        mask_indices = alpha_map > 0
        alpha_3d = alpha_map[mask_indices][:, None]
        base[mask_indices] = (base[mask_indices] * (1 - alpha_3d) + candidates_overlay[mask_indices] * alpha_3d).astype(np.uint8)

    def _draw_masks(self, base, masks, is_union, union_color):
        # Selected
        for i in self.state_manager.selected_indices:
            if 0 <= i < len(masks):
                m = masks[i] > 0
                color = union_color if (is_union and union_color is not None) else np.array(self._get_mask_color(i), dtype=np.uint8)
                alpha = self.mask_alpha
                base[m] = (base[m] * (1 - alpha) + color * alpha).astype(np.uint8)
        
        # Hover
        hover_idx = self._list_hover_idx if self._list_hover_idx is not None else self._hover_idx
        if hover_idx is not None and 0 <= hover_idx < len(masks):
            m = masks[hover_idx] > 0
            color = np.array(self._get_mask_color(hover_idx), dtype=np.uint8)
            alpha = min(1.0, self.mask_alpha + 0.2)
            base[m] = (base[m] * (1 - alpha) + color * alpha).astype(np.uint8)
            
            m_uint8 = m.astype(np.uint8)
            if not m_uint8.flags['C_CONTIGUOUS']:
                m_uint8 = np.ascontiguousarray(m_uint8)
            contours, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                if not base.flags['C_CONTIGUOUS']:
                    base = np.ascontiguousarray(base)
                cv2.polylines(base, contours, True, tuple(int(c) for c in color), 2)

    def _draw_bboxes(self, base, masks, is_union, union_color):
        H, W = base.shape[:2]
        
        # Ensure base is contiguous for cv2 operations
        if not base.flags['C_CONTIGUOUS']:
            base = np.ascontiguousarray(base)
            
        if is_union and self.state_manager.selected_indices:
            union_mask = np.zeros((H, W), dtype=np.uint8)
            for i in self.state_manager.selected_indices:
                if 0 <= i < len(masks):
                    union_mask = np.maximum(union_mask, masks[i])
            x, y, w, h = compute_bbox(union_mask > 0)
            color = tuple(int(c) for c in union_color) if union_color is not None else (0, 255, 0)
            cv2.rectangle(base, (x, y), (x + w, y + h), color, 3)
        else:
            for i in self.state_manager.selected_indices:
                if 0 <= i < len(masks):
                    x, y, w, h = compute_bbox(masks[i] > 0)
                    color = tuple(int(c) for c in self._get_mask_color(i))
                    cv2.rectangle(base, (x, y), (x + w, y + h), color, 2)
            
            if self._hover_idx is not None and 0 <= self._hover_idx < len(masks):
                x, y, w, h = compute_bbox(masks[self._hover_idx] > 0)
                color = tuple(int(c) for c in self._get_mask_color(self._hover_idx))
                cv2.rectangle(base, (x, y), (x + w, y + h), color, 3)

    def _load_current_image(self, recompute: bool = False):
        if not self.image_paths: return
        path = self.image_paths[self.idx]
        
        # Save previous state if switching images
        # (Actually we should save before switching idx, which is done in _prev/_next)
        
        cache_file = path.parent / f"{path.stem}.sam_cache.npz"
        
        if not recompute and cache_file.exists():
            try:
                self.status.message(f"載入快取: {path.name}")
                cached = np.load(cache_file, allow_pickle=True)
                bgr = cv2.imread(str(path))
                masks = [cached[f'mask_{i}'] for i in range(len([k for k in cached.keys() if k.startswith('mask_')]))]
                scores = cached['scores'].tolist()
                
                H, W = bgr.shape[:2]
                self.status.set_image_resolution(W, H)
                self.status.set_cursor_xy(None, None)
                masks = [(m > 0).astype(np.uint8) for m in masks]
                self.cache[path] = (bgr, masks, scores)
                self._update_ui_after_load(path)
                return
            except Exception as e:
                logger.warning(f"快取載入失敗: {e}")
        
        if recompute or path not in self.cache:
            from modules.presentation.qt.progress_dialog import ThemedProgressDialog
            self.progress = ThemedProgressDialog("處理中", f"正在分割影像 ({self.idx + 1}/{len(self.image_paths)}):\n{path.name}", self)
            self.progress.show()
            self.setEnabled(False)
            
            self.worker = SegmentationWorker(
                self.compute_masks_fn, path, 
                int(self.params["points_per_side"]), 
                float(self.params["pred_iou_thresh"])
            )
            self.worker.finished.connect(lambda b, m, s: self._on_worker_finished(path, b, m, s))
            self.worker.error.connect(self._on_worker_error)
            self.worker.start()
            return

        self._update_ui_after_load(path)

    def _on_worker_finished(self, path, bgr, masks, scores):
        if hasattr(self, 'progress'): self.progress.close()
        self.setEnabled(True)
        
        H, W = bgr.shape[:2]
        self.status.set_image_resolution(W, H)
        self.status.set_cursor_xy(None, None)
        
        cache_file = path.parent / f"{path.stem}.sam_cache.npz"
        try:
            cache_data = {'scores': np.array(scores)}
            for i, m in enumerate(masks): cache_data[f'mask_{i}'] = m
            np.savez_compressed(cache_file, **cache_data)
        except Exception as e: logger.warning(f"快取儲存失敗: {e}")

        masks = [(m > 0).astype(np.uint8) for m in masks]
        self.cache[path] = (bgr, masks, scores)
        self._update_ui_after_load(path)

    def _on_worker_error(self, err_msg):
        if hasattr(self, 'progress'): self.progress.close()
        self.setEnabled(True)
        QMessageBox.critical(self, "分割失敗", f"無法分割：{err_msg}")

    def _update_ui_after_load(self, path):
        # Load state
        if not self.state_manager.load_image_state(path):
            # Try load json
            json_path = path.parent / f"{path.stem}_annotations.json"
            if not self.state_manager.load_from_json(json_path):
                self.state_manager.clear_current_state()
        
        self._hover_idx = None
        self.control_panel.update_nav_buttons(self.idx > 0, self.idx < len(self.image_paths) - 1)
        self._update_ui_state()
        
        if path in self.cache:
            num_masks = len(self.cache[path][1])
            self.status.message(f"載入完成：{path.name}，共有 {num_masks} 個候選遮罩")

    def _prev_image(self):
        if self.idx > 0:
            self.state_manager.save_image_state(self.image_paths[self.idx])
            self.idx -= 1
            self._load_current_image()

    def _next_image(self):
        if self.idx < len(self.image_paths) - 1:
            self.state_manager.save_image_state(self.image_paths[self.idx])
            self.idx += 1
            self._load_current_image()

    def _reset_view_and_selections(self):
        self.view.reset_view()
        self.state_manager.clear_current_state()
        self._update_ui_state()
        self.status.message_temp("已重設視圖並清除所有選取", 1500)

    # ... (Keep _create_menu_bar, _setup_shortcuts, _start_batch_processing, _on_batch_*, _create_emoji_cursor, _on_tool_changed, _update_cursor_visual, _on_drawing_*, _apply_brush_stroke, _map_widget_to_image, _hit_test_xy, eventFilter)
    # Due to space, I'll implement the remaining essential methods below.

    def _create_menu_bar(self):
        menubar = self.menuBar()
        opts = menubar.addMenu("選項")
        opts.addAction("分割參數設定...", self._show_params_dialog)
        opts.addSeparator()
        opts.addAction("遮罩透明度...", self._change_mask_alpha)
        opts.addSeparator()
        opts.addAction("快捷鍵列表...", self._show_shortcuts_dialog)
        
        view = menubar.addMenu("檢視")
        view.addAction("淺色主題", lambda: self._apply_theme("light"))
        view.addAction("深色主題", lambda: self._apply_theme("dark"))
        
        help_m = menubar.addMenu("說明")
        help_m.addAction("使用說明", self._show_help)
        
        about = menubar.addMenu("關於")
        about.addAction("關於本專案...", self._show_about)

    def _show_params_dialog(self):
        from PySide6.QtWidgets import QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QDialogButtonBox
        d = QDialog(self)
        d.setWindowTitle("分割參數設定")
        layout = QFormLayout(d)
        
        spn_points = QSpinBox()
        spn_points.setRange(4, 128)
        spn_points.setValue(self.params["points_per_side"])
        layout.addRow("Points per side:", spn_points)
        
        spn_iou = QDoubleSpinBox()
        spn_iou.setRange(0.1, 0.99)
        spn_iou.setSingleStep(0.01)
        spn_iou.setValue(self.params["pred_iou_thresh"])
        layout.addRow("Pred IoU threshold:", spn_iou)
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(d.accept)
        btns.rejected.connect(d.reject)
        layout.addRow(btns)
        
        if d.exec() == QDialog.Accepted:
            self.params["points_per_side"] = spn_points.value()
            self.params["pred_iou_thresh"] = spn_iou.value()
            if QMessageBox.question(self, "套用參數", "是否使用新參數重新計算？", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                self._load_current_image(recompute=True)

    def _change_mask_alpha(self):
        from PySide6.QtWidgets import QInputDialog
        val, ok = QInputDialog.getInt(self, "遮罩透明度", "輸入透明度 (0-100):", int(self.mask_alpha * 100), 0, 100)
        if ok:
            self.mask_alpha = val / 100.0
            self._update_canvas()

    def _setup_shortcuts(self):
        from modules.presentation.qt.shortcut_manager import ShortcutManager
        try:
            sm = ShortcutManager()
            if k := sm.get_shortcut('nav.prev'): QShortcut(QKeySequence(k), self).activated.connect(self._prev_image)
            if k := sm.get_shortcut('nav.next'): QShortcut(QKeySequence(k), self).activated.connect(self._next_image)
            if k := sm.get_shortcut('save.selected'): QShortcut(QKeySequence(k), self).activated.connect(self._save_selected)
            if k := sm.get_shortcut('view.reset'): QShortcut(QKeySequence(k), self).activated.connect(self._reset_view_and_selections)
            if k := sm.get_shortcut('edit.undo'): QShortcut(QKeySequence(k), self).activated.connect(self._undo_annotation)
        except Exception: pass

    def _start_batch_processing(self):
        if not self.image_paths: return
        self._load_current_image(recompute=False)
        
        import time
        self._batch_start_time = time.time()
        
        from modules.presentation.qt.progress_dialog import ThemedProgressDialog
        self.batch_progress = ThemedProgressDialog("批次處理中", "準備開始...", self)
        self.batch_progress.set_range(0, len(self.image_paths))
        self.batch_progress.show()
        
        self.batch_worker = BatchSegmentationWorker(
            self.compute_masks_fn, self.image_paths,
            int(self.params["points_per_side"]), float(self.params["pred_iou_thresh"])
        )
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.finished.connect(self._on_batch_finished)
        self.batch_progress.rejected.connect(self.batch_worker.stop)
        self.batch_worker.start()

    def _on_batch_progress(self, current, total, msg):
        if hasattr(self, 'batch_progress'):
            self.batch_progress.set_value(current)
            self.batch_progress.set_message(f"({current}/{total}) {msg}")

    def _on_batch_finished(self):
        if hasattr(self, 'batch_progress'): self.batch_progress.close()
        self._load_current_image(recompute=False)

    def _on_tool_changed(self, tool_id):
        cursors = {
            0: Qt.CursorShape.ArrowCursor,
            1: self._create_emoji_cursor("🖌️"),
            2: self._create_emoji_cursor("🧽"),
            3: self._create_emoji_cursor("🧹")
        }
        self.view.viewport().setCursor(cursors.get(tool_id, Qt.CursorShape.ArrowCursor))
        self.status.set_tool_mode(["👆 選取", "🖌️ 畫筆", "🧽 橡皮擦", "🧹 魔法掃把"][tool_id])

    def _create_emoji_cursor(self, emoji: str, size: int = 32) -> QCursor:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setFont(QFont("Segoe UI Emoji", int(size * 0.8)))
        painter.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, emoji)
        painter.end()
        return QCursor(pixmap, size // 2, size // 2)

    def _update_cursor_visual(self, pos: QPoint):
        from PySide6.QtWidgets import QGraphicsEllipseItem
        if not hasattr(self, 'cursor_item'):
             self.cursor_item = QGraphicsEllipseItem()
             self.cursor_item.setZValue(100)
             self.view.scene().addItem(self.cursor_item)
             self.cursor_item.hide()
             
        tool_id = self.control_panel.tool_group.checkedId()
        if tool_id in [1, 2]:
            scene_pos = self.view.mapToScene(pos)
            size = self.control_panel.slider_brush.value()
            self.cursor_item.setRect(scene_pos.x() - size/2, scene_pos.y() - size/2, size, size)
            
            if tool_id == 1 and self.state_manager.selected_indices:
                last_idx = sorted(list(self.state_manager.selected_indices))[-1]
                color = self._get_mask_color(last_idx)
                self.cursor_item.setPen(QPen(QColor(color[2], color[1], color[0]), 1))
            else:
                self.cursor_item.setPen(QPen(QColor(255, 255, 255), 1))
            self.cursor_item.show()
        else:
            self.cursor_item.hide()

    def _on_drawing_started(self, x, y):
        tool_id = self.control_panel.tool_group.checkedId()
        if tool_id == 0: return
        if not self.state_manager.selected_indices and tool_id != 3:
            # Auto create mask if brush and no selection? Or warn?
            # Original logic: create new if brush
            if tool_id == 1:
                path = self.image_paths[self.idx]
                _, masks, scores = self.cache[path]
                H, W = masks[0].shape if masks else (100, 100) # Fallback
                new_mask = np.zeros((H, W), dtype=np.uint8)
                masks.append(new_mask)
                scores.append(1.0)
                new_idx = len(masks) - 1
                self.state_manager.selected_indices.add(new_idx)
                self.state_manager.annotations[new_idx] = 0
                self._update_ui_state()
            else:
                self.status.message_temp("請先選取一個物件", 2000)
                return
        
        self._is_drawing = True
        self._apply_brush_stroke(x, y, tool_id)

    def _on_drawing_moved(self, x, y):
        if getattr(self, "_is_drawing", False):
            self._apply_brush_stroke(x, y, self.control_panel.tool_group.checkedId())

    def _on_drawing_finished(self, x, y):
        self._is_drawing = False

    def _apply_brush_stroke(self, x, y, tool_id):
        path = self.image_paths[self.idx]
        if path not in self.cache: return
        _, masks, _ = self.cache[path]
        radius = self.control_panel.slider_brush.value() // 2
        
        for idx in self.state_manager.selected_indices:
            if 0 <= idx < len(masks):
                if not masks[idx].flags['C_CONTIGUOUS']:
                    masks[idx] = np.ascontiguousarray(masks[idx])
                cv2.circle(masks[idx], (x, y), radius, 1 if tool_id == 1 else 0, -1)
        self._update_canvas()

    def eventFilter(self, obj, event):
        if obj is self.view.viewport():
            if event.type() == QEvent.MouseMove:
                pos = event.position().toPoint()
                img_xy = self.view.map_widget_to_image(pos)
                if self.control_panel.tool_group.checkedId() == 0:
                    if img_xy:
                        x, y = img_xy
                        path = self.image_paths[self.idx]
                        _, masks, _ = self.cache[path]
                        self._hover_idx = self._hit_test_xy(masks, x, y)
                        self.status.set_cursor_xy(x, y)
                    else:
                        self._hover_idx = None
                        self.status.set_cursor_xy(None, None)
                    self._update_canvas()
                return False
            elif event.type() == QEvent.MouseButtonPress:
                if self.control_panel.tool_group.checkedId() == 0:
                    pos = event.position().toPoint()
                    img_xy = self.view.map_widget_to_image(pos)
                    if img_xy:
                        x, y = img_xy
                        path = self.image_paths[self.idx]
                        _, masks, _ = self.cache[path]
                        tgt = self._hit_test_xy(masks, x, y)
                        if tgt is not None:
                            self.state_manager.save_history()
                            if event.button() == Qt.MouseButton.LeftButton:
                                self.state_manager.selected_indices.add(tgt)
                                if tgt not in self.state_manager.annotations:
                                    self.state_manager.annotations[tgt] = 0
                            elif event.button() == Qt.MouseButton.RightButton:
                                if tgt in self.state_manager.selected_indices:
                                    self.state_manager.selected_indices.remove(tgt)
                                    if tgt in self.state_manager.annotations:
                                        del self.state_manager.annotations[tgt]
                            self._update_ui_state()
                return False
        return super().eventFilter(obj, event)

    def _hit_test_xy(self, masks, x, y):
        if not masks: return None
        if y < 0 or y >= masks[0].shape[0] or x < 0 or x >= masks[0].shape[1]: return None
        hits = [i for i, m in enumerate(masks) if m[y, x] > 0]
        if not hits: return None
        return sorted([(i, int(masks[i].sum())) for i in hits], key=lambda t: t[1])[0][0]

    def _show_shortcuts_dialog(self):
        from modules.presentation.qt.shortcut_manager import ShortcutManager
        try:
            sm = ShortcutManager()
            shortcuts = sm.get_all_shortcuts()
            msg = "快捷鍵列表：\n\n"
            for action, key in shortcuts.items():
                msg += f"{action}: {key}\n"
            QMessageBox.information(self, "快捷鍵", msg)
        except Exception:
            QMessageBox.information(self, "快捷鍵", "無法載入快捷鍵設定")

    def _show_help(self):
        QMessageBox.information(self, "使用說明", 
            "1. 左鍵點擊：選取物件\n"
            "2. 右鍵點擊：取消選取\n"
            "3. 滾輪：縮放影像\n"
            "4. 中鍵拖曳：移動視圖\n"
            "5. 左右鍵/PageUp/PageDown：切換影像"
        )

    def _show_about(self):
        QMessageBox.about(self, "關於", "Webcam Segmentation Tool\n\n基於 Segment Anything Model (SAM)")

    def _apply_theme(self, theme):
        # Theme logic would go here
        pass
