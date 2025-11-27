# modules/segmentation_viewer.py
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QDir, QEvent, QPoint, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPixmap, QTransform, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFileSystemModel,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


from modules.presentation.qt.status_footer import StatusFooter

logger = logging.getLogger(__name__)


# ---------- helpers ----------
def np_bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Convert a BGR numpy array to a QPixmap."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute the bounding box (x, y, w, h) for a binary mask."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)


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
                print(f"Error processing {path}: {e}")
        
        self.finished.emit()

    def stop(self):
        self._is_running = False


# ---------- QGraphicsView-based image view ----------


class ImageView(QGraphicsView):
    """A custom QGraphicsView for displaying images with zoom and pan support."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self.setRenderHints(
            self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)

    def set_image_bgr(self, bgr: np.ndarray) -> None:
        """Set the image to display from a BGR numpy array."""
        pix = np_bgr_to_qpixmap(bgr)
        if self._pix_item is None:
            self._pix_item = self._scene.addPixmap(pix)
            self._pix_item.setZValue(0)
            self._scene.setSceneRect(QRectF(pix.rect()))
            self.reset_view()
        else:
            self._pix_item.setPixmap(pix)
            self._scene.setSceneRect(QRectF(pix.rect()))

    def wheelEvent(self, ev) -> None:
        """Handle mouse wheel events for zooming."""
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = pow(1.0015, delta)  # 平滑倍率
        self.scale(factor, factor)

    def mousePressEvent(self, ev) -> None:
        """Handle mouse press events for panning."""
        if ev.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # 轉送成左鍵給 QGraphicsView 內部開始拖曳
            fake = type(ev)(
                QEvent.MouseButtonPress,
                ev.position(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            super().mousePressEvent(fake)
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        """Handle mouse release events to stop panning."""
        if ev.button() == Qt.MouseButton.MiddleButton:
            fake = type(ev)(
                QEvent.MouseButtonRelease,
                ev.position(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
            super().mouseReleaseEvent(fake)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)

    def reset_view(self) -> None:
        """Reset the view transform and center on the image."""
        self.setTransform(QTransform())
        if self._pix_item is not None:
            self.centerOn(self._pix_item)

    def map_widget_to_image(self, p: QPoint) -> Optional[Tuple[int, int]]:
        """Map a widget coordinate to image coordinates."""
        if self._pix_item is None:
            return None
        scene_pt = self.mapToScene(p)
        img_x = int(scene_pt.x())
        img_y = int(scene_pt.y())
        rect = self._pix_item.pixmap().rect()
        if not rect.contains(img_x, img_y):
            return None
        img_x = max(0, min(img_x, rect.width() - 1))
        img_y = max(0, min(img_y, rect.height() - 1))
        return img_x, img_y


class SegmentationViewer(QMainWindow):
    """Main window for the segmentation viewer, allowing interactive mask selection and saving."""
    
    # Signal emitted when the window is closed
    closed = Signal()

    def closeEvent(self, event):
        """Handle window close event."""
        self.closed.emit()
        super().closeEvent(event)

    def __init__(
        self,
        parent: Optional[QWidget],
        image_paths: List[Path],
        compute_masks_fn: Callable[
            [Path, int, float], Tuple[np.ndarray, List[np.ndarray], List[float]]
        ],
        params_defaults: Optional[Dict[str, float]] = None,
        title: str = "分割檢視",
        path_manager: Optional["PathManager"] = None,
    ) -> None:
        super().__init__(parent)
        print("DEBUG: SegmentationViewer initialized (v2)")

        self.setWindowTitle(title)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowModality(Qt.NonModal)
        
        # 視窗最大化顯示
        self.showMaximized()

        self.image_paths: List[Path] = list(image_paths)
        self.idx: int = 0
        self.compute_masks_fn = compute_masks_fn
        self.pm = path_manager  # 保存 PathManager 實例
        self.params = {
            "points_per_side": int((params_defaults or {}).get("points_per_side", 32)),
            "pred_iou_thresh": float((params_defaults or {}).get("pred_iou_thresh", 0.88)),
        }
        self.cache: Dict[Path, Tuple[np.ndarray, List[np.ndarray], List[float]]] = {}
        self.selected_indices: set[int] = set()
        self._hover_idx: Optional[int] = None

        # image view
        self.view = ImageView(self)
        self.view.viewport().installEventFilter(self)  # hover/點選 hit test

        # 右側群組 UI
        # 右側群組 UI
        grp_nav = QGroupBox("影像切換")
        self.btn_prev = QPushButton("◀ (PageUp)")
        self.btn_prev.setToolTip("切換至上一張影像")
        self.btn_next = QPushButton("▶ (PageDown)")
        self.btn_next.setToolTip("切換至下一張影像")
        self.btn_reset_view = QPushButton("🔄")
        self.btn_reset_view.setToolTip("重設影像縮放與位置")
        lay_nav = QHBoxLayout()
        lay_nav.addWidget(self.btn_prev)
        lay_nav.addWidget(self.btn_next)
        lay_nav.addWidget(self.btn_reset_view)
        grp_nav.setLayout(lay_nav)

        grp_crop = QGroupBox("裁切設定")
        self.rb_full = QRadioButton("全圖")
        self.rb_full.setToolTip("輸出整張原始圖片尺寸")
        self.rb_bbox = QRadioButton("僅物件")
        self.rb_bbox.setToolTip("僅輸出包含物件的最小矩形範圍")
        self.rb_bbox.setChecked(True)
        self.crop_group = QButtonGroup(self)
        self.crop_group.addButton(self.rb_full, 0)
        self.crop_group.addButton(self.rb_bbox, 1)
        lay_crop = QVBoxLayout()
        lay_crop.addWidget(self.rb_bbox)
        lay_crop.addWidget(self.rb_full)
        grp_crop.setLayout(lay_crop)

        grp_mode = QGroupBox("存檔方式")
        self.rb_mode_union = QRadioButton("合併")
        self.rb_mode_union.setToolTip("將所有選取物件合併為單一圖檔")
        self.rb_mode_indiv = QRadioButton("個別")
        self.rb_mode_indiv.setToolTip("每個選取物件分別存為獨立圖檔")
        self.rb_mode_indiv.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_mode_indiv, 0)
        self.mode_group.addButton(self.rb_mode_union, 1)
        lay_mode = QVBoxLayout()
        lay_mode.addWidget(self.rb_mode_indiv)
        lay_mode.addWidget(self.rb_mode_union)
        grp_mode.setLayout(lay_mode)
        # [新增] 顯示模式切換群組，放在 grp_mode 定義之後
        # [新增] 顯示模式切換群組，放在 grp_mode 定義之後
        grp_display = QGroupBox("檢視模式")
        self.rb_show_mask = QRadioButton("遮罩")
        self.rb_show_mask.setToolTip("顯示語意分割遮罩 (Mask)")
        self.rb_show_bbox = QRadioButton("外框")
        self.rb_show_bbox.setToolTip("顯示物件外接矩形 (Bounding Box)")
        self.rb_show_mask.setChecked(True)

        self.display_group = QButtonGroup(self)
        self.display_group.addButton(self.rb_show_mask, 0)  # 0=遮罩
        self.display_group.addButton(self.rb_show_bbox, 1)  # 1=BBox

        lay_display = QVBoxLayout()
        lay_display.addWidget(self.rb_show_mask)
        lay_display.addWidget(self.rb_show_bbox)
        grp_display.setLayout(lay_display)

        # 切換顯示模式即時重繪
        self.display_group.idClicked.connect(lambda _id: self._update_canvas())

        # [新增] 輸出模式切換時也要重繪（為了 BBox 聯集時只畫一個框）
        self.mode_group.idClicked.connect(lambda _id: self._update_canvas())

        # [新增] 建立在 grp_mode 與 grp_save 之間，與其它群組同一層級
        # [新增] 建立在 grp_mode 與 grp_save 之間，與其它群組同一層級
        grp_labels = QGroupBox("標註檔")
        
        # YOLO 格式
        self.chk_yolo_det = QCheckBox("YOLO (偵測)")
        self.chk_yolo_det.setToolTip("輸出 YOLO 格式的物件偵測標註 (BBox)")
        self.chk_yolo_seg = QCheckBox("YOLO (分割)")
        self.chk_yolo_seg.setToolTip("輸出 YOLO 格式的實例分割標註 (Polygon)")
        
        # COCO 格式
        self.chk_coco = QCheckBox("COCO")
        self.chk_coco.setToolTip("輸出 COCO JSON 格式標註")
        
        # Pascal VOC 格式
        self.chk_voc = QCheckBox("VOC")
        self.chk_voc.setToolTip("輸出 Pascal VOC XML 格式標註")
        
        # LabelMe 格式
        self.chk_labelme = QCheckBox("LabelMe")
        self.chk_labelme.setToolTip("輸出 LabelMe JSON 格式標註")

        self.spn_cls = QSpinBox()
        self.spn_cls.setRange(0, 999)
        self.spn_cls.setValue(0)
        self.spn_cls.setToolTip("設定輸出標註的類別 ID (Class ID)")

        lay_labels = QFormLayout()
        lay_labels.addRow(self.chk_yolo_det)
        lay_labels.addRow(self.chk_yolo_seg)
        lay_labels.addRow(self.chk_coco)
        lay_labels.addRow(self.chk_voc)
        lay_labels.addRow(self.chk_labelme)
        lay_labels.addRow("類別 ID", self.spn_cls)
        grp_labels.setLayout(lay_labels)

        # 顏色設定（初始化，UI 移至菜單）
        self.mask_color = [0, 255, 0]  # 預設綠色 (BGR)
        self.bbox_color = [0, 255, 0]  # 預設綠色 (BGR)

        grp_save = QGroupBox("輸出")
        
        # 輸出路徑設定
        output_path_layout = QHBoxLayout()
        output_path_label = QLabel("路徑:")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("預設為原影像同層資料夾")
        self.output_path_edit.setText("")  # 空白表示使用預設
        self.output_path_edit.setToolTip("設定檔案輸出的目標資料夾")
        btn_browse_output = QPushButton("瀏覽...")
        btn_browse_output.clicked.connect(self._browse_output_path)
        
        output_path_layout.addWidget(output_path_label)
        output_path_layout.addWidget(self.output_path_edit, 1)
        output_path_layout.addWidget(btn_browse_output)
        
        # 輸出格式選擇（重新命名）
        format_layout = QHBoxLayout()
        format_label = QLabel("格式:")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG", "BMP"])
        self.format_combo.setCurrentIndex(0)  # 預設 PNG
        self.format_combo.setToolTip("選擇輸出影像的檔案格式")
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo, 1)
        
        self.btn_save_selected = QPushButton("💾 選取物件")
        self.btn_save_selected.setToolTip("僅儲存目前已選取的物件")
        self.btn_save_all = QPushButton("💾 全部物件")
        self.btn_save_all.setToolTip("自動儲存影像中偵測到的所有物件")
        self.lbl_selected = QLabel("已選物件：0")
        self.lbl_selected.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lay_save = QVBoxLayout()
        lay_save.addLayout(output_path_layout)
        lay_save.addLayout(format_layout)
        lay_save.addWidget(self.btn_save_selected)
        lay_save.addWidget(self.btn_save_all)
        lay_save.addWidget(self.lbl_selected)
        grp_save.setLayout(lay_save)

        # 參數設定（移至菜單，但保留變數）

        # 使用 DockWidget 讓右側面板可拖曳
        right_box = QVBoxLayout()
        right_box.addWidget(grp_nav)
        right_box.addWidget(grp_crop)
        right_box.addWidget(grp_mode)
        right_box.addWidget(grp_display)
        right_box.addWidget(grp_labels)
        right_box.addWidget(grp_save)
        right_box.addStretch(1)
        
        right_widget = QWidget()
        right_widget.setLayout(right_box)
        
        # 建立可拖曳的 Dock
        self.dock_controls = QDockWidget("控制面板", self)
        self.dock_controls.setWidget(right_widget)
        self.dock_controls.setFeatures(
            QDockWidget.DockWidgetMovable | 
            QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_controls)

        # 設定中央widget為影像檢視
        self.setCentralWidget(self.view)
        
        # 建立菜單欄
        self._create_menu_bar()

        # connect
        self.btn_reset_view.clicked.connect(self.view.reset_view)
        self.btn_prev.clicked.connect(self._prev_image)
        self.btn_next.clicked.connect(self._next_image)
        self.btn_save_selected.clicked.connect(self._save_selected)
        self.btn_save_all.clicked.connect(self._save_all)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.status = StatusFooter.install(self)
        self._spawned_views: list[SegmentationViewer] = []
        self.status.message("準備就緒")
        
        # 設定快捷鍵
        self._setup_shortcuts()
        
        self._start_batch_processing()
    
    def _save_all(self) -> None:
        """Save all masks for the current image."""
        if not self.image_paths:
            return
        path = self.image_paths[self.idx]
        if path not in self.cache:
            return
        _, masks, _ = self.cache[path]
        if not masks:
            QMessageBox.information(self, "提示", "目前影像沒有任何分割目標")
            return
            
        # Confirm with user
        ret = QMessageBox.question(
            self, "確認儲存", 
            f"確定要儲存全部 {len(masks)} 個目標嗎？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if ret != QMessageBox.StandardButton.Yes:
            return

        # Reuse _save_indices logic
        self._save_indices(list(range(len(masks))))
    
    def _create_menu_bar(self):
        """建立菜單欄"""
        menubar = self.menuBar()
        
        # 選項菜單
        options_menu = menubar.addMenu("選項")
        
        # 顏色設定
        color_action = QAction("顏色設定...", self)
        color_action.triggered.connect(self._show_color_dialog)
        options_menu.addAction(color_action)
        
        # 分割參數
        params_action = QAction("分割參數...", self)
        params_action.triggered.connect(self._show_params_dialog)
        options_menu.addAction(params_action)

    def _setup_shortcuts(self):
        """設定快捷鍵"""
        from modules.presentation.qt.shortcut_manager import ShortcutManager
        
        try:
            shortcut_manager = ShortcutManager()
            
            # 上一張
            prev_key = shortcut_manager.get_shortcut('nav.prev')
            if prev_key:
                shortcut_prev = QShortcut(QKeySequence(prev_key), self)
                shortcut_prev.activated.connect(self._prev_image)
            
            # 下一張
            next_key = shortcut_manager.get_shortcut('nav.next')
            if next_key:
                shortcut_next = QShortcut(QKeySequence(next_key), self)
                shortcut_next.activated.connect(self._next_image)
            
            # 儲存選取
            save_key = shortcut_manager.get_shortcut('save.selected')
            if save_key:
                shortcut_save = QShortcut(QKeySequence(save_key), self)
                shortcut_save.activated.connect(self._save_selected)
            
            # 重設檢視
            reset_key = shortcut_manager.get_shortcut('view.reset')
            if reset_key:
                shortcut_reset = QShortcut(QKeySequence(reset_key), self)
                shortcut_reset.activated.connect(self.view.reset_view)
                
        except Exception as e:
            logger.warning(f"載入快捷鍵失敗: {e}")






    def _start_batch_processing(self):
        if not self.image_paths:
            return

        from modules.presentation.qt.progress_dialog import ThemedProgressDialog
        self.batch_progress = ThemedProgressDialog("批次處理中", "準備開始...", self)
        self.batch_progress.set_range(0, len(self.image_paths))
        self.batch_progress.show()
        
        self.batch_worker = BatchSegmentationWorker(
            self.compute_masks_fn,
            self.image_paths,
            int(self.params["points_per_side"]),
            float(self.params["pred_iou_thresh"])
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
        if hasattr(self, 'batch_progress'):
            self.batch_progress.close()
        # Load the first image (now likely cached)
        self._load_current_image(recompute=False)

    # ---- load / recompute ----

    def _load_current_image(self, recompute: bool = False) -> None:
        """Load the current image and compute/load masks."""
        if not self.image_paths:
            return
        path = self.image_paths[self.idx]
        
        # Check for cached results
        cache_file = path.parent / f"{path.stem}.sam_cache.npz"
        
        if not recompute and cache_file.exists():
            print(f"DEBUG: Found cache file {cache_file}")
            # Load from cache
            try:
                self.status.message(f"載入快取: {Path(path).name}")
                cached = np.load(cache_file, allow_pickle=True)
                bgr = cv2.imread(str(path))
                # masks are stored as individual arrays in the npz
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
                print(f"DEBUG: Cache load failed: {e}")
                logger.warning(f"快取載入失敗: {e}，重新分割")
        else:
            print(f"DEBUG: Cache missing or recompute=True. Exists: {cache_file.exists()}, Recompute: {recompute}")
        
        if recompute or path not in self.cache:
            from modules.presentation.qt.progress_dialog import ThemedProgressDialog
            self.progress = ThemedProgressDialog(
                "處理中", 
                f"正在分割影像 ({self.idx + 1}/{len(self.image_paths)}):\n{Path(path).name}", 
                self
            )
            self.progress.show()
            
            # Disable interaction
            self.setEnabled(False)
            
            self.worker = SegmentationWorker(
                self.compute_masks_fn,
                path,
                int(self.params["points_per_side"]),
                float(self.params["pred_iou_thresh"])
            )
            self.worker.finished.connect(lambda b, m, s: self._on_worker_finished(path, b, m, s))
            self.worker.error.connect(self._on_worker_error)
            self.worker.start()
            return

        self._update_ui_after_load(path)

    def _on_worker_finished(self, path, bgr, masks, scores):
        if hasattr(self, 'progress'):
            self.progress.close()
        self.setEnabled(True)
        
        H, W = bgr.shape[:2]
        self.status.set_image_resolution(W, H)
        self.status.set_cursor_xy(None, None)
        
        # Save to cache
        cache_file = path.parent / f"{path.stem}.sam_cache.npz"
        try:
            cache_data = {'scores': np.array(scores)}
            for i, m in enumerate(masks):
                cache_data[f'mask_{i}'] = m
            np.savez_compressed(cache_file, **cache_data)
            logger.info(f"已儲存快取: {cache_file}")
        except Exception as e:
            logger.warning(f"快取儲存失敗: {e}")

        masks = [(m > 0).astype(np.uint8) for m in masks]
        self.cache[path] = (bgr, masks, scores)
        
        self._update_ui_after_load(path)

    def _on_worker_error(self, err_msg):
        if hasattr(self, 'progress'):
            self.progress.close()
        self.setEnabled(True)
        logger.error(f"Segmentation failed: {err_msg}")
        QMessageBox.critical(self, "分割失敗", f"無法分割：{err_msg}")

    def _update_ui_after_load(self, path):
        # 嘗試載入已儲存的標註
        annotation_file = path.parent / f"{path.stem}_annotations.json"
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.selected_indices = set(data.get('selected_indices', []))
                    logger.info(f"載入標註: {len(self.selected_indices)} 個選取的遮罩")
                    self.status.message(f"載入標註: {len(self.selected_indices)} 個已選取的遮罩")
            except Exception as e:
                logger.warning(f"載入標註失敗: {e}")
                self.selected_indices.clear()
        else:
            self.selected_indices.clear()
        
        self._hover_idx = None
        self._update_selected_count()
        self._update_nav_buttons()
        self._update_canvas()  # 確保畫布更新以顯示已選取的遮罩
        
        if path in self.cache:
            num_masks = len(self.cache[path][1])
            num_selected = len(self.selected_indices)
            self.status.message(
                f"載入完成：{Path(path).name}，共有 {num_masks} 個候選遮罩，已選取 {num_selected} 個"
            )
    
    def _show_color_dialog(self):
        """顯示顏色設定對話框"""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("顏色設定")
        dialog.setModal(True)
        
        layout = QFormLayout()
        
        # Mask 顏色
        mask_layout = QHBoxLayout()
        btn_mask = QPushButton("選擇顏色")
        lbl_mask = QLabel()
        lbl_mask.setFixedSize(30, 20)
        lbl_mask.setStyleSheet(f"background-color: rgb({self.mask_color[2]}, {self.mask_color[1]}, {self.mask_color[0]}); border: 1px solid #666;")
        
        def choose_mask():
            color = QColorDialog.getColor(QColor(self.mask_color[2], self.mask_color[1], self.mask_color[0]), self, "選擇 Mask 顏色")
            if color.isValid():
                self.mask_color = [color.blue(), color.green(), color.red()]
                lbl_mask.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); border: 1px solid #666;")
                self._update_canvas()
        
        btn_mask.clicked.connect(choose_mask)
        mask_layout.addWidget(btn_mask)
        mask_layout.addWidget(lbl_mask)
        mask_layout.addStretch()
        
        # BBox 顏色
        bbox_layout = QHBoxLayout()
        btn_bbox = QPushButton("選擇顏色")
        lbl_bbox = QLabel()
        lbl_bbox.setFixedSize(30, 20)
        lbl_bbox.setStyleSheet(f"background-color: rgb({self.bbox_color[2]}, {self.bbox_color[1]}, {self.bbox_color[0]}); border: 1px solid #666;")
        
        def choose_bbox():
            color = QColorDialog.getColor(QColor(self.bbox_color[2], self.bbox_color[1], self.bbox_color[0]), self, "選擇 BBox 顏色")
            if color.isValid():
                self.bbox_color = [color.blue(), color.green(), color.red()]
                lbl_bbox.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); border: 1px solid #666;")
                self._update_canvas()
        
        btn_bbox.clicked.connect(choose_bbox)
        bbox_layout.addWidget(btn_bbox)
        bbox_layout.addWidget(lbl_bbox)
        bbox_layout.addStretch()
        
        layout.addRow("Mask 顏色:", mask_layout)
        layout.addRow("BBox 顏色:", bbox_layout)
        
        # 按鈕
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def _show_params_dialog(self):
        """顯示分割參數對話框"""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("分割參數設定")
        dialog.setModal(True)
        
        layout = QFormLayout()
        
        # Points per side
        spn_points = QSpinBox()
        spn_points.setRange(4, 128)
        spn_points.setValue(self.params["points_per_side"])
        
        # IoU threshold
        spn_iou = QDoubleSpinBox()
        spn_iou.setRange(0.1, 0.99)
        spn_iou.setSingleStep(0.01)
        spn_iou.setValue(self.params["pred_iou_thresh"])
        
        layout.addRow("Points per side:", spn_points)
        layout.addRow("Pred IoU threshold:", spn_iou)
        
        # 按鈕
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.Accepted:
            self.params["points_per_side"] = spn_points.value()
            self.params["pred_iou_thresh"] = spn_iou.value()
            # 詢問是否立即重算
            ret = QMessageBox.question(
                self, "套用參數",
                "是否使用新參數重新計算當前影像？",
                QMessageBox.Yes | QMessageBox.No
            )
            if ret == QMessageBox.Yes:
                # 清理 CUDA 記憶體
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                self._load_current_image(recompute=True)
                if hasattr(self, 'status'):
                    self.status.message_temp("參數已套用並重算", 1800)
    
    def _choose_mask_color(self):
        """選擇 Mask 顏色（舊版，保留向後兼容）"""
        self._show_color_dialog()
    
    def _choose_bbox_color(self):
        """選擇 BBox 顏色（舊版，保留向後兼容）"""
        self._show_color_dialog()
    
    def _browse_output_path(self):
        """瀏覽並選擇輸出路徑"""
        current_path = self.output_path_edit.text()
        if not current_path and self.image_paths:
            # 預設為第一張影像的目錄
            current_path = str(self.image_paths[0].parent)
        
        folder = QFileDialog.getExistingDirectory(
            self,
            "選擇輸出資料夾",
            current_path if current_path else str(Path.home())
        )
        if folder:
            self.output_path_edit.setText(folder)

    def _apply_params(self) -> None:
        """Apply new segmentation parameters and recompute masks."""
        pps = int(self.spn_points.value())
        iou = float(self.spn_iou.value())
        self.params["points_per_side"] = pps
        self.params["pred_iou_thresh"] = iou
        self._load_current_image(recompute=True)
        self.status.message_temp("參數已套用", 1800)

    # 若你有「視圖置入」按鈕或勾選, 也寫回
    def on_fit_on_open_toggled(self, on: bool):
        """Handle toggle of 'fit on open' setting."""
        self.params["fit_on_open"] = bool(on)

    # ---- navigation ----
    def _update_nav_buttons(self) -> None:
        """Update the enabled state of navigation buttons."""
        n = len(self.image_paths)
        self.btn_prev.setEnabled(self.idx > 0 and n > 0)
        self.btn_next.setEnabled(self.idx < n - 1 and n > 0)

    def _prev_image(self) -> None:
        """Navigate to the previous image."""
        if self.idx > 0:
            self.idx -= 1
            self._load_current_image(recompute=False)

    def _next_image(self) -> None:
        """Navigate to the next image."""
        if self.idx < len(self.image_paths) - 1:
            self.idx += 1
            self._load_current_image(recompute=False)

    # ---- mapping / hit ----
    def _map_widget_to_image(self, p: QPoint) -> Optional[Tuple[int, int]]:
        """Map widget coordinates to image coordinates."""
        return self.view.map_widget_to_image(p)

    def _hit_test_xy(self, masks: List[np.ndarray], x: int, y: int) -> Optional[int]:
        """Find the smallest mask index that contains the given (x, y) coordinate."""
        if not masks:
            return None
        if y < 0 or y >= masks[0].shape[0] or x < 0 or x >= masks[0].shape[1]:
            return None
        hits = [i for i, m in enumerate(masks) if m[y, x] > 0]
        if not hits:
            return None
        areas = [(i, int(masks[i].sum())) for i in hits]
        areas.sort(key=lambda t: t[1])
        return areas[0][0]

    # ---- draw ----
    def _update_canvas(self) -> None:
        """Redraw the image canvas with current masks and selections."""
        if not self.image_paths:
            return
        path = self.image_paths[self.idx]
        if path not in self.cache:
            return
        bgr, masks, _ = self.cache[path]
        base = bgr.copy()

        # 顯示模式: 0=遮罩, 1=BBox
        disp_id = self.display_group.checkedId() if hasattr(self, "display_group") else 0
        use_bbox = disp_id == 1

        # 輸出模式: 0=個別, 1=聯集
        mode_id = self.mode_group.checkedId() if hasattr(self, "mode_group") else 0
        is_union = mode_id == 1

        if not use_bbox:
            # 遮罩高亮模式
            if self.selected_indices:
                sel_union = np.zeros(base.shape[:2], dtype=np.uint8)
                for i in self.selected_indices:
                    if 0 <= i < len(masks):
                        sel_union = np.maximum(sel_union, masks[i])
                m = sel_union > 0
                # 使用自訂 mask 顏色
                mask_color_bgr = np.array(self.mask_color, dtype=np.uint8)
                base[m] = (base[m] * 0.4 + mask_color_bgr * 0.6).astype(np.uint8)

            if self._hover_idx is not None and 0 <= self._hover_idx < len(masks):
                hover_mask = masks[self._hover_idx]
                # 確保 mask 維度正確
                if hover_mask.shape[:2] == base.shape[:2]:
                    m = hover_mask > 0
                    mask_color_bgr = np.array(self.mask_color, dtype=np.uint8)
                    base[m] = (base[m] * 0.2 + mask_color_bgr * 0.8).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        # 使用自訂 bbox 顏色繪製輪廓
                        bbox_color_tuple = tuple(int(c) for c in self.bbox_color)
                        cv2.polylines(base, contours, True, bbox_color_tuple, 2)

        else:
            # BBox 模式
            H, W = base.shape[:2]
            bbox_color_tuple = tuple(int(c) for c in self.bbox_color)
            if is_union and self.selected_indices:
                # 聯集 + BBox: 只畫一個框線
                union_mask = np.zeros((H, W), dtype=np.uint8)
                for i in self.selected_indices:
                    if 0 <= i < len(masks):
                        union_mask = np.maximum(union_mask, masks[i])
                x, y, w, h = compute_bbox(union_mask > 0)
                cv2.rectangle(base, (x, y), (x + w, y + h), bbox_color_tuple, 3)
            else:
                # 個別 + BBox: 已選畫細線, 懸浮畫粗線
                for i in self.selected_indices:
                    if 0 <= i < len(masks):
                        x, y, w, h = compute_bbox(masks[i] > 0)
                        cv2.rectangle(base, (x, y), (x + w, y + h), bbox_color_tuple, 2)
                if self._hover_idx is not None and 0 <= self._hover_idx < len(masks):
                    x, y, w, h = compute_bbox(masks[self._hover_idx] > 0)
                    cv2.rectangle(base, (x, y), (x + w, y + h), bbox_color_tuple, 3)

        if hasattr(self, "status"):
            self.status.set_display_info(
                "BBox" if use_bbox else "遮罩", is_union, len(self.selected_indices)
            )
        self.view.set_image_bgr(base)

    def _update_selected_count(self) -> None:
        """Update the label showing the number of selected masks."""
        self.lbl_selected.setText(f"已選物件：{len(self.selected_indices)}")

    # ---- save ----
    def _save_selected(self) -> None:
        """Save the selected masks based on the current mode (union or individual)."""
        if not self.selected_indices and self._hover_idx is not None:
            ret = QMessageBox.question(
                self, "未選擇目標", "尚未選擇任何目標，是否儲存目前滑鼠指向的目標？"
            )
            if ret == QMessageBox.StandardButton.Yes:
                self._save_one(self._hover_idx)
            return
        if not self.selected_indices:
            QMessageBox.information(self, "提示", "尚未選擇任何目標")
            return
        if self.rb_mode_union.isChecked():
            self._save_union(sorted(self.selected_indices))
        else:
            self._save_indices(sorted(self.selected_indices))

    def _save_union(self, indices: List[int]) -> None:
        """Save the union of multiple masks as a single image."""
        path = self.image_paths[self.idx]
        bgr, masks, _ = self.cache[path]
        
        # 使用使用者設定的輸出路徑，或預設為原影像同層資料夾
        custom_path = self.output_path_edit.text().strip()
        if custom_path:
            out_dir = Path(custom_path)
        else:
            out_dir = Path(path).parent
        
        # 確保目錄存在
        out_dir.mkdir(parents=True, exist_ok=True)

        # [新增] 儲存標註狀態 (JSON)
        self._save_annotations_json(path, out_dir)

        H, W = bgr.shape[:2]
        union_mask = np.zeros((H, W), dtype=np.uint8)
        for i in indices:
            if 0 <= i < len(masks):
                union_mask = np.maximum(union_mask, (masks[i] > 0).astype(np.uint8))

        base_name = f"{path.stem}_union"
        
        # 準備輸出影像 (BGRA)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = union_mask * 255

        if self.rb_bbox.isChecked():
            # 裁成聯集的外接矩形
            x, y, w, h = compute_bbox(union_mask > 0)
            crop = bgra[y : y + h, x : x + w]
            img_h, img_w = h, w
            # 標註以裁後影像為座標系
            boxes = [(0, 0, w, h)]
            poly = self._compute_polygon(union_mask[y : y + h, x : x + w])
            polys = [poly]
        else:
            # 原圖大小
            crop = bgra
            img_h, img_w = H, W
            x, y, w, h = compute_bbox(union_mask > 0)
            boxes = [(x, y, w, h)]
            poly = self._compute_polygon(union_mask)
            polys = [poly]

        # 取得選擇的格式
        fmt = self.format_combo.currentText().lower()
        if fmt == "jpg":
            # JPG 不支援透明度，轉回 BGR
            save_img = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
            ext = ".jpg"
        elif fmt == "bmp":
            save_img = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
            ext = ".bmp"
        else:
            save_img = crop
            ext = ".png"

        save_path = out_dir / f"{base_name}{ext}"
        ok, buf = cv2.imencode(ext, save_img)
        if ok:
            save_path.write_bytes(buf.tobytes())
            
            # 寫出各種標註格式
            self._write_yolo_labels(out_dir, base_name, boxes, polys, img_w, img_h)
            self._write_coco_json(out_dir, base_name, boxes, polys, img_w, img_h)
            self._write_voc_xml(out_dir, base_name, boxes, img_w, img_h, save_path.name)
            self._write_labelme_json(out_dir, base_name, polys, img_w, img_h, save_path.name)
            
            QMessageBox.information(self, "完成", f"已儲存聯集影像至：\n{save_path}")
            self.status.message("儲存完成")
        else:
            QMessageBox.warning(self, "失敗", "影像編碼失敗")

    def _save_indices(self, indices: List[int]) -> None:
        """Save selected masks as individual images."""
        path = self.image_paths[self.idx]
        bgr, masks, _ = self.cache[path]
        
        custom_path = self.output_path_edit.text().strip()
        if custom_path:
            out_dir = Path(custom_path)
        else:
            out_dir = Path(path).parent
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # [新增] 儲存標註狀態 (JSON)
        self._save_annotations_json(path, out_dir)
        
        saved_count = 0
        H, W = bgr.shape[:2]
        
        # 取得選擇的格式
        fmt = self.format_combo.currentText().lower()
        ext = f".{fmt}"

        for i in indices:
            if not (0 <= i < len(masks)):
                continue
            m = masks[i] > 0
            
            # 準備輸出影像 (BGRA)
            bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = m.astype(np.uint8) * 255
            
            base_name = f"{path.stem}_{i:03d}"
            
            if self.rb_bbox.isChecked():
                x, y, w, h = compute_bbox(m)
                crop = bgra[y : y + h, x : x + w]
                img_h, img_w = h, w
                boxes = [(0, 0, w, h)]
                poly = self._compute_polygon(m[y : y + h, x : x + w])
                polys = [poly]
            else:
                crop = bgra
                img_h, img_w = H, W
                x, y, w, h = compute_bbox(m)
                boxes = [(x, y, w, h)]
                poly = self._compute_polygon(m)
                polys = [poly]
            
            if fmt in ["jpg", "bmp"]:
                save_img = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
            else:
                save_img = crop
                
            save_path = out_dir / f"{base_name}{ext}"
            ok, buf = cv2.imencode(ext, save_img)
            if ok:
                save_path.write_bytes(buf.tobytes())
                saved_count += 1
                
                # 寫出各種標註格式
                self._write_yolo_labels(out_dir, base_name, boxes, polys, img_w, img_h)
                self._write_coco_json(out_dir, base_name, boxes, polys, img_w, img_h)
                self._write_voc_xml(out_dir, base_name, boxes, img_w, img_h, save_path.name)
                self._write_labelme_json(out_dir, base_name, polys, img_w, img_h, save_path.name)
        
        if saved_count > 0:
            QMessageBox.information(self, "完成", f"已儲存 {saved_count} 個物件影像")
            self.status.message(f"已儲存 {saved_count} 個物件")
        else:
            QMessageBox.warning(self, "提示", "沒有儲存任何檔案")

    def _write_coco_json(self, out_dir, base_name, boxes, polys, w, h):
        """Export to COCO JSON format."""
        if not getattr(self, "chk_coco", None) or not self.chk_coco.isChecked():
            return
            
        cls_id = int(self.spn_cls.value()) if hasattr(self, "spn_cls") else 0
        
        coco_data = {
            "images": [{"id": 1, "file_name": f"{base_name}.png", "width": w, "height": h}],
            "annotations": [],
            "categories": [{"id": cls_id, "name": f"class_{cls_id}"}]
        }
        
        for i, (box, poly) in enumerate(zip(boxes, polys)):
            x, y, bw, bh = box
            segmentation = []
            if poly is not None and len(poly) > 0:
                segmentation = [poly.flatten().tolist()]
                
            ann = {
                "id": i + 1,
                "image_id": 1,
                "category_id": cls_id,
                "bbox": [x, y, bw, bh],
                "segmentation": segmentation,
                "area": bw * bh,
                "iscrowd": 0
            }
            coco_data["annotations"].append(ann)
            
        (out_dir / f"{base_name}_coco.json").write_text(json.dumps(coco_data, indent=2), encoding="utf-8")

    def _write_voc_xml(self, out_dir, base_name, boxes, w, h, filename):
        """Export to Pascal VOC XML format."""
        if not getattr(self, "chk_voc", None) or not self.chk_voc.isChecked():
            return
            
        cls_id = int(self.spn_cls.value()) if hasattr(self, "spn_cls") else 0
        cls_name = f"class_{cls_id}"
        
        import xml.etree.ElementTree as ET
        
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = out_dir.name
        ET.SubElement(root, "filename").text = filename
        ET.SubElement(root, "path").text = filename  # 使用相對路徑 (僅檔名)
        
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = "3"
        
        for x, y, bw, bh in boxes:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = cls_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(x)
            ET.SubElement(bndbox, "ymin").text = str(y)
            ET.SubElement(bndbox, "xmax").text = str(x + bw)
            ET.SubElement(bndbox, "ymax").text = str(y + bh)
            
        tree = ET.ElementTree(root)
        tree.write(out_dir / f"{base_name}.xml", encoding="utf-8", xml_declaration=True)

    def _write_labelme_json(self, out_dir, base_name, polys, w, h, filename):
        """Export to LabelMe JSON format."""
        if not getattr(self, "chk_labelme", None) or not self.chk_labelme.isChecked():
            return
            
        cls_id = int(self.spn_cls.value()) if hasattr(self, "spn_cls") else 0
        cls_name = f"class_{cls_id}"
        
        shapes = []
        for poly in polys:
            if poly is not None and len(poly) > 0:
                shape = {
                    "label": cls_name,
                    "points": poly.tolist(),
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                shapes.append(shape)
                
        data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": shapes,
            "imagePath": filename,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }
        
        (out_dir / f"{base_name}_labelme.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _save_one(self, idx: int) -> None:
        """Save a single mask index."""
        self._save_indices([idx])

    # ---- event filter on view viewport ----
    def eventFilter(self, obj, event):
        """Filter events to handle mouse interactions on the view viewport."""
        if obj is self.view.viewport():
            try:

                def _pt(ev):
                    return ev.position().toPoint() if hasattr(ev, "position") else ev.pos()

                if event.type() == QEvent.MouseMove:
                    pos = _pt(event)
                    img_xy = self._map_widget_to_image(pos)
                    if img_xy is None:
                        if self._hover_idx is not None:
                            self._hover_idx = None
                            self._update_canvas()
                        if hasattr(self, 'status'):
                            self.status.set_cursor_xy(None, None)  # 清空
                    else:
                        x, y = img_xy
                        path = self.image_paths[self.idx]
                        _, masks, _ = self.cache[path]
                        self._hover_idx = self._hit_test_xy(masks, x, y)
                        self._update_canvas()
                        if hasattr(self, 'status'):
                            self.status.set_cursor_xy(x, y)  # 即時更新游標座標
                    return False
                if event.type() == QEvent.MouseButtonPress:
                    pos = _pt(event)
                    img_xy = self._map_widget_to_image(pos)
                    if img_xy is None:
                        return False
                    x, y = img_xy
                    path = self.image_paths[self.idx]
                    _, masks, _ = self.cache[path]
                    tgt = self._hit_test_xy(masks, x, y)
                    if tgt is None:
                        return False
                    if event.button() == Qt.MouseButton.LeftButton:
                        self.selected_indices.add(tgt)
                        self._update_selected_count()
                        self._update_canvas()
                    elif event.button() == Qt.MouseButton.RightButton:
                        if tgt in self.selected_indices:
                            self.selected_indices.remove(tgt)
                            self._update_selected_count()
                            self._update_canvas()
                    return False
            except Exception:
                logger.warning("滑鼠事件處理發生例外", exc_info=True)
                return False
        return super().eventFilter(obj, event)

    # 新增：在 SegmentationViewer 類別中加入兩個 helper
    def _collect_images_with_pivot_first(self, pivot: Path) -> List[Path]:
        """Collect images from the same directory, placing the pivot image first."""
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
        imgs = [
            p for p in sorted(pivot.parent.glob("*")) if p.is_file() and p.suffix.lower() in exts
        ]
        pv = pivot.resolve() if hasattr(pivot, "resolve") else pivot
        head = [p for p in imgs if (p.resolve() if hasattr(p, "resolve") else p) == pv]
        tail = [p for p in imgs if (p.resolve() if hasattr(p, "resolve") else p) != pv]
        return (head or [pivot]) + tail



    def save_union_hotkey(self):
        """Slot for the save union shortcut."""
        if not self.selected_indices:
            QMessageBox.information(self, "提示", "尚未選擇任何目標")
            return
        self._save_union(sorted(self.selected_indices))

    def _save_annotations_json(self, image_path: Path, out_dir: Path) -> None:
        """Save current annotations (selected indices) to a JSON file."""
        try:
            data = {
                "image_path": image_path.name,
                "selected_indices": list(self.selected_indices)
            }
            # 儲存到與輸出影像相同的目錄，檔名為 [原始檔名]_annotations.json
            save_path = out_dir / f"{image_path.stem}_annotations.json"
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"已儲存標註狀態: {save_path}")
        except Exception as e:
            logger.error(f"儲存標註狀態失敗: {e}")

    # [新增] 放在 SegmentationViewer 類別內其它私有方法旁

    def _compute_polygon(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """回傳最大連通域的外輪廓座標，形狀為 (N,2)，整數像素座標。"""
        m = (mask > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        return c.reshape(-1, 2)  # (N,2)

    def _write_yolo_labels(
        self,
        out_dir: Path,
        base_name: str,
        boxes: List[Tuple[int, int, int, int]],
        polys: List[Optional[np.ndarray]],
        img_w: int,
        img_h: int,
    ) -> None:
        """依勾選輸出 YOLO 檢測與/或 YOLO 分割標註檔。兩者同時勾選時各自輸出到不同檔名。"""
        cls_id = int(self.spn_cls.value()) if hasattr(self, "spn_cls") else 0

        # YOLO 檢測: 每行 => cls xc yc w h (皆為 0~1)
        if getattr(self, "chk_yolo_det", None) and self.chk_yolo_det.isChecked():
            lines = []
            for x, y, w, h in boxes:
                if w <= 0 or h <= 0:
                    continue
                xc = (x + w / 2.0) / img_w
                yc = (y + h / 2.0) / img_h
                nw = w / img_w
                nh = h / img_h
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            if lines:
                (out_dir / f"{base_name}_yolo.txt").write_text("\n".join(lines), encoding="utf-8")

        # YOLO 分割: 每行 => cls x1 y1 x2 y2 ... (座標皆為 0~1)
        if getattr(self, "chk_yolo_seg", None) and self.chk_yolo_seg.isChecked():
            lines = []
            for poly in polys:
                if poly is None or len(poly) == 0:
                    continue
                pts = []
                for px, py in poly:
                    pts.append(f"{px / img_w:.6f} {py / img_h:.6f}")
                lines.append(f"{cls_id} " + " ".join(pts))
            if lines:
                (out_dir / f"{base_name}_seg.txt").write_text("\n".join(lines), encoding="utf-8")
