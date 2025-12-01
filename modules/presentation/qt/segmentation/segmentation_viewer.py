# modules/segmentation_viewer.py
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QDir, QEvent, QPoint, QRectF, Qt, QThread, Signal, QSize
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPixmap, QTransform, QKeySequence, QShortcut, QBrush, QPen, QCursor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
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
            logger.error(f"SegmentationWorker error processing {self.path}: {e}", exc_info=True)
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
                logger.error(f"BatchSegmentationWorker error processing {path}: {e}", exc_info=True)
        
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

    # Signals for drawing interaction
    drawing_started = Signal(int, int)  # x, y
    drawing_moved = Signal(int, int)    # x, y
    drawing_finished = Signal(int, int) # x, y

    def mousePressEvent(self, ev) -> None:
        """Handle mouse press events for panning or drawing."""
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
        elif ev.button() == Qt.MouseButton.LeftButton:
            # Check if we are in drawing mode (handled by parent logic via signals)
            # Map to image coordinates
            p = self.map_widget_to_image(ev.position().toPoint())
            if p:
                self.drawing_started.emit(p[0], p[1])
            super().mousePressEvent(ev)
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        """Handle mouse move events."""
        super().mouseMoveEvent(ev)
        if ev.buttons() & Qt.MouseButton.LeftButton:
            p = self.map_widget_to_image(ev.position().toPoint())
            if p:
                self.drawing_moved.emit(p[0], p[1])

    def mouseReleaseEvent(self, ev) -> None:
        """Handle mouse release events to stop panning or drawing."""
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
        elif ev.button() == Qt.MouseButton.LeftButton:
            p = self.map_widget_to_image(ev.position().toPoint())
            if p:
                self.drawing_finished.emit(p[0], p[1])
            super().mouseReleaseEvent(ev)
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
            "points_per_side": int((params_defaults or {}).get("points_per_side", 16)),
            "pred_iou_thresh": float((params_defaults or {}).get("pred_iou_thresh", 0.88)),
        }
        self.cache: Dict[Path, Tuple[np.ndarray, List[np.ndarray], List[float]]] = {}
        self.selected_indices: set[int] = set()
        self._hover_idx: Optional[int] = None
        
        # 標註系統
        self.annotations: Dict[int, int] = {}  # {mask_index: class_id}
        self.annotation_history: List[Dict] = []  # 歷史記錄
        self.annotation_redo_stack: List[Dict] = []  # Redo 堆疊
        self.max_history = 20  # 最多保留20步
        self._list_hover_idx: Optional[int] = None  # 列表懸浮的索引
        
        # 每張影像的標註狀態（獨立存儲）
        self.per_image_state: Dict[Path, Dict] = {}  # {image_path: {selected_indices, annotations}}
        
        # 多色彩系統 - 使用 HSV 動態生成無限顏色
        # 不再使用固定字典，改用函數生成

        # image view
        self.view = ImageView(self)
        self.view.viewport().installEventFilter(self)  # hover/點選 hit test

        # 右側群組 UI
        # ========== 1. 檢視與導航 ==========
        grp_view_nav = QGroupBox("檢視與導航")
        
        # 顯示模式
        self.rb_show_mask = QRadioButton("遮罩")
        self.rb_show_mask.setToolTip("顯示語意分割遮罩 (Mask)")
        self.rb_show_bbox = QRadioButton("外框")
        self.rb_show_bbox.setToolTip("顯示物件外接矩形 (Bounding Box)")
        self.rb_show_mask.setChecked(True)

        self.display_group = QButtonGroup(self)
        self.display_group.addButton(self.rb_show_mask, 0)  # 0=遮罩
        self.display_group.addButton(self.rb_show_bbox, 1)  # 1=BBox
        
        # 導航按鈕
        self.btn_prev = QPushButton("◀ 上一張")
        self.btn_prev.setToolTip("切換至上一張影像 (PageUp)")
        self.btn_next = QPushButton("下一張 ▶")
        self.btn_next.setToolTip("切換至下一張影像 (PageDown)")
        self.btn_reset_view = QPushButton("🔄 重設視圖")
        self.btn_reset_view.setToolTip("重設影像縮放與位置")
        
        # 佈局
        lay_view_nav = QVBoxLayout()
        lay_view_nav.addWidget(QLabel("顯示模式:"))
        display_layout = QHBoxLayout()
        display_layout.addWidget(self.rb_show_mask)
        display_layout.addWidget(self.rb_show_bbox)
        lay_view_nav.addLayout(display_layout)
        
        lay_view_nav.addWidget(QLabel("影像切換:"))
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        lay_view_nav.addLayout(nav_layout)
        lay_view_nav.addWidget(self.btn_reset_view)
        
        # 顯示所有候選遮罩
        self.chk_show_candidates = QCheckBox("顯示所有候選遮罩")
        self.chk_show_candidates.setToolTip("以低透明度顯示所有 SAM 生成的遮罩")
        self.chk_show_candidates.stateChanged.connect(lambda: self._update_canvas())
        lay_view_nav.addWidget(self.chk_show_candidates)
        
        grp_view_nav.setLayout(lay_view_nav)
        
        # 切換顯示模式即時重繪
        self.display_group.idClicked.connect(lambda _id: self._update_canvas())

        # ========== 2. 輸出設定 ==========
        grp_output_config = QGroupBox("輸出設定")
        
        # 裁切模式
        self.rb_full = QRadioButton("完整影像")
        self.rb_full.setToolTip("輸出整張原始圖片尺寸")
        self.rb_bbox = QRadioButton("僅物件區域")
        self.rb_bbox.setToolTip("僅輸出包含物件的最小矩形範圍")
        self.rb_bbox.setChecked(True)
        self.crop_group = QButtonGroup(self)
        self.crop_group.addButton(self.rb_full, 0)
        self.crop_group.addButton(self.rb_bbox, 1)
        
        # 輸出模式
        self.rb_mode_indiv = QRadioButton("個別物件")
        self.rb_mode_indiv.setToolTip("每個選取物件分別存為獨立圖檔")
        self.rb_mode_union = QRadioButton("合併物件")
        self.rb_mode_union.setToolTip("將所有選取物件合併為單一圖檔")
        self.rb_mode_indiv.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_mode_indiv, 0)
        self.mode_group.addButton(self.rb_mode_union, 1)
        
        # 輸出模式切換時也要重繪（為了 BBox 聯集時只畫一個框）
        self.mode_group.idClicked.connect(self._on_mode_changed)
        
        # 輸出格式
        format_label = QLabel("檔案格式:")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG", "BMP"])
        self.format_combo.setCurrentIndex(0)  # 預設 PNG
        self.format_combo.setToolTip("選擇輸出影像的檔案格式")
        
        # 輸出路徑
        output_path_label = QLabel("輸出路徑:")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("預設為原影像同層資料夾")
        self.output_path_edit.setText("")  # 空白表示使用預設
        self.output_path_edit.setToolTip("設定檔案輸出的目標資料夾")
        btn_browse_output = QPushButton("瀏覽...")
        btn_browse_output.clicked.connect(self._browse_output_path)
        
        # 佈局
        lay_output_config = QVBoxLayout()
        
        lay_output_config.addWidget(QLabel("裁切模式:"))
        crop_layout = QHBoxLayout()
        crop_layout.addWidget(self.rb_bbox)
        crop_layout.addWidget(self.rb_full)
        lay_output_config.addLayout(crop_layout)
        
        lay_output_config.addWidget(QLabel("存檔方式:"))
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.rb_mode_indiv)
        mode_layout.addWidget(self.rb_mode_union)
        lay_output_config.addLayout(mode_layout)
        
        format_layout = QHBoxLayout()
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo, 1)
        lay_output_config.addLayout(format_layout)
        
        lay_output_config.addWidget(output_path_label)
        output_path_layout = QHBoxLayout()
        output_path_layout.addWidget(self.output_path_edit, 1)
        output_path_layout.addWidget(btn_browse_output)
        lay_output_config.addLayout(output_path_layout)
        
        grp_output_config.setLayout(lay_output_config)

        # ========== 3. 標註格式 ==========
        grp_labels = QGroupBox("標註格式")
        
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

        # 佈局：2x2 網格
        lay_labels = QGridLayout()
        lay_labels.addWidget(self.chk_yolo_det, 0, 0)
        lay_labels.addWidget(self.chk_yolo_seg, 0, 1)
        lay_labels.addWidget(self.chk_coco, 1, 0)
        lay_labels.addWidget(self.chk_voc, 1, 1)
        grp_labels.setLayout(lay_labels)

        # 顏色設定（初始化，UI 移至菜單）
        self.mask_color = [0, 255, 0]  # 預設綠色 (BGR)
        self.bbox_color = [0, 255, 0]  # 預設綠色 (BGR)
        self.mask_alpha = 0.4          # 預設遮罩透明度

        # ========== 4. 手動修飾工具 ==========
        grp_manual_tools = QGroupBox("手動修飾")
        
        # 工具按鈕（僅 icon）
        self.btn_tool_cursor = QPushButton("👆")
        self.btn_tool_cursor.setCheckable(True)
        self.btn_tool_cursor.setChecked(True)
        self.btn_tool_cursor.setToolTip("選取模式：點選物件進行選取")
        self.btn_tool_cursor.setFixedSize(50, 50)
        
        self.btn_tool_brush = QPushButton("🖌️")
        self.btn_tool_brush.setCheckable(True)
        self.btn_tool_brush.setToolTip("畫筆模式：手動增加遮罩區域")
        self.btn_tool_brush.setFixedSize(50, 50)
        
        self.btn_tool_eraser = QPushButton("🧽")
        self.btn_tool_eraser.setCheckable(True)
        self.btn_tool_eraser.setToolTip("橡皮擦模式：手動擦除遮罩區域")
        self.btn_tool_eraser.setFixedSize(50, 50)
        
        self.btn_tool_magic = QPushButton("🧹")
        self.btn_tool_magic.setCheckable(True)
        self.btn_tool_magic.setToolTip("魔法掃把：點選區域自動清除相似顏色範圍")
        self.btn_tool_magic.setFixedSize(50, 50)
        
        # 工具群組（互斥）
        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_tool_cursor, 0)
        self.tool_group.addButton(self.btn_tool_brush, 1)
        self.tool_group.addButton(self.btn_tool_eraser, 2)
        self.tool_group.addButton(self.btn_tool_magic, 3)
        # 連接工具切換信號以更新游標
        self.tool_group.idClicked.connect(self._on_tool_changed)
        
        # 筆刷大小滑桿
        self.lbl_brush_size = QLabel("筆刷大小: 10px")
        self.slider_brush_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush_size.setRange(1, 50)
        self.slider_brush_size.setValue(10)
        self.slider_brush_size.setToolTip("調整畫筆與橡皮擦的大小")
        self.slider_brush_size.valueChanged.connect(lambda v: self.lbl_brush_size.setText(f"筆刷大小: {v}px"))
        
        # 佈局：工具按鈕排成一列
        lay_manual = QVBoxLayout()
        tools_layout = QHBoxLayout()
        tools_layout.addWidget(self.btn_tool_cursor)
        tools_layout.addWidget(self.btn_tool_brush)
        tools_layout.addWidget(self.btn_tool_eraser)
        tools_layout.addWidget(self.btn_tool_magic)
        lay_manual.addLayout(tools_layout)
        
        lay_manual.addWidget(self.lbl_brush_size)
        lay_manual.addWidget(self.slider_brush_size)
        
        # Undo/Redo 按鈕
        undo_redo_layout = QHBoxLayout()
        self.btn_undo = QPushButton("↶ 復原")
        self.btn_undo.setToolTip("撤銷上一步操作 (Ctrl+Z)")
        self.btn_undo.setEnabled(False)
        self.btn_redo = QPushButton("↷ 重做")
        self.btn_redo.setToolTip("重做已撤銷的操作 (Ctrl+Y)")
        self.btn_redo.setEnabled(False)
        undo_redo_layout.addWidget(self.btn_undo)
        undo_redo_layout.addWidget(self.btn_redo)
        lay_manual.addLayout(undo_redo_layout)
        
        grp_manual_tools.setLayout(lay_manual)

        # ========== 5. 儲存操作 ==========
        grp_save_actions = QGroupBox("儲存操作")
        
        self.btn_save_selected = QPushButton("💾 儲存選取物件")
        self.btn_save_selected.setToolTip("僅儲存目前已選取的物件")
        self.btn_save_all = QPushButton("💾 儲存全部物件")
        self.btn_save_all.setToolTip("自動儲存影像中偵測到的所有物件")
        self.lbl_selected = QLabel("已選物件：0")
        self.lbl_selected.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lay_save_actions = QVBoxLayout()
        lay_save_actions.addWidget(self.btn_save_selected)
        lay_save_actions.addWidget(self.btn_save_all)
        lay_save_actions.addWidget(self.lbl_selected)
        grp_save_actions.setLayout(lay_save_actions)

        # 參數設定（移至菜單，但保留變數）

        # ========== 左側物件列表面板（使用表格） ==========
        grp_objects = QGroupBox("")
        # 與控制面板保持一致的邊距 (Left, Top, Right, Bottom)
        # 控制面板通常有預設邊距，這裡我們設定一個合理的邊距來對齊
        grp_objects.setContentsMargins(5, 5, 5, 5) 
        objects_layout = QVBoxLayout()
        
        # 使用 QTableWidget 替代 QListWidget
        self.object_table = QTableWidget()
        self.object_table.setColumnCount(4)
        self.object_table.setHorizontalHeaderLabels(["色塊", "物件", "類別", "操作"])
        self.object_table.setToolTip("滑鼠懸浮可高亮顯示對應物件")
        self.object_table.setMouseTracking(True)
        self.object_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.object_table.setSelectionMode(QTableWidget.SingleSelection)
        self.object_table.verticalHeader().setVisible(False)
        
        # 設定欄位寬度
        self.object_table.setColumnWidth(0, 40)   # 色塊
        self.object_table.setColumnWidth(1, 80)   # 物件
        self.object_table.setColumnWidth(2, 60)   # 類別
        self.object_table.setColumnWidth(3, 50)   # 操作
        
        # 連接懸浮事件
        self.object_table.cellEntered.connect(self._on_table_cell_hover)
        
        objects_layout.addWidget(self.object_table)
        grp_objects.setLayout(objects_layout)
        
        # 保留舊的 object_list 變數以避免錯誤（設為 None）
        self.object_list = None

        # ========== 組裝左側面板（物件列表） ==========
        left_widget = QWidget()
        left_box = QVBoxLayout()
        left_box.addWidget(grp_objects)
        left_box.setContentsMargins(0, 0, 0, 0)
        left_widget.setLayout(left_box)
        
        # 建立左側 Dock
        self.dock_objects = QDockWidget("標註物件", self)
        self.dock_objects.setWidget(left_widget)
        self.dock_objects.setFeatures(
            QDockWidget.DockWidgetMovable | 
            QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_objects)
        self.dock_objects.show()  # 確保 dock 可見
        
        # ========== 組裝右側面板 ==========
        right_box = QVBoxLayout()
        right_box.addWidget(grp_view_nav)        # 1. 檢視與導航
        right_box.addWidget(grp_manual_tools)    # 2. 手動修飾 (新增)
        right_box.addWidget(grp_output_config)   # 3. 輸出設定
        right_box.addWidget(grp_labels)          # 4. 標註格式
        right_box.addWidget(grp_save_actions)    # 5. 儲存操作
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
        self.dock_controls.show()  # 確保 dock 可見

        # 設定中央widget為影像檢視
        self.setCentralWidget(self.view)
        
        # 建立菜單欄
        self._create_menu_bar()

        # connect
        self.btn_reset_view.clicked.connect(self._reset_view_and_selections)
        self.btn_prev.clicked.connect(self._prev_image)
        
        # 連接繪圖信號
        self.view.drawing_started.connect(self._on_drawing_started)
        self.view.drawing_moved.connect(self._on_drawing_moved)
        self.view.drawing_finished.connect(self._on_drawing_finished)
        self.btn_next.clicked.connect(self._next_image)
        self.btn_save_selected.clicked.connect(self._save_selected)
        self.btn_save_all.clicked.connect(self._save_all)
        self.btn_undo.clicked.connect(self._undo_annotation)
        self.btn_redo.clicked.connect(self._redo_annotation)

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
    
    def _generate_class_color(self, class_id: int) -> list:
        """使用 HSV 色彩空間動態生成類別顏色（BGR 格式）"""
        import colorsys
        
        # 使用黃金比例來分散色相，確保顏色差異明顯
        golden_ratio = 0.618033988749895
        hue = (class_id * golden_ratio) % 1.0
        
        # 固定飽和度和明度以獲得鮮豔的顏色
        saturation = 0.9
        value = 0.95
        
        # 轉換 HSV 到 RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # 轉換到 0-255 範圍並返回 BGR 格式（OpenCV 使用 BGR）
        return [int(b * 255), int(g * 255), int(r * 255)]
    
    def _get_class_color(self, class_id: int) -> list:
        """獲取類別顏色（BGR 格式）"""
        return self._generate_class_color(class_id)
    
    def _get_mask_color(self, mask_idx: int) -> list:
        """根據 mask 的 class 取得顏色（BGR 格式）"""
        class_id = self.annotations.get(mask_idx, 0)
        return self._get_class_color(class_id)
    
    def _create_menu_bar(self):
        """建立菜單欄"""
        menubar = self.menuBar()
        
        # 選項菜單
        options_menu = menubar.addMenu("選項")
        
        # 1. 分割參數 (最重要)
        params_action = QAction("分割參數設定...", self)
        params_action.triggered.connect(self._show_params_dialog)
        options_menu.addAction(params_action)
        
        options_menu.addSeparator()

        # 2. 顯示設定 (顏色、透明度)
        # 遮罩透明度
        alpha_action = QAction("遮罩透明度...", self)
        alpha_action.triggered.connect(self._change_mask_alpha)
        options_menu.addAction(alpha_action)
        
        # 顏色設定 (保留但重要性降低，因為現在是自動顏色)
        color_action = QAction("自訂顏色 (僅用於單色模式)...", self)
        color_action.triggered.connect(self._show_color_dialog)
        # options_menu.addAction(color_action) # 暫時隱藏，因為現在是多色模式

        options_menu.addSeparator()

        # 3. 快捷鍵
        act_shortcuts = QAction("快捷鍵列表...", self)
        act_shortcuts.triggered.connect(self._show_shortcuts_dialog)
        options_menu.addAction(act_shortcuts)
        
        # 檢視選單
        view_menu = menubar.addMenu("檢視")
        
        act_light = QAction("淺色主題", self)
        act_light.triggered.connect(lambda: self._apply_theme("light"))
        
        act_dark = QAction("深色主題", self)
        act_dark.triggered.connect(lambda: self._apply_theme("dark"))
        
        view_menu.addAction(act_light)
        view_menu.addAction(act_dark)
        
        # 說明選單
        help_menu = menubar.addMenu("說明")
        
        act_help = QAction("使用說明", self)
        act_help.triggered.connect(self._show_help)
        
        help_menu.addAction(act_help)
        
        # 關於選單
        about_menu = menubar.addMenu("關於")
        
        act_about = QAction("關於本專案...", self)
        act_about.triggered.connect(self._show_about)
        
        about_menu.addAction(act_about)

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
                shortcut_reset.activated.connect(self._reset_view_and_selections)
            
            # 復原標註 (Undo)
            undo_key = shortcut_manager.get_shortcut('edit.undo')
            if undo_key:
                shortcut_undo = QShortcut(QKeySequence(undo_key), self)
                shortcut_undo.activated.connect(self._undo_annotation)
                
        except Exception as e:
            logger.warning(f"載入快捷鍵失敗: {e}")

    def _start_batch_processing(self):
        if not self.image_paths:
            return

        # 先載入並顯示第一張影像，讓視窗有內容
        self._load_current_image(recompute=False)
        
        # 記錄批次處理開始時間
        import time
        self._batch_start_time = time.time()
        
        # 然後啟動批次處理（會跳過已有快取的影像）
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
            
            # 計算預估剩餘時間
            if hasattr(self, '_batch_start_time') and current > 0:
                import time
                elapsed = time.time() - self._batch_start_time
                avg_time = elapsed / current
                remaining = avg_time * (total - current)
                
                # 格式化時間
                if remaining < 60:
                    time_str = f"{int(remaining)}秒"
                elif remaining < 3600:
                    mins = int(remaining / 60)
                    secs = int(remaining % 60)
                    time_str = f"{mins}分{secs}秒"
                else:
                    hours = int(remaining / 3600)
                    mins = int((remaining % 3600) / 60)
                    time_str = f"{hours}小時{mins}分"
                
                # 提取影像名稱
                if current <= len(self.image_paths):
                    img_name = self.image_paths[current - 1].name if current > 0 else ""
                    self.batch_progress.set_message(
                        f"({current}/{total}) {img_name} - 預估剩餘 {time_str}"
                    )
                else:
                    self.batch_progress.set_message(f"({current}/{total}) {msg}")
            else:
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
        # 載入此影像的標註狀態
        self._load_image_state(path)
        
        self._hover_idx = None
        self._update_selected_count()
        self._update_object_list()  # 更新物件列表
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
    
    def _reset_view_and_selections(self) -> None:
        """重設視圖並清除所有選取"""
        # 重設視圖縮放與位置
        self.view.reset_view()
        
        # 清除所有選取
        if self.selected_indices:
            self.selected_indices.clear()
            self.annotations.clear()
            self._hover_idx = None
            
            # 更新 UI
            self._update_selected_count()
            self._update_object_list()
            self._update_canvas()
            
            self.status.message_temp("已重設視圖並清除所有選取", 1500)
        else:
            self.status.message_temp("已重設視圖", 1000)
    
    def _save_current_image_state(self) -> None:
        """保存當前影像的標註狀態"""
        if not self.image_paths or self.idx >= len(self.image_paths):
            return
        
        current_path = self.image_paths[self.idx]
        self.per_image_state[current_path] = {
            'selected_indices': self.selected_indices.copy(),
            'annotations': self.annotations.copy()
        }
    
    def _load_image_state(self, path: Path) -> None:
        """載入指定影像的標註狀態，如果不存在則清空"""
        if path in self.per_image_state:
            # 恢復保存的狀態
            state = self.per_image_state[path]
            self.selected_indices = state['selected_indices'].copy()
            self.annotations = state['annotations'].copy()
        else:
            # 清空狀態（新影像或尚未標註）
            self.selected_indices.clear()
            self.annotations.clear()
        
        # 清空歷史記錄（每張影像獨立）
        self.annotation_history.clear()
    
    def _prev_image(self) -> None:
        """Navigate to the previous image."""
        if self.idx > 0:
            self._save_current_image_state()
            self.idx -= 1
            self._load_current_image(recompute=False)

    def _next_image(self) -> None:
        """Navigate to the next image."""
        if self.idx < len(self.image_paths) - 1:
            self._save_current_image_state()
            self.idx += 1
            self._load_current_image(recompute=False)


    def _create_emoji_cursor(self, emoji: str, size: int = 32) -> QCursor:
        """Create a cursor from an emoji."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        # Try to use a font that supports emojis well on Windows
        font = QFont("Segoe UI Emoji", int(size * 0.8)) 
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        painter.setFont(font)
        
        # Center the emoji
        rect = QRectF(0, 0, size, size)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, emoji)
        painter.end()
        
        # Create cursor with hotspot at center
        return QCursor(pixmap, size // 2, size // 2)

    def _on_tool_changed(self, tool_id: int):
        """Handle tool change events to update cursor."""
        viewport = self.view.viewport()
        
        # 工具名稱映射
        tool_names = {
            0: "👆 選取",
            1: "🖌️ 畫筆",
            2: "🧽 橡皮擦",
            3: "🧹 魔法掃把"
        }
        
        if tool_id == 0: # Cursor
            viewport.setCursor(Qt.CursorShape.ArrowCursor)
        elif tool_id == 1: # Brush
            viewport.setCursor(self._create_emoji_cursor("🖌️"))
        elif tool_id == 2: # Eraser
            viewport.setCursor(self._create_emoji_cursor("🧽"))
        elif tool_id == 3: # Magic Broom
            viewport.setCursor(self._create_emoji_cursor("🧹"))
        
        # 更新狀態欄顯示當前工具
        if hasattr(self, 'status') and tool_id in tool_names:
            self.status.set_tool_mode(tool_names[tool_id])

    def _update_cursor_visual(self, pos: QPoint) -> None:
        """Update the visual cursor position and size."""
        if not hasattr(self, 'cursor_item'):
            return
            
        tool_id = self.tool_group.checkedId()
        # 1=Brush, 2=Eraser
        if tool_id in [1, 2]:
            # Map widget pos to scene pos (image coordinates)
            scene_pos = self.view.mapToScene(pos)
            
            size = self.slider_brush_size.value()
            
            # Update cursor item
            # Center the ellipse on the mouse position
            self.cursor_item.setRect(
                scene_pos.x() - size / 2,
                scene_pos.y() - size / 2,
                size,
                size
            )
            
            # Update cursor color based on selected object
            if tool_id == 1 and self.selected_indices: # Brush
                # Use the color of the last selected object
                last_idx = sorted(list(self.selected_indices))[-1]
                color = self._get_mask_color(last_idx)
                # BGR to RGB for QColor
                qcolor = QColor(color[2], color[1], color[0])
                self.cursor_item.setPen(QPen(qcolor, 1, Qt.PenStyle.SolidLine))
            else:
                # Eraser or no selection: White
                self.cursor_item.setPen(QPen(QColor(255, 255, 255), 1, Qt.PenStyle.SolidLine))
                
            self.cursor_item.show()
        else:
            self.cursor_item.hide()

    # ---- drawing events ----
    def _on_drawing_started(self, pos: QPoint) -> None:
        self._last_draw_pos = self._map_widget_to_image(pos)
        self._last_brush_pos = self._last_draw_pos # For smoothing
        
        # 儲存狀態以供 Undo
        self._save_annotation_state()
        
        # 立即應用第一點
        if self._last_draw_pos:
            self._apply_brush_stroke(self._last_draw_pos)

    def _on_drawing_moved(self, pos: QPoint) -> None:
        # Update visual cursor
        self._update_cursor_visual(pos)
        
        current_pos = self._map_widget_to_image(pos)
        if current_pos is None:
            return
            
        # Smooth drawing using Bresenham's line algorithm
        if self._last_brush_pos:
            x0, y0 = self._last_brush_pos
            x1, y1 = current_pos
            points = self._get_line_points(x0, y0, x1, y1)
            for p in points:
                self._apply_brush_stroke(p)
        else:
            self._apply_brush_stroke(current_pos)
            
        self._last_brush_pos = current_pos

    def _on_drawing_finished(self, pos: QPoint) -> None:
        self._last_draw_pos = None
        self._last_brush_pos = None
        # Final canvas update
        self._update_canvas()

    def _get_line_points(self, x0: int, y0: int, x1: int, y1: int) -> list:
        """Bresenham's line algorithm to get all points between two pixels."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x1, y1))
        return points

    def _apply_brush_stroke(self, pos: Tuple[int, int]) -> None:
        tool_id = self.tool_group.checkedId()
        if tool_id not in [1, 2, 3]: # Brush, Eraser, Magic
            return
            
        path = self.image_paths[self.idx]
        if path not in self.cache:
            return
            
        bgr, masks, scores = self.cache[path]
        x, y = pos
        brush_size = self.slider_brush_size.value()
        radius = brush_size // 2
        
        H, W = bgr.shape[:2]
        
        # 1. Magic Broom
        if tool_id == 3:
            # ... (Magic broom logic implementation if needed, skipping for now as user focused on brush/eraser)
            # For now, let's just implement basic brush/eraser
            pass
            
        # 2. Brush / Eraser
        else:
            # 如果沒有選取任何物件，且是畫筆模式，則創建一個新物件
            if not self.selected_indices and tool_id == 1:
                new_mask = np.zeros((H, W), dtype=np.uint8)
                masks.append(new_mask)
                scores.append(1.0) # Dummy score
                new_idx = len(masks) - 1
                self.selected_indices.add(new_idx)
                self.annotations[new_idx] = 0 # Default class
                self._update_object_list()
                self._update_selected_count()
            
            # 對所有選取的 mask 進行操作
            for idx in self.selected_indices:
                if 0 <= idx < len(masks):
                    mask = masks[idx]
                    if tool_id == 1: # Brush
                        cv2.circle(mask, (x, y), radius, 1, -1)
                    elif tool_id == 2: # Eraser
                        cv2.circle(mask, (x, y), radius, 0, -1)
        
        # Update canvas (partial update could be optimized, but full update is safer)
        self._update_canvas()


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
        
        # 顯示所有候選遮罩 (低透明度)
        if getattr(self, "chk_show_candidates", None) and self.chk_show_candidates.isChecked():
            # 建立一個全黑的遮罩層
            candidates_overlay = np.zeros_like(base)
            # 建立一個 alpha 通道層，用於處理重疊
            alpha_map = np.zeros(base.shape[:2], dtype=np.float32)
            
            for i, m in enumerate(masks):
                # 跳過已選取的 (避免重複繪製)
                if i in self.selected_indices:
                    continue
                
                # 取得該遮罩的區域
                mask_bool = m > 0
                
                # 生成唯一顏色
                color = np.array(self._generate_class_color(i), dtype=np.uint8)
                
                # 在 overlay 上繪製顏色
                # 對於重疊區域，這裡採用"最後繪製優先"的策略
                # 這符合"以交集的為主"的一種解釋（顯示最上層的遮罩）
                candidates_overlay[mask_bool] = color
                
                # 標記有遮罩的區域
                alpha_map[mask_bool] = 0.3  # 設定候選遮罩的透明度
            
            # 混合到底圖
            # 只有在有候選遮罩的地方才進行混合
            mask_indices = alpha_map > 0
            
            # 向量化混合計算
            # base = base * (1 - alpha) + overlay * alpha
            alpha_3d = alpha_map[mask_indices][:, None]
            base[mask_indices] = (base[mask_indices] * (1 - alpha_3d) + candidates_overlay[mask_indices] * alpha_3d).astype(np.uint8)

        # 顯示模式: 0=遮罩, 1=BBox
        disp_id = self.display_group.checkedId() if hasattr(self, "display_group") else 0
        use_bbox = disp_id == 1

        # 輸出模式: 0=個別, 1=聯集
        mode_id = self.mode_group.checkedId() if hasattr(self, "mode_group") else 0
        is_union = mode_id == 1

        # 決定統一顏色 (用於聯集模式)
        union_color_bgr = None
        if is_union and self.selected_indices:
            # 使用第一個選取物件的顏色作為統一顏色
            first_idx = sorted(list(self.selected_indices))[0]
            union_color_bgr = np.array(self._get_mask_color(first_idx), dtype=np.uint8)

        if not use_bbox:
            # 遮罩高亮模式 - 使用多色彩系統
            if self.selected_indices:
                # 為每個選取的物件繪製顏色
                for i in self.selected_indices:
                    if 0 <= i < len(masks):
                        m = masks[i] > 0
                        # 決定顏色: 聯集模式用統一顏色，否則用個別顏色
                        if is_union and union_color_bgr is not None:
                            color_bgr = union_color_bgr
                        else:
                            color_bgr = np.array(self._get_mask_color(i), dtype=np.uint8)
                            
                        # 使用 self.mask_alpha
                        alpha = self.mask_alpha
                        base[m] = (base[m] * (1 - alpha) + color_bgr * alpha).astype(np.uint8)

            # 懸浮高亮（來自滑鼠或列表）
            hover_idx = self._list_hover_idx if self._list_hover_idx is not None else self._hover_idx
            if hover_idx is not None and 0 <= hover_idx < len(masks):
                hover_mask = masks[hover_idx]
                # 確保 mask 維度正確
                if hover_mask.shape[:2] == base.shape[:2]:
                    m = hover_mask > 0
                    color_bgr = np.array(self._get_mask_color(hover_idx), dtype=np.uint8)
                    # 懸浮時稍微不透明一點
                    alpha = min(1.0, self.mask_alpha + 0.2)
                    base[m] = (base[m] * (1 - alpha) + color_bgr * alpha).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        # 使用該物件的類別顏色繪製輪廓
                        bbox_color_tuple = tuple(int(c) for c in color_bgr.tolist())
                        cv2.polylines(base, contours, True, bbox_color_tuple, 2)

        else:
            # BBox 模式
            H, W = base.shape[:2]
            
            if is_union and self.selected_indices:
                # 聯集 + BBox: 只畫一個大框線
                union_mask = np.zeros((H, W), dtype=np.uint8)
                for i in self.selected_indices:
                    if 0 <= i < len(masks):
                        union_mask = np.maximum(union_mask, masks[i])
                
                x, y, w, h = compute_bbox(union_mask > 0)
                
                # 使用統一顏色
                if union_color_bgr is not None:
                    bbox_color_tuple = tuple(int(c) for c in union_color_bgr.tolist())
                else:
                    bbox_color_tuple = (0, 255, 0) # Fallback green
                    
                cv2.rectangle(base, (x, y), (x + w, y + h), bbox_color_tuple, 3)
            else:
                # 個別 + BBox: 已選畫細線
                for i in self.selected_indices:
                    if 0 <= i < len(masks):
                        x, y, w, h = compute_bbox(masks[i] > 0)
                        # 使用該物件的類別顏色
                        color_bgr = self._get_mask_color(i)
                        bbox_color_tuple = tuple(int(c) for c in color_bgr)
                        cv2.rectangle(base, (x, y), (x + w, y + h), bbox_color_tuple, 2)
                        
                # 懸浮畫粗線
                if self._hover_idx is not None and 0 <= self._hover_idx < len(masks):
                    x, y, w, h = compute_bbox(masks[self._hover_idx] > 0)
                    # 使用該物件的類別顏色
                    color_bgr = self._get_mask_color(self._hover_idx)
                    bbox_color_tuple = tuple(int(c) for c in color_bgr)
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
            self._write_yolo_labels(out_dir, base_name, boxes, polys, img_w, img_h, indices)
            self._write_coco_json(out_dir, base_name, boxes, polys, img_w, img_h, indices)
            self._write_voc_xml(out_dir, base_name, boxes, img_w, img_h, save_path.name, indices)
            self._write_labelme_json(out_dir, base_name, polys, img_w, img_h, save_path.name, indices)
            
            QMessageBox.information(self, "完成", f"已儲存聯集影像至：\n{save_path}")
            self.status.message("儲存完成")
        else:
            QMessageBox.warning(self, "失敗", "影像編碼失敗")

    def _save_indices(self, indices: List[int]) -> None:
        """Save selected masks as individual images and export combined annotations."""
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

        # 收集原始影像座標的標註資料，用於輸出單一標註檔
        all_boxes = []
        all_polys = []
        valid_indices = []

        for i in indices:
            if not (0 <= i < len(masks)):
                continue
            m = masks[i] > 0
            
            # 收集原始座標資料
            x_orig, y_orig, w_orig, h_orig = compute_bbox(m)
            poly_orig = self._compute_polygon(m)
            
            all_boxes.append((x_orig, y_orig, w_orig, h_orig))
            all_polys.append(poly_orig)
            valid_indices.append(i)
            
            # 準備輸出影像 (BGRA)
            bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = m.astype(np.uint8) * 255
            
            base_name = f"{path.stem}_{i:03d}"
            
            if self.rb_bbox.isChecked():
                # 裁切模式：儲存裁切後的影像
                crop = bgra[y_orig : y_orig + h_orig, x_orig : x_orig + w_orig]
            else:
                # 原圖模式：儲存整張影像（背景透明）
                crop = bgra
            
            if fmt in ["jpg", "bmp"]:
                save_img = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
            else:
                save_img = crop
                
            save_path = out_dir / f"{base_name}{ext}"
            ok, buf = cv2.imencode(ext, save_img)
            if ok:
                save_path.write_bytes(buf.tobytes())
                saved_count += 1
        
        if saved_count > 0:
            # 輸出單一標註檔案 (對應原始影像)
            # 使用原始影像檔名 (不帶 _000 後綴)
            base_name_orig = path.stem
            
            # 寫出各種標註格式 (使用原始影像尺寸和座標)
            self._write_yolo_labels(out_dir, base_name_orig, all_boxes, all_polys, W, H, valid_indices)
            self._write_coco_json(out_dir, base_name_orig, all_boxes, all_polys, W, H, valid_indices)
            self._write_voc_xml(out_dir, base_name_orig, all_boxes, W, H, path.name, valid_indices)
            self._write_labelme_json(out_dir, base_name_orig, all_polys, W, H, path.name, valid_indices)
            
            QMessageBox.information(self, "完成", f"已儲存 {saved_count} 個物件影像及標註檔案")
            self.status.message(f"已儲存 {saved_count} 個物件")
        else:
            QMessageBox.warning(self, "提示", "沒有儲存任何檔案")

    def _write_coco_json(self, out_dir, base_name, boxes, polys, img_w, img_h, indices):
        """Export to COCO JSON format."""
        if not getattr(self, "chk_coco", None) or not self.chk_coco.isChecked():
            return
            
        coco_data = {
            "images": [
                {"id": 1, "width": img_w, "height": img_h, "file_name": f"{base_name}.png"}
            ],
            "annotations": [],
            "categories": []
        }
        
        # 建立 Categories
        used_classes = set()
        for idx in indices:
            cls_id = self.annotations.get(idx, 0)
            used_classes.add(cls_id)
            
        for cls_id in sorted(used_classes):
            coco_data["categories"].append({
                "id": cls_id,
                "name": f"class_{cls_id}",
                "supercategory": "object"
            })
        
        for i, (box, poly) in enumerate(zip(boxes, polys)):
            # 取得對應的 index 和 class
            obj_idx = indices[i] if i < len(indices) else 0
            cls_id = self.annotations.get(obj_idx, 0)
            
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

    def _write_voc_xml(self, out_dir, base_name, boxes, w, h, filename, indices):
        """Export to Pascal VOC XML format."""
        if not getattr(self, "chk_voc", None) or not self.chk_voc.isChecked():
            return
            
        import xml.etree.ElementTree as ET
        
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = out_dir.name
        ET.SubElement(root, "filename").text = filename
        ET.SubElement(root, "path").text = filename  # 使用相對路徑 (僅檔名)
        
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = "3"
        
        for i, (x, y, bw, bh) in enumerate(boxes):
            # 取得對應的 index 和 class
            obj_idx = indices[i] if i < len(indices) else 0
            cls_id = self.annotations.get(obj_idx, 0)
            cls_name = f"class_{cls_id}"
            
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

    def _change_mask_alpha(self):
        """Change mask transparency."""
        from PySide6.QtWidgets import QInputDialog
        
        current_alpha = int(self.mask_alpha * 100)
        val, ok = QInputDialog.getInt(
            self, 
            "遮罩透明度", 
            "請輸入透明度 (0-100，數值越小越透明):", 
            current_alpha, 
            0, 100, 1
        )
        if ok:
            self.mask_alpha = val / 100.0
            self._update_canvas()

    def _write_labelme_json(self, out_dir, base_name, polys, w, h, filename, indices):
        """Export to LabelMe JSON format."""
        if not getattr(self, "chk_labelme", None) or not self.chk_labelme.isChecked():
            return
            
        shapes = []
        for i, poly in enumerate(polys):
            if poly is not None and len(poly) > 0:
                # 取得對應的 index 和 class
                obj_idx = indices[i] if i < len(indices) else 0
                cls_id = self.annotations.get(obj_idx, 0)
                cls_name = f"class_{cls_id}"
                
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
        indices: List[int],  # 新增：物件索引列表
    ) -> None:
        """依勾選輸出 YOLO 檢測與/或 YOLO 分割標註檔。使用每個物件的 class ID。"""

        # YOLO 檢測: 每行 => cls xc yc w h (皆為 0~1)
        if getattr(self, "chk_yolo_det", None) and self.chk_yolo_det.isChecked():
            lines = []
            for idx, (x, y, w, h) in enumerate(boxes):
                if w <= 0 or h <= 0:
                    continue
                # 使用對應物件的 class ID
                obj_idx = indices[idx] if idx < len(indices) else 0
                cls_id = self.annotations.get(obj_idx, 0)
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
            for idx, poly in enumerate(polys):
                if poly is None or len(poly) == 0:
                    continue
                # 使用對應物件的 class ID
                obj_idx = indices[idx] if idx < len(indices) else 0
                cls_id = self.annotations.get(obj_idx, 0)
                pts = []
                for px, py in poly:
                    pts.append(f"{px / img_w:.6f} {py / img_h:.6f}")
                lines.append(f"{cls_id} " + " ".join(pts))
            if lines:
                (out_dir / f"{base_name}_seg.txt").write_text("\n".join(lines), encoding="utf-8")

    # ===== 新增：標註系統方法 =====
    
    def _save_annotation_state(self) -> None:
        """儲存當前標註狀態到歷史記錄"""
        state = {
            'selected_indices': self.selected_indices.copy(),
            'annotations': self.annotations.copy()
        }
        self.annotation_history.append(state)
        # 限制歷史記錄數量
        if len(self.annotation_history) > self.max_history:
            self.annotation_history.pop(0)
        # 清空 redo stack（新操作會使 redo 失效）
        self.annotation_redo_stack.clear()
        self._update_undo_redo_buttons()
    
    def _undo_annotation(self) -> None:
        """復原上一步標註"""
        if not self.annotation_history:
            self.status.message_temp("無可復原的操作", 1000)
            return
        
        # 儲存當前狀態到 redo stack
        current_state = {
            'selected_indices': self.selected_indices.copy(),
            'annotations': self.annotations.copy()
        }
        self.annotation_redo_stack.append(current_state)
        
        # 恢復上一個狀態
        state = self.annotation_history.pop()
        self.selected_indices = state['selected_indices']
        self.annotations = state['annotations']
        
        # 更新UI
        self._update_canvas()
        self._update_selected_count()
        self._update_object_list()
        self._update_undo_redo_buttons()
        self.status.message_temp("已復原", 1000)
    
    def _redo_annotation(self) -> None:
        """重做已撤銷的操作"""
        if not self.annotation_redo_stack:
            self.status.message_temp("無可重做的操作", 1000)
            return
        
        # 儲存當前狀態到歷史
        current_state = {
            'selected_indices': self.selected_indices.copy(),
            'annotations': self.annotations.copy()
        }
        self.annotation_history.append(current_state)
        
        # 恢復 redo 狀態
        state = self.annotation_redo_stack.pop()
        self.selected_indices = state['selected_indices']
        self.annotations = state['annotations']
        
        # 更新UI
        self._update_canvas()
        self._update_selected_count()
        self._update_object_list()
        self._update_undo_redo_buttons()
        self.status.message_temp("已重做", 1000)
    
    def _update_undo_redo_buttons(self) -> None:
        """更新 Undo/Redo 按鈕的啟用狀態"""
        if hasattr(self, 'btn_undo'):
            self.btn_undo.setEnabled(len(self.annotation_history) > 0)
        if hasattr(self, 'btn_redo'):
            self.btn_redo.setEnabled(len(self.annotation_redo_stack) > 0)
    
    def _on_mode_changed(self, mode_id: int) -> None:
        """處理輸出模式切換"""
        is_union = mode_id == 1
        
        if is_union:
            # 切換到 Union 模式：將所有選取物件設為預設類別 0
            for mask_idx in self.selected_indices:
                self.annotations[mask_idx] = 0
        
        # 更新物件列表（會根據模式禁用/啟用類別選擇器）
        self._update_object_list()
        # 更新畫布（使用統一顏色）
        self._update_canvas()
    
    def _update_object_list(self) -> None:
        """更新物件列表顯示（使用表格，支援無限類別）"""
        # 清空表格
        self.object_table.setRowCount(0)
        
        # 檢查是否為 Union 模式
        mode_id = self.mode_group.checkedId() if hasattr(self, "mode_group") else 0
        is_union = mode_id == 1
        
        for row_idx, mask_idx in enumerate(sorted(self.selected_indices)):
            class_id = self.annotations.get(mask_idx, 0)
            
            # 插入新行
            self.object_table.insertRow(row_idx)
            
            # 欄位 0: 色塊（使用 QLabel 顯示顏色）
            color_bgr = self._get_class_color(class_id)
            color_hex = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"  # BGR to Hex
            color_widget = QWidget()
            color_layout = QHBoxLayout(color_widget)
            color_layout.setContentsMargins(5, 2, 5, 2)
            color_label = QLabel("  ")
            color_label.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #666; border-radius: 3px;")
            color_label.setFixedSize(24, 24)
            color_layout.addWidget(color_label)
            color_layout.addStretch()
            self.object_table.setCellWidget(row_idx, 0, color_widget)
            
            # 欄位 1: 物件編號
            obj_item = QTableWidgetItem(f"#{mask_idx}")
            obj_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            obj_item.setData(Qt.ItemDataRole.UserRole, mask_idx)  # 儲存 mask_idx
            obj_item.setFlags(obj_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # 不可編輯
            self.object_table.setItem(row_idx, 1, obj_item)
            
            # 欄位 2: 類別 ID（使用 SpinBox）
            spin = QSpinBox()
            spin.setRange(0, 9999)  # 支援無限類別
            spin.setValue(class_id)
            spin.setToolTip("修改類別 ID" if not is_union else "Union 模式下類別固定為 0")
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Union 模式下禁用類別編輯
            spin.setEnabled(not is_union)
            
            # 連接信號，使用 lambda 捕捉當前的 mask_idx
            spin.valueChanged.connect(lambda val, idx=mask_idx, r=row_idx: self._on_table_class_changed(idx, val, r))
            self.object_table.setCellWidget(row_idx, 2, spin)
            
            # 欄位 3: 刪除按鈕
            btn_delete = QPushButton("×")
            btn_delete.setToolTip("從選取中移除")
            btn_delete.setFixedSize(30, 24)
            btn_delete.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; }")
            btn_delete.clicked.connect(lambda checked, idx=mask_idx: self._on_delete_object(idx))
            delete_widget = QWidget()
            delete_layout = QHBoxLayout(delete_widget)
            delete_layout.setContentsMargins(2, 0, 2, 0)
            delete_layout.addWidget(btn_delete)
            self.object_table.setCellWidget(row_idx, 3, delete_widget)
            
            # 設定行高
            self.object_table.setRowHeight(row_idx, 32)

    def _on_table_class_changed(self, mask_idx: int, new_class_id: int, row_idx: int) -> None:
        """當使用者在表格中修改 Class ID 時"""
        if mask_idx in self.selected_indices:
            # 更新 annotations
            self.annotations[mask_idx] = new_class_id
            
            # 更新畫布
            self._update_canvas()
            
            # 更新該行的色塊顏色
            color_bgr = self._get_class_color(new_class_id)
            color_hex = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"
            
            # 獲取色塊 widget 並更新顏色
            color_widget = self.object_table.cellWidget(row_idx, 0)
            if color_widget:
                color_label = color_widget.findChild(QLabel)
                if color_label:
                    color_label.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #666; border-radius: 3px;")
    
    def _on_delete_object(self, mask_idx: int) -> None:
        """從選取中移除物件"""
        if mask_idx in self.selected_indices:
            # 儲存歷史狀態
            self._save_annotation_state()
            
            # 移除選取
            self.selected_indices.remove(mask_idx)
            if mask_idx in self.annotations:
                del self.annotations[mask_idx]
            
            # 更新 UI
            self._update_selected_count()
            self._update_object_list()
            self._update_canvas()
    
    def _on_table_cell_hover(self, row: int, column: int) -> None:
        """當滑鼠懸浮在表格儲存格上時"""
        if row >= 0:
            # 獲取該行的 mask_idx
            item = self.object_table.item(row, 1)
            if item:
                mask_idx = item.data(Qt.ItemDataRole.UserRole)
                self._list_hover_idx = mask_idx
        else:
            self._list_hover_idx = None
        self._update_canvas()
    
    def _on_list_item_hover(self, item: QListWidgetItem) -> None:
        """當滑鼠懸浮在列表項目上時（舊方法，保留以避免錯誤）"""
        # 此方法已不再使用，因為改用表格
        pass
    
    def _save_annotations_json(self, image_path: Path, out_dir: Path) -> None:
        """Save current annotations (selected indices and classes) to a JSON file."""
        try:
            # 使用新格式：包含 class 資訊
            annotations = []
            for idx in sorted(self.selected_indices):
                class_id = self.annotations.get(idx, 0)
                annotations.append({
                    "index": idx,
                    "class_id": class_id
                })
            
            data = {
                "image_path": image_path.name,
                "annotations": annotations
            }
            
            # 儲存到與輸出影像相同的目錄，檔名為 [原始檔名]_annotations.json
            save_path = out_dir / f"{image_path.stem}_annotations.json"
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"已儲存標註狀態: {save_path} ({len(annotations)} 個物件)")
        except Exception as e:
            logger.error(f"儲存標註狀態失敗: {e}")

    def _load_annotations(self, image_path: Path) -> None:
        """載入影像的標註資料"""
        # 嘗試從同目錄載入 annotations.json
        json_path = image_path.parent / f"{image_path.stem}_annotations.json"
        
        if not json_path.exists():
            # 沒有標註檔案，清空狀態
            self.selected_indices.clear()
            self.annotations.clear()
            self.annotation_history.clear()
            return
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 載入標註資料
            if 'annotations' in data:
                # 新格式：包含 class 資訊
                self.selected_indices.clear()
                self.annotations.clear()
                for ann in data['annotations']:
                    idx = ann['index']
                    class_id = ann.get('class_id', 0)
                    self.selected_indices.add(idx)
                    self.annotations[idx] = class_id
            elif 'selected_indices' in data:
                # 舊格式：只有索引列表
                self.selected_indices = set(data['selected_indices'])
                self.annotations = {idx: 0 for idx in self.selected_indices}
            
            # 清空歷史記錄
            self.annotation_history.clear()
            
            logger.info(f"已載入標註: {len(self.selected_indices)} 個物件")
            
        except Exception as e:
            logger.error(f"載入標註失敗: {e}")
            self.selected_indices.clear()
            self.annotations.clear()

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
                    # Update visual cursor
                    self._update_cursor_visual(event.position().toPoint() if hasattr(event, "position") else event.pos())

                    # 在繪圖模式下不處理 hover
                    tool_id = self.tool_group.checkedId()
                    if tool_id != 0:  # 非選取模式
                        if hasattr(self, 'status'):
                            pos = _pt(event)
                            img_xy = self._map_widget_to_image(pos)
                            if img_xy:
                                self.status.set_cursor_xy(img_xy[0], img_xy[1])
                            else:
                                self.status.set_cursor_xy(None, None)
                        return False
                    
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
                    # 在繪圖模式下不處理點擊選取
                    tool_id = self.tool_group.checkedId()
                    if tool_id != 0:  # 非選取模式
                        return False
                    
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
                        # 儲存歷史狀態
                        self._save_annotation_state()
                        # 加入選取
                        self.selected_indices.add(tgt)
                        # 如果還沒有 class，設為預設 class 0
                        if tgt not in self.annotations:
                            self.annotations[tgt] = 0
                        # 更新 UI
                        self._update_selected_count()
                        self._update_object_list()
                        self._update_canvas()
                    elif event.button() == Qt.MouseButton.RightButton:
                        if tgt in self.selected_indices:
                            # 儲存歷史狀態
                            self._save_annotation_state()
                            # 移除選取
                            self.selected_indices.remove(tgt)
                            if tgt in self.annotations:
                                del self.annotations[tgt]
                            # 更新 UI
                            self._update_selected_count()
                            self._update_object_list()
                            self._update_canvas()
                    return False
            except Exception:
                logger.warning("滑鼠事件處理發生例外", exc_info=True)
                return False
        return super().eventFilter(obj, event)

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
    
    # ===== 手動修飾工具方法 =====
    
    def _on_drawing_started(self, x: int, y: int):
        """處理繪圖開始事件"""
        tool_id = self.tool_group.checkedId()
        
        # 0: Cursor (不處理，交給原本的點擊邏輯)
        if tool_id == 0:
            return
            
        # 檢查是否有選取物件
        if not self.selected_indices:
            self.status.message_temp("請先選取一個物件進行修飾", 2000)
            return
            
        # 3: Magic Broom (點擊觸發)
        if tool_id == 3:
            self._apply_magic_broom(x, y)
            return
            
        # 1: Brush, 2: Eraser (開始筆觸)
        self._is_drawing = True
        self._apply_brush_stroke(x, y, tool_id)
    
    def _on_drawing_moved(self, x: int, y: int):
        """處理繪圖移動事件"""
        if not getattr(self, "_is_drawing", False):
            return
            
        tool_id = self.tool_group.checkedId()
        if tool_id in [1, 2]:  # Brush or Eraser
            self._apply_brush_stroke(x, y, tool_id)
    
    def _on_drawing_finished(self, x: int, y: int):
        """處理繪圖結束事件"""
        if getattr(self, "_is_drawing", False):
            self._is_drawing = False
            self._last_brush_pos = None  # 清除上一個位置
            # 可以在這裡儲存歷史記錄
            # self._save_annotation_state()
    
    def _apply_brush_stroke(self, x: int, y: int, tool_id: int):
        """應用畫筆或橡皮擦筆觸"""
        if not self.image_paths or self.idx >= len(self.image_paths):
            return
            
        path = self.image_paths[self.idx]
        if path not in self.cache:
            return
            
        _, masks, _ = self.cache[path]
        
        # 針對所有選取的 mask 進行修改
        brush_size = self.slider_brush_size.value()
        radius = brush_size // 2
        
        # 1: Brush (Add), 2: Eraser (Remove)
        value = 1 if tool_id == 1 else 0
        
        # 改善平滑度：如果有上一個位置，繪製線段上的所有點
        changed = False
        if hasattr(self, '_last_brush_pos') and self._last_brush_pos:
            x0, y0 = self._last_brush_pos
            # 使用 Bresenham 線段算法獲取線段上的所有點
            points = self._get_line_points(x0, y0, x, y)
        else:
            points = [(x, y)]
        
        # 儲存當前位置
        self._last_brush_pos = (x, y)
        
        for px, py in points:
            for idx in self.selected_indices:
                if 0 <= idx < len(masks):
                    mask = masks[idx]
                    
                    # 確保 mask 是 uint8 且連續的，以便 OpenCV 繪圖
                    if mask.dtype == bool:
                        mask = mask.astype(np.uint8)
                        masks[idx] = mask
                    
                    if not mask.flags['C_CONTIGUOUS']:
                        mask = np.ascontiguousarray(mask)
                        masks[idx] = mask
                    
                    # 使用 OpenCV 繪製圓形來修改 mask
                    cv2.circle(mask, (px, py), radius, value, -1)
                    changed = True
        
        if changed:
            self._update_canvas()
    
    def _get_line_points(self, x0: int, y0: int, x1: int, y1: int) -> list:
        """使用 Bresenham 算法獲取線段上的所有點"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
            
    def _apply_magic_broom(self, x: int, y: int):
        """應用魔法掃把 (Region Growing 清除)"""
        if not self.image_paths or self.idx >= len(self.image_paths):
            return
            
        path = self.image_paths[self.idx]
        if path not in self.cache:
            return
            
        bgr, masks, _ = self.cache[path]
        H, W = bgr.shape[:2]
        
        if not (0 <= x < W and 0 <= y < H):
            return
            
        # 1. 找出連通區域 (Flood Fill)
        # 建立 mask for floodFill (H+2, W+2)
        flood_mask = np.zeros((H + 2, W + 2), np.uint8)
        
        # 容許度
        loDiff = (20, 20, 20)
        upDiff = (20, 20, 20)
        
        # 執行 floodFill，結果會標記在 flood_mask 中
        # flags: 4-connectivity + (255 << 8) to fill with 255 + FLOODFILL_MASK_ONLY
        flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
        
        cv2.floodFill(bgr, flood_mask, (x, y), (0, 0, 0), loDiff, upDiff, flags)
        
        # 取出實際大小的 mask (去除邊框)
        region_mask = flood_mask[1:-1, 1:-1]
        
        # 2. 從選取的 mask 中移除該區域
        changed = False
        count_removed = 0
        
        for idx in self.selected_indices:
            if 0 <= idx < len(masks):
                mask = masks[idx]
                # 計算重疊區域
                overlap = (mask > 0) & (region_mask > 0)
                if np.any(overlap):
                    # 移除重疊區域
                    mask[overlap] = 0
                    changed = True
                    count_removed += np.sum(overlap)
        
        if changed:
            self.status.message_temp(f"魔法掃把已清除 {count_removed} 像素", 2000)
            self._update_canvas()
        else:
            self.status.message_temp("點選區域不在選取範圍內", 1000)

    # ===== 選單處理方法 =====
    
    def _apply_theme(self, theme_name: str):
        """套用主題"""
        from modules.presentation.qt.theme_manager import apply_theme
        apply_theme(self, theme_name)
        self.status.message_temp(f"已切換至{theme_name}主題", 1000)
    
    def _show_shortcuts_dialog(self):
        """顯示快捷鍵設定對話框"""
        from modules.presentation.qt.shortcut_dialog import ShortcutEditorDialog
        dialog = ShortcutEditorDialog(self)
        dialog.exec()
    
    def _show_help(self):
        """顯示使用說明"""
        help_text = """
        <h2>影像標註工具使用說明</h2>
        <p><b>基本操作：</b></p>
        <ul>
            <li><b>左鍵點擊：</b> 選擇分割區域 (加入選取)</li>
            <li><b>右鍵點擊：</b> 取消選擇分割區域 (移除選取)</li>
            <li><b>滾輪：</b> 縮放影像</li>
            <li><b>中鍵拖曳：</b> 移動影像</li>
        </ul>
        <p><b>快捷鍵：</b></p>
        <ul>
            <li><b>A：</b> 切換到上一張影像</li>
            <li><b>D：</b> 切換到下一張影像</li>
            <li><b>Ctrl + S：</b> 儲存目前已選取的目標</li>
            <li><b>Ctrl + Z：</b> 復原上一步標註</li>
            <li><b>R：</b> 重設檢視</li>
        </ul>
        <p><b>功能說明：</b></p>
        <ul>
            <li><b>輸出裁切模式：</b> 選擇輸出僅包含物件的最小矩形或整張原圖。</li>
            <li><b>輸出模式：</b>
                <ul>
                    <li><b>個別獨立：</b> 每個選取的物件存成單獨的檔案。</li>
                    <li><b>疊加聯集：</b> 所有選取的物件合併成單一檔案。</li>
                </ul>
            </li>
            <li><b>輸出標註格式：</b> 支援 YOLO, COCO, VOC, LabelMe 等多種格式。</li>
            <li><b>標註物件列表：</b> 顯示已標註的物件，滑鼠懸浮可高亮顯示。</li>
        </ul>
        <hr>
        <p><i>Created by Coffee ☕</i></p>
        """
        QMessageBox.about(self, "使用說明", help_text)
    
    def _show_about(self):
        """顯示關於對話框"""
        about_text = """
        <h2>影像標註工具 v1.0.0</h2>
        <p><b>作者：</b>Coffee ☕</p>
        
        <h3>專案資訊</h3>
        <p>本專案為基於 Segment Anything Model (SAM) 的影像標註工具，<br>
        提供直覺的介面讓使用者快速標註影像中的物件。</p>
        
        <h3>授權與使用限制</h3>
        <p><b>本專案僅供學術研究與個人學習使用。</b><br>
        未經授權，請勿用於商業用途。</p>
        
        <h3>使用的開源套件</h3>
        <ul>
            <li><b>PySide6 (Qt for Python)</b><br>
                授權：LGPL v3 / Commercial License<br>
                說明：PySide6 採用 LGPL v3 授權，允許在遵守 LGPL 條款下用於商業專案。<br>
                若需要閉源商業使用，可購買 Qt 商業授權。</li>
            <li><b>Segment Anything Model (SAM)</b><br>
                授權：Apache License 2.0<br>
                說明：Meta AI 開發的模型，允許商業使用。</li>
            <li><b>OpenCV</b><br>
                授權：Apache License 2.0<br>
                說明：開源電腦視覺函式庫，允許商業使用。</li>
            <li><b>PyTorch</b><br>
                授權：BSD-3-Clause License<br>
                說明：開源深度學習框架，允許商業使用。</li>
        </ul>
        
        <h3>商業使用說明</h3>
        <p>雖然本專案使用的主要套件（PySide6、SAM、OpenCV、PyTorch）<br>
        在遵守各自授權條款下允許商業使用，但<b>本專案程式碼本身</b><br>
        未經作者授權不得用於商業用途。</p>
        
        <p>如需商業授權，請聯繫作者。</p>
        
        <hr>
        <p style="font-size: 11px; color: #666;">© 2025 Coffee. All rights reserved.</p>
        """
        QMessageBox.about(self, "關於", about_text)
