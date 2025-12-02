from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QRadioButton, QButtonGroup, 
    QPushButton, QLabel, QHBoxLayout, QCheckBox, QComboBox, 
    QLineEdit, QFileDialog, QGridLayout, QSlider
)

class ControlPanelWidget(QWidget):
    """Right-side control panel containing navigation, output settings, and tools."""
    
    # Signals
    display_mode_changed = Signal()
    nav_prev_clicked = Signal()
    nav_next_clicked = Signal()
    reset_view_clicked = Signal()
    show_candidates_toggled = Signal(bool)
    output_mode_changed = Signal(int)
    tool_changed = Signal(int)
    brush_size_changed = Signal(int)
    undo_clicked = Signal()
    redo_clicked = Signal()
    save_selected_clicked = Signal()
    save_all_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        
        # 1. View & Nav
        layout.addWidget(self._create_view_nav_group())
        
        # 2. Manual Tools
        layout.addWidget(self._create_manual_tools_group())
        
        # 3. Output Config
        layout.addWidget(self._create_output_config_group())
        
        # 4. Label Format
        layout.addWidget(self._create_label_format_group())
        
        # 5. Save Actions
        layout.addWidget(self._create_save_actions_group())
        
        layout.addStretch(1)
        self.setLayout(layout)

    def _create_view_nav_group(self) -> QGroupBox:
        grp = QGroupBox("檢視與導航")
        
        self.rb_show_mask = QRadioButton("遮罩")
        self.rb_show_bbox = QRadioButton("外框")
        self.rb_show_mask.setChecked(True)
        
        self.display_group = QButtonGroup(self)
        self.display_group.addButton(self.rb_show_mask, 0)
        self.display_group.addButton(self.rb_show_bbox, 1)
        self.display_group.idClicked.connect(lambda _: self.display_mode_changed.emit())
        
        self.btn_prev = QPushButton("◀ 上一張")
        self.btn_next = QPushButton("下一張 ▶")
        self.btn_reset_view = QPushButton("🔄 重設視圖")
        
        self.btn_prev.clicked.connect(self.nav_prev_clicked)
        self.btn_next.clicked.connect(self.nav_next_clicked)
        self.btn_reset_view.clicked.connect(self.reset_view_clicked)
        
        self.chk_show_candidates = QCheckBox("顯示所有候選遮罩")
        self.chk_show_candidates.stateChanged.connect(lambda state: self.show_candidates_toggled.emit(state == Qt.CheckState.Checked.value))
        
        lay = QVBoxLayout()
        
        d_lay = QHBoxLayout()
        d_lay.addWidget(self.rb_show_mask)
        d_lay.addWidget(self.rb_show_bbox)
        lay.addLayout(d_lay)
        
        n_lay = QHBoxLayout()
        n_lay.addWidget(self.btn_prev)
        n_lay.addWidget(self.btn_next)
        lay.addLayout(n_lay)
        lay.addWidget(self.btn_reset_view)
        lay.addWidget(self.chk_show_candidates)
        
        grp.setLayout(lay)
        return grp

    def _create_output_config_group(self) -> QGroupBox:
        grp = QGroupBox("輸出設定")
        
        self.rb_full = QRadioButton("完整影像")
        self.rb_bbox = QRadioButton("僅物件區域")
        self.rb_bbox.setChecked(True)
        self.crop_group = QButtonGroup(self)
        self.crop_group.addButton(self.rb_full, 0)
        self.crop_group.addButton(self.rb_bbox, 1)
        
        self.rb_mode_indiv = QRadioButton("個別物件")
        self.rb_mode_union = QRadioButton("合併物件")
        self.rb_mode_indiv.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_mode_indiv, 0)
        self.mode_group.addButton(self.rb_mode_union, 1)
        self.mode_group.idClicked.connect(self.output_mode_changed)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG", "BMP"])
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("預設為原影像同層資料夾")
        btn_browse = QPushButton("瀏覽...")
        btn_browse.clicked.connect(self._browse_path)
        
        lay = QVBoxLayout()
        
        c_lay = QHBoxLayout()
        c_lay.addWidget(self.rb_bbox)
        c_lay.addWidget(self.rb_full)
        lay.addLayout(c_lay)
        
        m_lay = QHBoxLayout()
        m_lay.addWidget(self.rb_mode_indiv)
        m_lay.addWidget(self.rb_mode_union)
        lay.addLayout(m_lay)
        
        f_lay = QHBoxLayout()
        f_lay.addWidget(QLabel("格式:"))
        f_lay.addWidget(self.format_combo)
        lay.addLayout(f_lay)
        
        p_lay = QHBoxLayout()
        p_lay.addWidget(self.output_path_edit)
        p_lay.addWidget(btn_browse)
        lay.addLayout(p_lay)
        
        grp.setLayout(lay)
        return grp

    def _create_label_format_group(self) -> QGroupBox:
        grp = QGroupBox("標註格式")
        
        self.chk_yolo_det = QCheckBox("YOLO (偵測)")
        self.chk_yolo_seg = QCheckBox("YOLO (分割)")
        self.chk_coco = QCheckBox("COCO")
        self.chk_voc = QCheckBox("VOC")
        self.chk_labelme = QCheckBox("LabelMe") # Added missing one
        
        lay = QGridLayout()
        lay.addWidget(self.chk_yolo_det, 0, 0)
        lay.addWidget(self.chk_yolo_seg, 0, 1)
        lay.addWidget(self.chk_coco, 1, 0)
        lay.addWidget(self.chk_voc, 1, 1)
        lay.addWidget(self.chk_labelme, 2, 0)
        
        grp.setLayout(lay)
        return grp

    def _create_manual_tools_group(self) -> QGroupBox:
        grp = QGroupBox("手動修飾")
        
        self.btn_tool_cursor = QPushButton("👆")
        self.btn_tool_cursor.setCheckable(True)
        self.btn_tool_cursor.setChecked(True)
        self.btn_tool_cursor.setFixedSize(40, 40)
        
        self.btn_tool_brush = QPushButton("🖌️")
        self.btn_tool_brush.setCheckable(True)
        self.btn_tool_brush.setFixedSize(40, 40)
        
        self.btn_tool_eraser = QPushButton("🧽")
        self.btn_tool_eraser.setCheckable(True)
        self.btn_tool_eraser.setFixedSize(40, 40)
        
        self.btn_tool_magic = QPushButton("🧹")
        self.btn_tool_magic.setCheckable(True)
        self.btn_tool_magic.setFixedSize(40, 40)
        
        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.btn_tool_cursor, 0)
        self.tool_group.addButton(self.btn_tool_brush, 1)
        self.tool_group.addButton(self.btn_tool_eraser, 2)
        self.tool_group.addButton(self.btn_tool_magic, 3)
        self.tool_group.idClicked.connect(self.tool_changed)
        
        self.slider_brush = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush.setRange(1, 50)
        self.slider_brush.setValue(10)
        self.lbl_brush = QLabel("筆刷: 10px")
        self.slider_brush.valueChanged.connect(lambda v: self.lbl_brush.setText(f"筆刷: {v}px"))
        self.slider_brush.valueChanged.connect(self.brush_size_changed)
        
        self.btn_undo = QPushButton("↶ 復原")
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo_clicked)
        
        self.btn_redo = QPushButton("↷ 重做")
        self.btn_redo.setEnabled(False)
        self.btn_redo.clicked.connect(self.redo_clicked)
        
        lay = QVBoxLayout()
        t_lay = QHBoxLayout()
        t_lay.addWidget(self.btn_tool_cursor)
        t_lay.addWidget(self.btn_tool_brush)
        t_lay.addWidget(self.btn_tool_eraser)
        t_lay.addWidget(self.btn_tool_magic)
        lay.addLayout(t_lay)
        
        lay.addWidget(self.lbl_brush)
        lay.addWidget(self.slider_brush)
        
        ur_lay = QHBoxLayout()
        ur_lay.addWidget(self.btn_undo)
        ur_lay.addWidget(self.btn_redo)
        lay.addLayout(ur_lay)
        
        grp.setLayout(lay)
        return grp

    def _create_save_actions_group(self) -> QGroupBox:
        grp = QGroupBox("儲存操作")
        
        self.btn_save_selected = QPushButton("💾 儲存選取物件")
        self.btn_save_all = QPushButton("💾 儲存全部物件")
        self.lbl_selected_count = QLabel("已選物件：0")
        self.lbl_selected_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_save_selected.clicked.connect(self.save_selected_clicked)
        self.btn_save_all.clicked.connect(self.save_all_clicked)
        
        lay = QVBoxLayout()
        lay.addWidget(self.btn_save_selected)
        lay.addWidget(self.btn_save_all)
        lay.addWidget(self.lbl_selected_count)
        
        grp.setLayout(lay)
        return grp

    def _browse_path(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾")
        if folder:
            self.output_path_edit.setText(folder)

    def update_nav_buttons(self, has_prev: bool, has_next: bool):
        self.btn_prev.setEnabled(has_prev)
        self.btn_next.setEnabled(has_next)

    def update_selected_count(self, count: int):
        self.lbl_selected_count.setText(f"已選物件：{count}")

    def update_undo_redo(self, can_undo: bool, can_redo: bool):
        self.btn_undo.setEnabled(can_undo)
        self.btn_redo.setEnabled(can_redo)
