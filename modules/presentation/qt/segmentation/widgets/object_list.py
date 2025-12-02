from __future__ import annotations
from typing import Dict, Set, Optional, Callable
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QWidget, QHBoxLayout, QLabel, QSpinBox, QPushButton
)

class ObjectListWidget(QGroupBox):
    """Widget for displaying the list of selected objects."""
    
    # Signals
    hover_changed = Signal(int)  # mask_idx (or -1 if none)
    class_changed = Signal(int, int)  # mask_idx, new_class_id
    delete_requested = Signal(int)  # mask_idx

    def __init__(self, title: str = "", parent=None):
        super().__init__(title, parent)
        self.setContentsMargins(5, 5, 5, 5)
        
        layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["色塊", "物件", "類別", "操作"])
        self.table.setToolTip("滑鼠懸浮可高亮顯示對應物件")
        self.table.setMouseTracking(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        
        # Column widths
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 60)
        self.table.setColumnWidth(3, 50)
        
        self.table.cellEntered.connect(self._on_cell_hover)
        
        layout.addWidget(self.table)
        self.setLayout(layout)
        
        self._color_getter: Optional[Callable[[int], list]] = None

    def set_color_getter(self, func: Callable[[int], list]):
        self._color_getter = func

    def update_list(self, selected_indices: Set[int], annotations: Dict[int, int], is_union_mode: bool):
        """Update the table with current selection."""
        self.table.setRowCount(0)
        
        for row_idx, mask_idx in enumerate(sorted(selected_indices)):
            class_id = annotations.get(mask_idx, 0)
            
            self.table.insertRow(row_idx)
            
            # Col 0: Color
            if self._color_getter:
                color_bgr = self._color_getter(class_id)
                color_hex = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"
            else:
                color_hex = "#00ff00"
                
            color_widget = QWidget()
            color_layout = QHBoxLayout(color_widget)
            color_layout.setContentsMargins(5, 2, 5, 2)
            color_label = QLabel("  ")
            color_label.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #666; border-radius: 3px;")
            color_label.setFixedSize(24, 24)
            color_layout.addWidget(color_label)
            color_layout.addStretch()
            self.table.setCellWidget(row_idx, 0, color_widget)
            
            # Col 1: ID
            obj_item = QTableWidgetItem(f"#{mask_idx}")
            obj_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            obj_item.setData(Qt.ItemDataRole.UserRole, mask_idx)
            obj_item.setFlags(obj_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row_idx, 1, obj_item)
            
            # Col 2: Class ID
            spin = QSpinBox()
            spin.setRange(0, 9999)
            spin.setValue(class_id)
            spin.setToolTip("修改類別 ID" if not is_union_mode else "Union 模式下類別固定為 0")
            spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
            spin.setEnabled(not is_union_mode)
            
            # Connect signal
            spin.valueChanged.connect(lambda val, idx=mask_idx: self.class_changed.emit(idx, val))
            self.table.setCellWidget(row_idx, 2, spin)
            
            # Col 3: Delete
            btn_delete = QPushButton("×")
            btn_delete.setToolTip("從選取中移除")
            btn_delete.setFixedSize(30, 24)
            btn_delete.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; }")
            btn_delete.clicked.connect(lambda checked, idx=mask_idx: self.delete_requested.emit(idx))
            
            delete_widget = QWidget()
            delete_layout = QHBoxLayout(delete_widget)
            delete_layout.setContentsMargins(2, 0, 2, 0)
            delete_layout.addWidget(btn_delete)
            self.table.setCellWidget(row_idx, 3, delete_widget)
            
            self.table.setRowHeight(row_idx, 32)

    def _on_cell_hover(self, row: int, column: int):
        if row >= 0:
            item = self.table.item(row, 1)
            if item:
                mask_idx = item.data(Qt.ItemDataRole.UserRole)
                self.hover_changed.emit(mask_idx)
        else:
            self.hover_changed.emit(-1)
