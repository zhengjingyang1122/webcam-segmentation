# modules/presentation/qt/shortcut_dialog.py
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QMessageBox, QHeaderView, QKeySequenceEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from modules.presentation.qt.shortcut_manager import ShortcutManager


class ShortcutEditorDialog(QDialog):
    """快捷鍵編輯對話框"""
    
    # 動作名稱對應的中文描述
    ACTION_NAMES = {
        'nav.prev': '上一張影像',
        'nav.next': '下一張影像',
        'save.selected': '儲存選取物件',
        'view.reset': '重設檢視'
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("快捷鍵設定")
        self.setModal(True)
        self.resize(500, 400)
        
        self.shortcut_manager = ShortcutManager()
        self.modified = False
        
        self._build_ui()
        self._load_shortcuts()
    
    def _build_ui(self):
        """建立UI"""
        layout = QVBoxLayout()
        
        # 說明文字
        info_label = QLabel("點擊快捷鍵欄位以修改按鍵組合")
        info_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(info_label)
        
        # 快捷鍵表格
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["功能", "快捷鍵"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)
        
        # 按鈕
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.btn_reset = QPushButton("重設為預設值")
        self.btn_reset.clicked.connect(self._reset_defaults)
        button_layout.addWidget(self.btn_reset)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        self.btn_save = QPushButton("儲存")
        self.btn_save.clicked.connect(self._save_shortcuts)
        button_layout.addWidget(self.btn_save)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _load_shortcuts(self):
        """載入快捷鍵到表格"""
        shortcuts = self.shortcut_manager.get_all_shortcuts()
        self.table.setRowCount(len(shortcuts))
        
        for row, (action, key) in enumerate(shortcuts.items()):
            # 功能名稱
            name_item = QTableWidgetItem(self.ACTION_NAMES.get(action, action))
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, name_item)
            
            # 快捷鍵
            key_item = QTableWidgetItem(key)
            key_item.setData(Qt.ItemDataRole.UserRole, action)  # 儲存 action
            self.table.setItem(row, 1, key_item)
        
        self.table.itemChanged.connect(self._on_item_changed)
    
    def _on_item_changed(self, item):
        """當表格項目改變時"""
        if item.column() == 1:
            self.modified = True
    
    def _reset_defaults(self):
        """重設為預設值"""
        reply = QMessageBox.question(
            self, "確認重設",
            "確定要將所有快捷鍵重設為預設值嗎？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            defaults = {
                'nav.prev': 'PageUp',
                'nav.next': 'PageDown',
                'save.selected': 'Ctrl+S',
                'view.reset': 'R'
            }
            
            for row in range(self.table.rowCount()):
                key_item = self.table.item(row, 1)
                action = key_item.data(Qt.ItemDataRole.UserRole)
                if action in defaults:
                    key_item.setText(defaults[action])
            
            self.modified = True
    
    def _save_shortcuts(self):
        """儲存快捷鍵"""
        # 更新 ShortcutManager
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 1)
            action = key_item.data(Qt.ItemDataRole.UserRole)
            new_key = key_item.text().strip()
            
            if new_key:
                self.shortcut_manager.set_shortcut(action, new_key)
        
        # 儲存到檔案
        self.shortcut_manager.save_shortcuts()
        
        QMessageBox.information(
            self, "儲存成功",
            "快捷鍵設定已儲存。\n重新啟動應用程式後生效。"
        )
        
        self.accept()
