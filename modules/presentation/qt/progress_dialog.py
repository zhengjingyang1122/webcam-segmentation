import sys
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QWidget
from PySide6.QtCore import Qt

class ThemedProgressDialog(QDialog):
    """A themed progress dialog."""
    
    def __init__(self, title: str, message: str, parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(350)
        
        # Remove context help button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        self.lbl_message = QLabel(message)
        self.lbl_message.setWordWrap(True)
        layout.addWidget(self.lbl_message)
        
        self.pbar = QProgressBar()
        self.pbar.setRange(0, 0)  # Indeterminate by default
        self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(6)
        layout.addWidget(self.pbar)
        
        from PySide6.QtWidgets import QPushButton
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        layout.addWidget(self.btn_cancel, 0, Qt.AlignRight)
        
        self.setLayout(layout)
        
    def set_message(self, message: str):
        self.lbl_message.setText(message)
        QApplication.processEvents()
        
    def set_range(self, min_val: int, max_val: int):
        self.pbar.setRange(min_val, max_val)
        
    def set_value(self, value: int):
        self.pbar.setValue(value)
        QApplication.processEvents()

# Helper to get QApplication instance for processEvents
from PySide6.QtWidgets import QApplication
