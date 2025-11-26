import sys
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QComboBox, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QGroupBox
from PySide6.QtGui import QAction
from modules.infrastructure.vision.sam_engine import SamEngine
from modules.presentation.qt.segmentation.segmentation_viewer import SegmentationViewer
from modules.presentation.qt.theme_manager import apply_theme
from utils.get_base_path import get_base_path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SegmentationLauncher(QMainWindow):
    """Launcher window for segmentation tool."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Segmentation Tool")
        self.resize(400, 200)
        
        self.sam = None
        self.viewer = None
        self.current_theme = "dark"  # Track current theme
        
        # Model selection mapping
        self.model_files = {
            "SAM-B (Fast)": ("sam_vit_b_01ec64.pth", "vit_b"),
            "SAM-L (Balanced)": ("sam_vit_l_0b3195.pth", "vit_l"),
            "SAM-H (Best Quality)": ("sam_vit_h_4b8939.pth", "vit_h"),
        }
        
        # Create menu bar
        self._create_menus()
        
        # Apply default dark theme
        apply_theme(self, "dark")
        
        # Build UI
        self._build_ui()
    
    def _build_ui(self):
        """Build the launcher UI."""
        from PySide6.QtCore import Qt
        
        central = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Welcome label
        label = QLabel("歡迎使用影像分割工具\n\n請選擇模型並從「檔案」選單選擇要分割的影像")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 14px;")
        layout.addWidget(label)
        
        # Model selection group
        model_group = QGroupBox("模型選擇")
        model_layout = QHBoxLayout()
        
        model_label = QLabel("SAM 模型:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.model_files.keys()))
        self.model_combo.setCurrentIndex(2)  # Default to SAM-H
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 1)
        model_group.setLayout(model_layout)
        
        layout.addWidget(model_group)
        layout.addStretch()
        
        central.setLayout(layout)
        self.setCentralWidget(central)
    
    def _create_menus(self):
        """Create menu bar with File and View menus."""
        # File menu
        file_menu = self.menuBar().addMenu("檔案")
        
        act_open_image = QAction("開啟影像...", self)
        act_open_image.triggered.connect(self._open_image)
        
        act_open_folder = QAction("開啟資料夾...", self)
        act_open_folder.triggered.connect(self._open_folder)
        
        act_exit = QAction("結束", self)
        act_exit.triggered.connect(self.close)
        
        file_menu.addAction(act_open_image)
        file_menu.addAction(act_open_folder)
        file_menu.addSeparator()
        file_menu.addAction(act_exit)
        
        # View menu (theme selection)
        view_menu = self.menuBar().addMenu("檢視")
        
        act_light = QAction("淺色主題", self)
        act_light.triggered.connect(lambda: self._apply_theme("light"))
        
        act_dark = QAction("深色主題", self)
        act_dark.triggered.connect(lambda: self._apply_theme("dark"))
        
        view_menu.addAction(act_light)
        view_menu.addAction(act_dark)
    
    def _apply_theme(self, theme_name: str):
        """Apply theme to launcher and viewer if open."""
        self.current_theme = theme_name
        apply_theme(self, theme_name)
        if self.viewer:
            apply_theme(self.viewer, theme_name)
    
    def _ensure_sam_loaded(self) -> bool:
        """Load SAM model if not already loaded."""
        if self.sam is not None:
            return True
        
        # Get selected model
        selected = self.model_combo.currentText()
        model_file, model_type = self.model_files[selected]
        
        base_path = Path(get_base_path())
        model_path = base_path / "models" / model_file
        
        if not model_path.exists():
            QMessageBox.critical(self, "錯誤", f"找不到 SAM 模型: {model_path}\n請先下載模型檔案。")
            return False
        
        try:
            from modules.presentation.qt.progress_dialog import ThemedProgressDialog
            
            progress = ThemedProgressDialog("載入中", f"正在載入 {selected} 模型...", self)
            progress.show()
            QApplication.processEvents()
            
            self.sam = SamEngine(model_path, model_type=model_type, device="cuda")
            self.sam.load()
            
            progress.close()
            return True
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"載入 SAM 模型失敗: {e}")
            return False
    
    def _open_image(self):
        """Open single image for segmentation."""
        if not self._ensure_sam_loaded():
            return
        
        f, _ = QFileDialog.getOpenFileName(
            self, 
            "選擇影像", 
            str(Path.home()), 
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not f:
            return
        
        img_path = Path(f)
        self._launch_viewer([img_path], f"分割檢視 - {img_path.name}")
    
    def _open_folder(self):
        """Open all images in a folder for segmentation."""
        if not self._ensure_sam_loaded():
            return
        
        folder = QFileDialog.getExistingDirectory(self, "選擇資料夾", str(Path.home()))
        if not folder:
            return
        
        folder_path = Path(folder)
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        imgs = [p for p in sorted(folder_path.glob("*")) if p.is_file() and p.suffix.lower() in exts]
        
        if not imgs:
            QMessageBox.information(self, "提示", "該資料夾內沒有支援格式的影像檔。")
            return
        
        self._launch_viewer(imgs, f"分割檢視 - {folder_path.name}")
    
    def _launch_viewer(self, image_paths, title: str):
        """Launch segmentation viewer with given images."""
        def compute_masks(path, points_per_side, pred_iou_thresh):
            return self.sam.auto_masks_from_image(
                path, 
                points_per_side=points_per_side, 
                pred_iou_thresh=pred_iou_thresh
            )
        
        self.viewer = SegmentationViewer(
            None,
            image_paths,
            compute_masks,
            title=title
        )
        
        # Apply current theme to viewer
        apply_theme(self.viewer, self.current_theme)
        
        self.viewer.show()


def main():
    app = QApplication(sys.argv)
    
    launcher = SegmentationLauncher()
    launcher.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
