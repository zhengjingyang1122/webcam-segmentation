import sys
import logging
from pathlib import Path
import torch
import warnings
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QComboBox, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QGroupBox, QLineEdit, QPushButton
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtCore import Qt
from modules.infrastructure.vision.sam_engine import SamEngine
from modules.presentation.qt.segmentation.segmentation_viewer import SegmentationViewer
from modules.presentation.qt.theme_manager import apply_theme
from utils.get_base_path import get_base_path

# 忽略 PyTorch 的 FutureWarning 和其他警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='.*torch.*')

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SegmentationLauncher(QMainWindow):
    """Launcher window for segmentation tool."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("影像標註工具 By Coffee☕ v1.0.0")
        self.resize(600, 200)
        
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
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 頂部區域：標題 + 頭貼
        top_widget = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(20, 20, 20, 10)
        
        # 左側：標題
        title_label = QLabel("SAM 影像標註工具")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        top_layout.addWidget(title_label)
        top_layout.addStretch()
        
        # 右側：作者頭貼
        avatar_label = QLabel()
        avatar_path = Path(get_base_path()) / "assets" / "Coffee.png"
        if avatar_path.exists():
            pixmap = QPixmap(str(avatar_path))
            scaled_pixmap = pixmap.scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            avatar_label.setPixmap(scaled_pixmap)
            avatar_label.setToolTip("作者: Coffee")
        else:
            avatar_label.setText("☕")
            avatar_label.setStyleSheet("font-size: 24px;")
        
        top_layout.addWidget(avatar_label)
        top_widget.setLayout(top_layout)
        main_layout.addWidget(top_widget)
        
        # 主要內容區域
        content_layout = QVBoxLayout()
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(20, 10, 20, 20)
        
        # Welcome label
        label = QLabel("歡迎使用影像標註工具\n\n請選擇模型並從「檔案」選單選擇要分割的影像")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 14px;")
        content_layout.addWidget(label)
        
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
        
        content_layout.addWidget(model_group)
        
        # Path input group
        path_group = QGroupBox("快速路徑")
        path_layout = QVBoxLayout()
        
        # Image path
        img_path_layout = QHBoxLayout()
        img_path_label = QLabel("影像路徑:")
        self.img_path_edit = QLineEdit()
        self.img_path_edit.setPlaceholderText("選擇單一影像檔案...")
        self.img_path_edit.setText(str(Path.home() / "Pictures"))
        btn_browse_img = QPushButton("瀏覽...")
        btn_browse_img.clicked.connect(self._browse_image_path)
        btn_open_img = QPushButton("開啟影像")
        btn_open_img.clicked.connect(self._open_image_from_path)
        
        img_path_layout.addWidget(img_path_label)
        img_path_layout.addWidget(self.img_path_edit, 1)
        img_path_layout.addWidget(btn_browse_img)
        img_path_layout.addWidget(btn_open_img)
        
        # Folder path
        folder_path_layout = QHBoxLayout()
        folder_path_label = QLabel("資料夾路徑:")
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("選擇包含影像的資料夾...")
        self.folder_path_edit.setText(str(Path.home() / "Pictures"))
        btn_browse_folder = QPushButton("瀏覽...")
        btn_browse_folder.clicked.connect(self._browse_folder_path)
        btn_open_folder = QPushButton("開啟資料夾")
        btn_open_folder.clicked.connect(self._open_folder_from_path)
        
        folder_path_layout.addWidget(folder_path_label)
        folder_path_layout.addWidget(self.folder_path_edit, 1)
        folder_path_layout.addWidget(btn_browse_folder)
        folder_path_layout.addWidget(btn_open_folder)
        
        path_layout.addLayout(img_path_layout)
        path_layout.addLayout(folder_path_layout)
        path_group.setLayout(path_layout)
        
        content_layout.addWidget(path_group)
        content_layout.addStretch()
        
        main_layout.addLayout(content_layout)
        central.setLayout(main_layout)
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
        # Get selected model
        selected = self.model_combo.currentText()
        model_file, model_type = self.model_files[selected]
        
        # 檢查是否需要重新載入（模型類型改變）
        if self.sam is not None:
            if hasattr(self.sam, 'model_type') and self.sam.model_type == model_type:
                return True
            else:
                # 模型類型改變，需要卸載舊模型
                logger.info(f"模型類型改變，卸載舊模型...")
                try:
                    self.sam.unload()
                    self.sam = None
                    # 強制垃圾回收
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(f"卸載舊模型時發生錯誤: {e}")
        
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
            logger.info(f"成功載入模型: {model_type}")
            return True
        except Exception as e:
            logger.error(f"載入 SAM 模型失敗: {e}", exc_info=True)
            QMessageBox.critical(self, "錯誤", f"載入 SAM 模型失敗: {e}")
            return False
    
    def _browse_image_path(self):
        """瀏覽並選擇影像檔案路徑"""
        f, _ = QFileDialog.getOpenFileName(
            self, 
            "選擇影像", 
            self.img_path_edit.text() if self.img_path_edit.text() else str(Path.home()), 
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if f:
            self.img_path_edit.setText(f)
    
    def _browse_folder_path(self):
        """瀏覽並選擇資料夾路徑"""
        folder = QFileDialog.getExistingDirectory(
            self, 
            "選擇資料夾", 
            self.folder_path_edit.text() if self.folder_path_edit.text() else str(Path.home())
        )
        if folder:
            self.folder_path_edit.setText(folder)
    
    def _open_image_from_path(self):
        """從路徑輸入欄位開啟影像"""
        if not self._ensure_sam_loaded():
            return
        
        path_text = self.img_path_edit.text().strip()
        if not path_text:
            QMessageBox.information(self, "提示", "請先輸入或選擇影像路徑")
            return
        
        img_path = Path(path_text)
        if not img_path.exists() or not img_path.is_file():
            QMessageBox.warning(self, "錯誤", f"找不到影像檔案：{path_text}")
            return
        
        self._launch_viewer([img_path], f"分割檢視 - {img_path.name}")
    
    def _open_folder_from_path(self):
        """從路徑輸入欄位開啟資料夾"""
        if not self._ensure_sam_loaded():
            return
        
        path_text = self.folder_path_edit.text().strip()
        if not path_text:
            QMessageBox.information(self, "提示", "請先輸入或選擇資料夾路徑")
            return
        
        folder_path = Path(path_text)
        if not folder_path.exists() or not folder_path.is_dir():
            QMessageBox.warning(self, "錯誤", f"找不到資料夾：{path_text}")
            return
        
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        imgs = [p for p in sorted(folder_path.glob("*")) if p.is_file() and p.suffix.lower() in exts]
        
        if not imgs:
            QMessageBox.information(self, "提示", "該資料夾內沒有支援格式的影像檔。")
            return
        
        self._launch_viewer(imgs, f"分割檢視 - {folder_path.name}")
    
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
