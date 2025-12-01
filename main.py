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
        self.resize(600, 250)
        
        self.sam = None
        self._active_viewers = []  # Track active viewer windows
        self.current_theme = "dark"  # Track current theme
        
        # Model selection mapping
        self.model_files = {
            "SAM-B (Fast)": ("sam_vit_b_01ec64.pth", "vit_b", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
            "SAM-L (Balanced)": ("sam_vit_l_0b3195.pth", "vit_l", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
            "SAM-H (Best Quality)": ("sam_vit_h_4b8939.pth", "vit_h", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
        }
        
        # Create menu bar
        self._create_menus()
        
        # Apply default dark theme
        apply_theme(self, "dark")
        
        # Build UI
        self._build_ui()

    def closeEvent(self, event):
        """Override close event to prevent closing if viewers are active."""
        if self._active_viewers:
            QMessageBox.warning(self, "警告", "請先關閉所有分割視窗後再結束程式。")
            event.ignore()
        else:
            event.accept()
    
    def _build_ui(self):
        """Build the launcher UI."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QStyle
        
        central = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 主要內容區域
        content_layout = QVBoxLayout()
        content_layout.setSpacing(15)
        content_layout.setContentsMargins(20, 10, 20, 20)

        # 頂部區域：作者與歡迎訊息 (群組化)
        author_group = QGroupBox("關於作者")
        author_layout = QHBoxLayout()
        author_layout.setContentsMargins(15, 15, 15, 15)
        
        # 左側：頭貼
        avatar_label = QLabel()
        avatar_path = Path(get_base_path()) / "assets" / "Coffee.png"
        if avatar_path.exists():
            pixmap = QPixmap(str(avatar_path))
            scaled_pixmap = pixmap.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            avatar_label.setPixmap(scaled_pixmap)
        else:
            avatar_label.setText("☕")
            avatar_label.setStyleSheet("font-size: 40px;")
        
        # 右側：文字訊息
        text_layout = QVBoxLayout()
        text_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        text_layout.setContentsMargins(20, 0, 15, 0)  # 左側留白 20px，右側留白 15px（與頭貼距離邊緣一致）
        
        # Title 改為 Tip 形式 (設定在頭貼上)
        avatar_label.setToolTip("Coffee ☕")
        
        msg_label = QLabel("It's a beautiful day to achieve great things.\nRemember to stay focused and take breaks.")
        msg_label.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 16px; 
                color: #bbb; 
                font-weight: 500;
                line-height: 2.0;
                background: transparent;
            }
        """)
        msg_label.setWordWrap(True)  # 允許換行
        msg_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # 設定大小策略，讓文字區域可以隨視窗寬度延展
        from PySide6.QtWidgets import QSizePolicy
        msg_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        msg_label.setMinimumWidth(300)  # 設定最小寬度
        msg_label.setMaximumWidth(800)  # 設定最大寬度，避免過寬
        
        text_layout.addWidget(msg_label)
        
        # 排版：頭貼固定最左，接著文字，右側彈簧
        author_layout.addWidget(avatar_label)
        author_layout.addLayout(text_layout, 1)  # 設定 stretch factor 為 1，讓 text_layout 可以延展
        author_layout.addStretch()
        
        author_group.setLayout(author_layout)
        
        # Settings group (Model & Device)
        settings_group = QGroupBox("系統設定")
        settings_layout = QHBoxLayout()
        
        # Model
        model_label = QLabel("模型:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.model_files.keys()))
        self.model_combo.setCurrentIndex(0)  # Default to SAM-B (Fast)
        
        # Device
        device_label = QLabel("運算:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto (自動)", "GPU", "CPU"])
        self.device_combo.setToolTip("優先使用 GPU 加速，若發生記憶體不足(OOM)會自動切換至 CPU")
        
        settings_layout.addWidget(model_label)
        settings_layout.addWidget(self.model_combo, 2)
        settings_layout.addWidget(device_label)
        settings_layout.addWidget(self.device_combo, 1)
        settings_group.setLayout(settings_layout)
        
        content_layout.addWidget(settings_group)
        
        # Path input group
        path_group = QGroupBox("快速執行")
        path_layout = QVBoxLayout()
        path_layout.setSpacing(10)
        
        # Image path
        img_path_layout = QHBoxLayout()
        img_path_label = QLabel("📄") # Icon for Image Path
        img_path_label.setToolTip("單一影像路徑")
        self.img_path_edit = QLineEdit()
        self.img_path_edit.setPlaceholderText("選擇單一影像檔案...")
        self.img_path_edit.setText(str(Path.home() / "Pictures"))
        
        btn_browse_img = QPushButton("...")
        btn_browse_img.setFixedWidth(30)
        btn_browse_img.clicked.connect(self._browse_image_path)
        
        btn_open_img = QPushButton("🖼️ 單一分割") # Icon for Action
        btn_open_img.setToolTip("執行單一影像分割")
        btn_open_img.clicked.connect(self._open_image_from_path)
        
        img_path_layout.addWidget(img_path_label)
        img_path_layout.addWidget(self.img_path_edit, 1)
        img_path_layout.addWidget(btn_browse_img)
        img_path_layout.addWidget(btn_open_img)
        
        # Folder path
        folder_path_layout = QHBoxLayout()
        folder_path_label = QLabel("📁") # Icon for Folder Path
        folder_path_label.setToolTip("資料夾路徑")
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("選擇包含影像的資料夾...")
        self.folder_path_edit.setText(str(Path.home() / "Pictures"))
        
        btn_browse_folder = QPushButton("...")
        btn_browse_folder.setFixedWidth(30)
        btn_browse_folder.clicked.connect(self._browse_folder_path)
        
        btn_open_folder = QPushButton("🗂️ 批次分割")  # 更換 icon 為檔案夾卡片
        btn_open_folder.setToolTip("執行資料夾批次分割")
        btn_open_folder.clicked.connect(self._open_folder_from_path)
        
        folder_path_layout.addWidget(folder_path_label)
        folder_path_layout.addWidget(self.folder_path_edit, 1)
        folder_path_layout.addWidget(btn_browse_folder)
        folder_path_layout.addWidget(btn_open_folder)
        
        path_layout.addLayout(img_path_layout)
        path_layout.addLayout(folder_path_layout)
        path_group.setLayout(path_layout)
        
        content_layout.addWidget(path_group)
        
        # 將作者群組移動到最下方
        content_layout.addWidget(author_group)
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
        view_menu.addAction(act_light)
        view_menu.addAction(act_dark)
        
        # Edit menu
        edit_menu = self.menuBar().addMenu("編輯")
        
        act_shortcuts = QAction("快捷鍵設定...", self)
        act_shortcuts.triggered.connect(self._show_shortcuts_dialog)
        
        edit_menu.addAction(act_shortcuts)

        # Help menu
        help_menu = self.menuBar().addMenu("說明")
        
        act_help = QAction("使用說明", self)
        act_help.triggered.connect(self._show_help)
        
        help_menu.addAction(act_help)
        
        # About menu
        about_menu = self.menuBar().addMenu("關於")
        
        act_about = QAction("關於本專案...", self)
        act_about.triggered.connect(self._show_about)
        
        about_menu.addAction(act_about)
    
    def _apply_theme(self, theme_name: str):
        """Apply theme to launcher and viewer if open."""
        self.current_theme = theme_name
        apply_theme(self, theme_name)
        for viewer in self._active_viewers:
            apply_theme(viewer, theme_name)
    
    def _check_and_download_model(self, model_path: Path, url: str) -> bool:
        """Check if model exists, if not, ask user to download."""
        if model_path.exists():
            return True
            
        reply = QMessageBox.question(
            self, 
            "模型缺失", 
            f"找不到模型檔案：{model_path.name}\n\n是否要現在下載？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return False
            
        # Download logic
        try:
            import urllib.request
            from PySide6.QtWidgets import QProgressDialog
            
            progress = QProgressDialog(f"正在下載 {model_path.name}...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setAutoClose(True)
            progress.show()
            
            def report_hook(block_num, block_size, total_size):
                if progress.wasCanceled():
                    raise InterruptedError("Download canceled")
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = int(downloaded * 100 / total_size)
                    progress.setValue(percent)
            
            # Ensure models directory exists
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            urllib.request.urlretrieve(url, str(model_path), report_hook)
            return True
            
        except InterruptedError:
            QMessageBox.warning(self, "下載取消", "模型下載已取消。")
            if model_path.exists():
                model_path.unlink() # Remove partial file
            return False
        except Exception as e:
            logger.error(f"下載模型失敗: {e}")
            QMessageBox.critical(self, "下載失敗", f"無法下載模型：{e}")
            if model_path.exists():
                model_path.unlink()
            return False

    def _ensure_sam_loaded(self) -> bool:
        """Load SAM model if not already loaded."""
        # Get selected model
        selected = self.model_combo.currentText()
        model_file, model_type, model_url = self.model_files[selected]
        
        # Get selected device
        device_idx = self.device_combo.currentIndex()
        device_map = {0: "auto", 1: "cuda", 2: "cpu"}
        device = device_map.get(device_idx, "auto")
        
        # 檢查是否需要重新載入（模型類型或裝置改變）
        if self.sam is not None:
            # Check if model type matches
            type_match = hasattr(self.sam, 'model_type') and self.sam.model_type == model_type
            # Check if requested device matches
            device_match = hasattr(self.sam, 'requested_device') and self.sam.requested_device == device
            
            if type_match and device_match:
                return True
            else:
                # 模型類型或裝置改變，需要卸載舊模型
                logger.info(f"設定改變，重新載入模型...")
                try:
                    self.sam.unload()
                    self.sam = None
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(f"卸載舊模型時發生錯誤: {e}")
        
        base_path = Path(get_base_path())
        model_path = base_path / "models" / model_file
        
        # Check and download
        if not self._check_and_download_model(model_path, model_url):
            return False
        
        try:
            from modules.presentation.qt.progress_dialog import ThemedProgressDialog
            
            progress = ThemedProgressDialog("載入中", f"正在載入 {selected} 模型...", self)
            progress.show()
            QApplication.processEvents()
            
            self.sam = SamEngine(model_path, model_type=model_type, device=device)
            self.sam.load()
            
            progress.close()
            logger.info(f"成功載入模型: {model_type} on {self.sam.device}")
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
        
        viewer = SegmentationViewer(
            None,
            image_paths,
            compute_masks,
            title=title
        )
        if hasattr(viewer, 'closed'):
            viewer.closed.connect(lambda: self._on_viewer_closed(viewer))
        else:
            viewer.destroyed.connect(lambda: self._on_viewer_closed(viewer))
            
    def _on_viewer_closed(self, viewer):
        """Handle viewer closing."""
        if viewer in self._active_viewers:
            self._active_viewers.remove(viewer)

    def _show_shortcuts_dialog(self):
        """Show shortcuts configuration dialog."""
        from modules.presentation.qt.shortcut_dialog import ShortcutEditorDialog
        dialog = ShortcutEditorDialog(self)
        dialog.exec()

    def _show_about(self):
        """Show about dialog."""
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

    def _show_help(self):
        """Show help dialog."""
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
            <li><b>PageUp / PageDown：</b> 切換上一張 / 下一張影像</li>
            <li><b>Ctrl + S：</b> 儲存目前已選取的目標</li>
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
        </ul>
        <hr>
        <p><i>Created by Coffee ☕</i></p>
        """
        QMessageBox.about(self, "使用說明", help_text)


def main():
    app = QApplication(sys.argv)
    
    launcher = SegmentationLauncher()
    launcher.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
