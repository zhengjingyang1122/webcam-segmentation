"""SAM-Labeling-Tool 的分割控制器。

此控制器處理所有與使用 Segment Anything 模型進行自動影像和影片分割相關的操作。
它管理 SAM 模型的載入和卸載，顯示用於選擇要分割的影像或資料夾的選單，
並協調分割檢視視窗的啟動。透過將此邏輯收集在專用類別中，
減少了龐大的 ``Actions`` 類別的職責，使程式碼庫更易於理解和擴充。

請注意，此控制器仍與 Qt 小工具互動，並可能直接顯示訊息框。
進一步的解耦（例如透過注入回調函數或使用事件匯流排）留待未來工作。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox

from modules.presentation.qt.segmentation.segmentation_viewer import (
    SegmentationViewer,
)
from utils.utils import clear_current_path_manager
from utils.get_base_path import get_base_path
from modules.presentation.qt.ui_state import update_ui_state

try:
    from ..infrastructure.vision import sam_engine as sam_engine_mod
except ImportError:
    sam_engine_mod = None


logger = logging.getLogger(__name__)


DEFAULT_SAM_MODEL_TYPE = "vit_h"
# Default checkpoint path.  The repository now stores SAM weights under
# a single ``model`` directory rather than ``models``.  The H‑size model
# (vit_h) uses the 4b8939 checkpoint name.  See ``MODEL_FILE_NAMES``
# below for mapping of other model types to filenames.
DEFAULT_SAM_CKPT = Path(get_base_path()) / "models" / "sam_vit_h_4b8939.pth"
DEFAULT_SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Mapping of supported SAM model types to their expected filename under ``./model``
MODEL_FILE_NAMES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}


class SegmentationController:
    """封裝分割相關行為。

    參數
    ----------
    win : object
        主視窗或小工具，提供對 UI 元素（如目錄編輯框和狀態列）的存取。
    explorer_ctrl : object, optional
        可選的檔案瀏覽控制器，用於獲取最後一張影像或影片路徑，並綁定右鍵選單。
    sam_engine_instance : object, optional
        可選的預先實例化 SamEngine。如果未提供，``SegmentationController``
        將在首次需要時延遲建立一個。
    """

    def __init__(
        self,
        win: object,
        explorer_ctrl: Optional[object] = None,
        sam_engine_instance: Optional[object] = None,
    ) -> None:
        self.w = win
        self.explorer = explorer_ctrl
        self.sam = sam_engine_instance
        self._last_ckpt: Optional[Path] = None
        # Default parameters for segmentation
        self.default_params = {
            "points_per_side": 32,
            "pred_iou_thresh": 0.88,
            "union_morph_enabled": True,
            "union_morph_scale": 0.003,
            "fit_on_open": True,
        }
        # Connect Explorer signal for right‑click segmentation requests
        try:
            dock = getattr(self.explorer, "explorer", None)
            if dock is not None and hasattr(dock, "files_segment_requested"):
                dock.files_segment_requested.connect(self.open_segmentation_for_file_list)
        except Exception:
            pass

        # 連接 UI 下拉選單以便在使用者切換模型或運算方式時卸載當前模型。
        try:
            # 監聽模型大小改變
            if hasattr(self.w, "sam_model_combo") and hasattr(self.w.sam_model_combo, "currentIndexChanged"):
                self.w.sam_model_combo.currentIndexChanged.connect(self._on_sam_settings_changed)
            # 監聽運行方式改變
            if hasattr(self.w, "sam_device_combo") and hasattr(self.w.sam_device_combo, "currentIndexChanged"):
                self.w.sam_device_combo.currentIndexChanged.connect(self._on_sam_settings_changed)
        except Exception:
            # 如果連接失敗（例如在非 Qt 環境），忽略
            pass

    # Model loading and unloading
    def _ensure_sam_loaded_interactive(self) -> bool:
        """確保 SAM 模型已載入，必要時提示使用者。

        如果模型已載入，此方法立即回傳。否則，它會嘗試載入預設權重檔，
        或提示使用者下載或選擇權重檔。任何失敗都會導致錯誤訊息並回傳 ``False``。
        """
        # Already loaded
        if self._resolve_callable(self.sam, ["auto_masks_from_image"]):
            return True
        # If the module is missing, we cannot proceed
        if sam_engine_mod is None or not hasattr(sam_engine_mod, "SamEngine"):
            QMessageBox.warning(
                self.w,
                "無法載入",
                "找不到 SamEngine 類別，請確認 modules/infrastructure/vision/sam_engine.py",
            )
            return False

        # 先根據介面決定模型類型
        model_type = DEFAULT_SAM_MODEL_TYPE
        try:
            if hasattr(self.w, "sam_model_combo"):
                data = self.w.sam_model_combo.currentData()
                if isinstance(data, str) and data:
                    model_type = data
        except Exception:
            pass

        ckpt: Optional[Path] = self._last_ckpt
        # 對應模型類型至預設檔名，優先從已載入記憶的 ckpt 讀取
        if ckpt is None or not Path(ckpt).exists():
            # 根據當前選擇的模型類型決定預設檔案名稱
            fname = MODEL_FILE_NAMES.get(model_type)
            if fname:
                candidate = Path(get_base_path()) / "models" / fname
                if candidate.exists():
                    ckpt = candidate
                else:
                    # 對 vit_h 模型允許下載預設權重；其餘類型則需手動選擇
                    if model_type == DEFAULT_SAM_MODEL_TYPE:
                        try:
                            maybe = self._download_sam_with_prompt()
                            if maybe is not None:
                                ckpt = maybe
                        except Exception as e:
                            logger.exception("下載 SAM 權重失敗")
                            QMessageBox.critical(self.w, "下載 SAM 權重失敗", str(e))
                    else:
                        ckpt = None

        # As a last resort ask the user to pick a .pth file
        if ckpt is None or not Path(ckpt).exists():
            # 未找到對應權重時，讓使用者選擇檔案
            f, _ = QFileDialog.getOpenFileName(
                self.w,
                "選擇 SAM 權重檔 .pth",
                str(Path.home()),
                "SAM Checkpoint (*.pth *.pt)",
            )
            if not f:
                return False
            chosen = Path(f)
            # 檢查所選檔案名稱是否含有對應模型類型的識別字串，避免載入錯誤權重
            expected_tag = f"sam_{model_type}"
            if expected_tag not in chosen.name:
                QMessageBox.warning(
                    self.w,
                    "權重不符",
                    f"所選檔案與模型大小 {model_type} 不匹配，請選擇對應的權重檔 (包含 {expected_tag})。",
                )
                return False
            ckpt = chosen

        try:
            # 讀取使用者所選模型類型；若無則使用預設
            model_type = DEFAULT_SAM_MODEL_TYPE
            try:
                if hasattr(self.w, "sam_model_combo"):
                    # 使用 userData 儲存的模型代號，如 vit_h、vit_l、vit_b
                    data = self.w.sam_model_combo.currentData()
                    if isinstance(data, str) and data:
                        model_type = data
            except Exception:
                pass

            # 取得使用者所選的運算裝置 (GPU or CPU)；如未提供則預設由 SamEngine 判斷
            device = None
            try:
                if hasattr(self.w, "sam_device_combo"):
                    text = self.w.sam_device_combo.currentText().strip().lower()
                    if text == "gpu":
                        device = "cuda"
                    elif text == "cpu":
                        device = "cpu"
            except Exception:
                pass
            # 建立 SamEngine，傳入指定的 model_type 與 device
            self.sam = sam_engine_mod.SamEngine(Path(ckpt), model_type=model_type, device=device)
            # Show a simulated loading animation via the status footer.
            # Use start_scifi_simulated to provide the start/stop range parameters.
            self.w.status.start_scifi_simulated(
                "載入 SAM 模型中...", start=25, stop_at=99
            )
            self.sam.load()
            self.w.status.stop_scifi("狀態：模型已載入")
            self._last_ckpt = Path(ckpt)
            return True
        except Exception as e:
            self.w.status.stop_scifi("狀態：模型載入失敗")
            logger.exception("SAM 模型載入失敗")
            QMessageBox.critical(self.w, "載入失敗", str(e))
            self.sam = None
            return False

    def toggle_preload_sam(self, checked: bool) -> None:
        """回應「預先載入 SAM 模型」核取方塊切換的插槽。"""
        if checked:
            ok = self._ensure_sam_loaded_interactive()
            if not ok:
                # disable the checkbox until model is loaded
                self.w.chk_preload_sam.blockSignals(True)
                self.w.chk_preload_sam.setChecked(False)
                self.w.chk_preload_sam.blockSignals(False)
        else:
            try:
                if self.sam and self._resolve_callable(self.sam, ["unload"]):
                    self.w.status.start_scifi("卸載 SAM 模型中...")
                    try:
                        self.sam.unload()
                    finally:
                        self.w.status.stop_scifi("狀態：模型已卸載")
                else:
                    self.w.status.message("狀態：模型已卸載")
                self.sam = None
            except Exception as e:
                self.w.status.stop_scifi("狀態：模型卸載失敗")
                QMessageBox.warning(self.w, "卸載警告", str(e))

    def _on_sam_settings_changed(self) -> None:
        """回應 SAM 設定（模型大小或裝置）的變更。

        當使用者選擇不同的模型或裝置時，卸載目前載入的 SAM 引擎（如果有的話）以釋放資源。
        隨後的分割請求將使用新設定延遲重新載入模型。
        """
        # 如果尚未載入則無需處理
        if not self._resolve_callable(self.sam, ["auto_masks_from_image"]):
            return
        try:
            # 卸載當前模型
            if self._resolve_callable(self.sam, ["unload"]):
                self.w.status.start_scifi("卸載 SAM 模型中...")
                try:
                    self.sam.unload()
                finally:
                    self.w.status.stop_scifi("狀態：模型已卸載")
            else:
                self.w.status.message("狀態：模型已卸載")
        except Exception as e:
            # 若卸載失敗，仍清空 sam 並顯示警告
            try:
                self.w.status.stop_scifi("狀態：模型卸載失敗")
            except Exception:
                pass
            QMessageBox.warning(self.w, "卸載警告", str(e))
        # 清除引用與之前保存的 ckpt，下一次需要時再載入
        self.sam = None
        self._last_ckpt = None

    # Segmentation menu actions
    def open_auto_segment_menu(self) -> None:
        """顯示上下文選單，允許使用者選擇分割目標。"""
        btn = getattr(self.w, "btn_auto_seg_image", None)
        menu = QMenu(self.w)

        act_single = QAction("自動分割：選擇單一影像...", self.w)
        act_folder = QAction("自動分割：瀏覽資料夾（批次）...", self.w)

        act_single.triggered.connect(self.open_segmentation_view_for_chosen_image)
        act_folder.triggered.connect(self.open_segmentation_view_for_folder_prompt)

        menu.addAction(act_single)
        menu.addAction(act_folder)
        menu.exec(btn.mapToGlobal(QPoint(0, btn.height())))

    def open_segmentation_view_for_chosen_image(self) -> None:
        """提示使用者選擇單一影像並開啟分割檢視。"""
        if not self._ensure_sam_available(interactive=True):
            return
        f, _ = QFileDialog.getOpenFileName(
            self.w,
            "選擇影像",
            str(self.w.dir_edit.text()),
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not f:
            return
        path = Path(f)
        imgs = self._collect_images_with_pivot_first(path)
        compute_masks_fn = self._make_compute_fn_for_image()
        self._open_view(imgs, compute_masks_fn, title=f"自動分割檢視（{path.name}）")

    def open_segmentation_view_for_folder_prompt(self) -> None:
        """提示使用者選擇資料夾並執行批次分割。"""
        if not self._ensure_sam_available(interactive=True):
            return
        d = QFileDialog.getExistingDirectory(
            self.w, "選擇資料夾", str(self.w.dir_edit.text())
        )
        if not d:
            return
        folder = Path(d)
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        imgs = [p for p in sorted(folder.glob("*")) if p.is_file() and p.suffix.lower() in exts]
        if not imgs:
            QMessageBox.information(self.w, "沒有影像", "該資料夾內沒有支援格式的影像檔。")
            return
        compute_masks_fn = self._make_compute_fn_for_image()
        # Precompute cache for each image
        try:
            self.w.status.start_scifi("批次分割中：建立快取與 embedding")
            for p in imgs:
                try:
                    pps = self.default_params["points_per_side"]
                    iou = self.default_params["pred_iou_thresh"]
                    self.sam.auto_masks_from_image_cached(
                        p, points_per_side=pps, pred_iou_thresh=iou
                    )
                except Exception:
                    logger.exception("批次建立快取時發生錯誤: %s", p)
        finally:
            self.w.status.stop_scifi()
        self._open_view(imgs, compute_masks_fn, title=f"自動分割檢視（{folder.name}）")

    def open_segmentation_view_for_last_photo(self) -> None:
        """為最近拍攝的照片開啟分割檢視。"""
        if not self._ensure_sam_available(interactive=True):
            return
        last = None
        if hasattr(self.explorer, "last_image_path"):
            last = self.explorer.last_image_path()
        if last is None or not Path(last).exists():
            f, _ = QFileDialog.getOpenFileName(
                self.w,
                "選擇影像",
                str(self.w.dir_edit.text()),
                "Images (*.png *.jpg *.jpeg *.bmp)",
            )
            if not f:
                return
            last = Path(f)
        else:
            last = Path(last)
        imgs = self._collect_images_with_pivot_first(last)
        compute_masks_fn = self._make_compute_fn_for_image()
        self._open_view(imgs, compute_masks_fn, title="自動分割檢視（上次拍攝影像）")

    def open_segmentation_view_for_video_file(self) -> None:
        """提示使用者選擇影片檔案並分割其第一幀。"""
        if not self._ensure_sam_available(interactive=True):
            return
        f, _ = QFileDialog.getOpenFileName(
            self.w,
            "選擇影片",
            str(self.w.dir_edit.text()),
            "Videos (*.mp4 *.mov *.avi *.mkv)",
        )
        if not f:
            return
        video_path = Path(f)
        compute_masks_fn = self._make_compute_fn_for_video_first_frame(video_path)
        self._open_view(
            [video_path], compute_masks_fn, title=f"影片第一幀分割檢視（{video_path.name}）"
        )

    def open_segmentation_view_for_last_video(self) -> None:
        """為最近錄製的影片的第一幀開啟分割檢視。"""
        if not self._ensure_sam_available(interactive=True):
            return
        if hasattr(self.explorer, "last_video_path"):
            vp = self.explorer.last_video_path()
        else:
            vp = None
        if vp is None or not Path(vp).exists():
            return self.open_segmentation_view_for_video_file()
        compute_masks_fn = self._make_compute_fn_for_video_first_frame(Path(vp))
        self._open_view(
            [Path(vp)], compute_masks_fn, title=f"影片第一幀分割檢視（{Path(vp).name}）"
        )

    # Explorer context menu entry point
    def open_segmentation_for_file_list(self, file_list: List[str]) -> None:
        """為來自檔案瀏覽器的檔案列表開啟分割檢視。"""
        if not file_list:
            return
        
        # The incoming file_list is List[str], convert to List[Path]
        paths = [Path(p) for p in file_list]

        if not self._ensure_sam_available(interactive=True):
            return
        imgs: List[Path] = []
        videos: List[Path] = []
        video_exts = {".mp4", ".mov", ".avi", ".mkv"}
        for p in paths:
            if p.suffix.lower() in video_exts:
                videos.append(p)
            else:
                imgs.append(p)
        if imgs:
            # Use pivot order for the first image only
            ordered = []
            for i, img in enumerate(imgs):
                if i == 0:
                    ordered.extend(self._collect_images_with_pivot_first(Path(img)))
                else:
                    ordered.append(Path(img))
            compute_masks_fn = self._make_compute_fn_for_image()
            self._open_view(ordered, compute_masks_fn, title="自動分割檢視（檔案選擇）")
        elif videos:
            # Segment first frame of the first video only
            compute_masks_fn = self._make_compute_fn_for_video_first_frame(Path(videos[0]))
            self._open_view(
                [Path(videos[0])], compute_masks_fn, title=f"影片第一幀分割檢視（{Path(videos[0]).name}）"
            )

    # Private helper to open a segmentation viewer
    def _open_view(self, image_paths, compute_masks_fn, title: str) -> None:
        if not hasattr(self, "_seg_windows"):
            self._seg_windows = []
        from utils.utils import get_path_manager
        base_dir = Path(self.w.dir_edit.text())
        pm = None
        try:
            # Derive a path manager from the first image path
            p = Path(image_paths[0])
            timestamp = p.parent.parent.name
            pm = get_path_manager(base_dir, timestamp=timestamp)
        except Exception:
            # External file – do not use a path manager
            pass
        params = {
            "points_per_side": self.default_params["points_per_side"],
            "pred_iou_thresh": self.default_params["pred_iou_thresh"],
            "union_morph_enabled": self.default_params["union_morph_enabled"],
            "union_morph_scale": self.default_params["union_morph_scale"],
            "fit_on_open": self.default_params["fit_on_open"],
        }
        viewer = SegmentationViewer(
            None,
            image_paths,
            compute_masks_fn,
            params_defaults=params,
            title=title,
            path_manager=pm,
        )
        viewer.setAttribute(Qt.WA_DeleteOnClose, True)
        viewer.setWindowFlag(Qt.Window, True)

        def _drop_ref(*_):
            if viewer in self._seg_windows:
                self._seg_windows.remove(viewer)

        try:
            viewer.destroyed.connect(_drop_ref)
        except Exception:
            pass
        self._seg_windows.append(viewer)
        viewer.show()
        viewer.raise_()
        viewer.activateWindow()

    # Download helper
    def _download_sam_with_prompt(self) -> Optional[Path]:
        dst = Path(get_base_path()) / "models" / "sam_vit_h_4b8939.pth"
        if dst.exists():
            return dst
        ret = QMessageBox.question(
            self.w,
            "下載 SAM 權重",
            f"找不到預設 SAM 權重檔:\n{dst}\n\n要立即下載並儲存到該位置嗎？\n檔案約 2.5GB，時間視網路速度而定。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if ret != QMessageBox.Yes:
            return None
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            self.w.status.start_scifi("下載 SAM 權重中...")
            last_percent = -1

            def hook(blocknum, blocksize, totalsize):
                nonlocal last_percent
                if totalsize > 0:
                    percent = int(min(100, (blocknum * blocksize * 100) // totalsize))
                    if percent != last_percent:
                        last_percent = percent
                        self.w.status.set_scifi_progress(percent, f"下載 SAM 權重中... {percent}%")
            from urllib.request import urlretrieve
            urlretrieve(DEFAULT_SAM_URL, str(dst), reporthook=hook)
            self.w.status.stop_scifi("狀態：SAM 權重下載完成")
            return dst
        except Exception as e:
            self.w.status.stop_scifi("狀態：SAM 權重下載失敗")
            logger.exception("下載 SAM 權重失敗")
            QMessageBox.critical(self.w, "下載失敗", str(e))
            try:
                if dst.exists():
                    dst.unlink()
            except Exception:
                logger.warning("清理未完成 SAM 權重暫存檔失敗", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_callable(self, obj, method_names: List[str]) -> bool:
        """Check if an object has a callable method with one of the given names."""
        if obj is None:
            return False
        for name in method_names:
            if hasattr(obj, name) and callable(getattr(obj, name)):
                return True
        return False

    def _collect_images_from_dir(self, folder: Path) -> List[Path]:
        """Collect all supported image files from a directory."""
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        return [p for p in sorted(folder.glob("*")) if p.is_file() and p.suffix.lower() in exts]

    def _safe_resolve(self, p: Path) -> Path:
        """Resolve a path, returning the absolute path."""
        try:
            return p.resolve()
        except Exception:
            return p.absolute()

    def _collect_images_with_pivot_first(self, pivot: Path) -> List[Path]:
        """Collect images from the same directory as pivot, placing pivot first."""
        folder = pivot.parent
        imgs = self._collect_images_from_dir(folder)
        # Ensure pivot is first
        try:
            pivot_abs = self._safe_resolve(pivot)
            imgs.sort(key=lambda p: (0 if self._safe_resolve(p) == pivot_abs else 1, p.name))
        except Exception:
            pass
        return imgs

    def _ensure_sam_available(self, interactive: bool = False) -> bool:
        """Check if SAM is available, optionally prompting to load it."""
        if self._resolve_callable(self.sam, ["auto_masks_from_image"]):
            return True
        if interactive:
            return self._ensure_sam_loaded_interactive()
        return False

    def _make_compute_fn_for_image(self) -> Callable:
        """Create a callback function for computing masks from an image."""
        def compute(img_path, **kwargs):
            if not self.sam:
                raise RuntimeError("SAM model not loaded")
            return self.sam.auto_masks_from_image(img_path, **kwargs)
        return compute

    def _make_compute_fn_for_video_first_frame(self, video_path: Path) -> Callable:
        """Create a callback function for computing masks from a video's first frame."""
        def compute(dummy_path, **kwargs):
            if not self.sam:
                raise RuntimeError("SAM model not loaded")
            return self.sam.auto_masks_from_video_first_frame(video_path, **kwargs)
        return compute