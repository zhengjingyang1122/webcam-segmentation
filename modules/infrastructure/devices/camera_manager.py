# modules/infrastructure/devices/camera_manager.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import QVideoSink
from PySide6.QtMultimedia import (
    QAudioInput,
    QCamera,
    QCameraDevice,
    QImageCapture,
    QMediaCaptureSession,
    QMediaDevices,
    QMediaFormat,
    QMediaRecorder,
    QVideoFrame,
)
from PySide6.QtMultimediaWidgets import QVideoWidget

from modules.app.config_manager import config
from modules.infrastructure.io.burst import BurstShooter
from modules.infrastructure.io.photo import PhotoCapture
from modules.infrastructure.io.recorder import VideoRecorder

logger = logging.getLogger(__name__)


class CameraManager(QObject):
    """封裝相機裝置清單、啟停、Session 與控制器建置"""

    focusUpdated = Signal(float, bool)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._camera: Optional[QCamera] = None
        self._session: Optional[QMediaCaptureSession] = None
        self._image: Optional[QImageCapture] = None
        self._recorder: Optional[QMediaRecorder] = None
        self._audio: Optional[QAudioInput] = None
        self._selected: Optional[QCameraDevice] = None

        self._focus_threshold: float = 120.0
        self._frame_counter: int = 0
        self._filter_type: str = "None"

        # 封裝後對外提供的控制器
        self.photo: Optional[PhotoCapture] = None
        self.burst: Optional[BurstShooter] = None
        self.rec: Optional[VideoRecorder] = None

    def _on_focus_score_calculated(self, score: float):
        """接收來自 Sink 的原始分數, 判斷後對外發射最終訊號"""
        if score < 1.0:  # 過濾掉純黑畫面等無效分數
            return
        is_sharp = score >= self._focus_threshold
        self.focusUpdated.emit(score, is_sharp)

    def set_focus_threshold(self, value: float):
        """設定用於判斷清晰度的對焦分數閾值。"""
        self._focus_threshold = max(0, float(value))

    def get_focus_threshold(self) -> float:
        """取得目前的對焦分數閾值。"""
        return self._focus_threshold

    def set_filter_type(self, filter_type: str):
        """設定目前的濾鏡類型（例如 'None', 'Grayscale', 'Sepia'）。"""
        self._filter_type = filter_type

    def get_filter_type(self) -> str:
        """取得目前的濾鏡類型。"""
        return self._filter_type

    def _process_frame(self, frame: QVideoFrame):
        """此方法在 videoSink().videoFrameChanged 訊號觸發時執行"""
        try:
            self._frame_counter += 1
            if self._frame_counter % 5 != 0:  # 每 5 幀評估一次以降低負載
                return

            if not frame.isValid() or frame.size().isEmpty():
                return

            img = frame.toImage()
            if img.isNull():
                return

            if img.format() != QImage.Format.Format_Grayscale8:
                img = img.convertToFormat(QImage.Format.Format_Grayscale8)

            ptr = img.constBits()
            arr = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine())[
                :, : img.width()
            ]

            score = float(cv2.Laplacian(arr, cv2.CV_64F).var())

            self._on_focus_score_calculated(score)

        except Exception:
            logger.debug("Frame processing for focus score failed", exc_info=True)

    def _apply_filter_to_file(self, path: str, filter_type: str):
        """將濾鏡應用於已儲存的影像檔案。"""
        try:
            p = Path(path)
            if not p.exists():
                logger.warning(f"File not found for filtering: {path}")
                return

            # Read image using cv2 (handle unicode paths)
            stream = open(p, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            stream.close()

            if img is None:
                logger.warning(f"Failed to read image for filtering: {path}")
                return

            # Apply filter
            if filter_type == "Grayscale":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif filter_type == "Sepia":
                # Sepia kernel for BGR:
                # B = 0.272R + 0.534G + 0.131B -> [0.131, 0.534, 0.272]
                # G = 0.349R + 0.686G + 0.168B -> [0.168, 0.686, 0.349]
                # R = 0.393R + 0.769G + 0.189B -> [0.189, 0.769, 0.393]
                kernel = np.array([[0.131, 0.534, 0.272],
                                   [0.168, 0.686, 0.349],
                                   [0.189, 0.769, 0.393]])
                img = cv2.transform(img, kernel)

            # Save back
            is_success, im_buf = cv2.imencode(p.suffix, img)
            if is_success:
                im_buf.tofile(str(p))
                logger.info(f"Applied filter {filter_type} to {path}")
            else:
                logger.warning(f"Failed to save filtered image: {path}")

        except Exception:
            logger.exception(f"Error applying filter {filter_type} to {path}")

    def _find_best_camera_format(self, dev: QCameraDevice):
        """根據設定偏好尋找最佳的相機格式。"""
        try:
            cam_cfg = config["performance"]["camera"]
            pref_width = cam_cfg["preferred_resolution"]["width"]
            pref_height = cam_cfg["preferred_resolution"]["height"]
            pref_fps = cam_cfg["preferred_framerate"]
        except KeyError:
            logger.debug("Camera performance preferences not found in config.")
            return None

        supported_formats = dev.videoFormats()
        if not supported_formats:
            return None

        best_format = None
        highest_score = -1e9  # Use a very small number for initial score

        for fmt in supported_formats:
            res = fmt.resolution()
            width, height = res.width(), res.height()
            # Frame rate can be a range, we check the maximum
            fps = fmt.maxFrameRate()

            score = 0
            # Heavily penalize formats with lower resolution than preferred
            if width < pref_width or height < pref_height:
                score -= 100000

            # Score based on how close the resolution area is
            score -= abs(width * height - pref_width * pref_height)

            # Bonus for matching framerate, with penalties for being off
            if fps >= pref_fps:
                score += 1000  # Bonus for meeting or exceeding framerate
                score -= (fps - pref_fps) * 10  # Smaller penalty for being over
            else:
                score -= (pref_fps - fps) * 50  # Heavier penalty for being under

            if score > highest_score:
                highest_score = score
                best_format = fmt

        if best_format:
            res = best_format.resolution()
            logger.info(
                f"Selected camera format: {res.width()}x{res.height()} "
                f"@{best_format.maxFrameRate():.2f}fps (based on user preference)"
            )
        return best_format

    def list_devices(self) -> list[tuple[str, QCameraDevice]]:
        """列出可用的視訊輸入裝置。"""
        devs = list(QMediaDevices.videoInputs())
        out = []
        for i, d in enumerate(devs):
            try:
                name = d.description()
            except Exception:
                name = f"Camera {i}"
            out.append((name, d))
        return out

    def set_selected_device_index(self, idx: int):
        """透過列表中的索引選擇相機裝置。"""
        devs = list(QMediaDevices.videoInputs())
        self._selected = devs[idx] if 0 <= idx < len(devs) else None

    def start(
        self,
        video_widget: QVideoWidget,
        on_image_saved: Optional[Callable[[int, str], None]] = None,
        on_image_error: Optional[Callable[[int, int, str], None]] = None,
        on_rec_state_changed: Optional[Callable[[int], None]] = None,
        on_rec_error: Optional[Callable[[int, str], None]] = None,
    ):
        """啟動相機並初始化擷取 Session。"""
        if self._camera is not None:
            logger.info("Camera already active, skip start()")
            return

        dev = self._selected or QMediaDevices.defaultVideoInput()
        self._camera = QCamera(dev)

        # --- Find and set best camera format based on config ---
        best_format = self._find_best_camera_format(dev)
        if best_format:
            try:
                self._camera.setCameraFormat(best_format)
            except Exception:
                logger.warning("Failed to set preferred camera format.", exc_info=True)

        self._session = QMediaCaptureSession()
        self._session.setCamera(self._camera)

        # 1. 設定 UI 顯示
        self._session.setVideoOutput(video_widget)

        # 2. 連接影像幀處理
        sink = video_widget.videoSink()
        if sink:
            sink.videoFrameChanged.connect(self._process_frame)
        else:
            logger.warning("無法從 QVideoWidget 取得 videoSink，對焦計算功能將被停用。")

        self._image = QImageCapture()
        self._session.setImageCapture(self._image)

        # 註冊影像訊號
        if on_image_saved:
            # Wrap the callback to apply filter
            def wrapped_on_image_saved(id, path):
                if self._filter_type and self._filter_type != "None":
                    self._apply_filter_to_file(path, self._filter_type)
                on_image_saved(id, path)
            
            self._image.imageSaved.connect(wrapped_on_image_saved)
        
        if on_image_error:
            self._image.errorOccurred.connect(on_image_error)

        self._audio = QAudioInput(QMediaDevices.defaultAudioInput())
        self._session.setAudioInput(self._audio)

        self._recorder = QMediaRecorder()
        self._session.setRecorder(self._recorder)

        # 媒體格式（從設定檔讀取）
        try:
            rec_cfg = config["performance"]["video_recording"]
            fmt = QMediaFormat()

            # --- Map container format ---
            container_map = {
                "mp4": QMediaFormat.FileFormat.MPEG4,
                "avi": QMediaFormat.FileFormat.AVI,
            }
            container_format = container_map.get(
                rec_cfg["container"], QMediaFormat.FileFormat.MPEG4
            )
            fmt.setFileFormat(container_format)

            # --- Map video codec ---
            video_codec_map = {
                "avc1": QMediaFormat.VideoCodec.H264,
                "h264": QMediaFormat.VideoCodec.H264,
                "mpeg4": QMediaFormat.VideoCodec.MPEG4,
            }
            video_codec = video_codec_map.get(rec_cfg["codec"], QMediaFormat.VideoCodec.H264)
            fmt.setVideoCodec(video_codec)

            # --- For simplicity, audio codec is kept for now ---
            fmt.setAudioCodec(QMediaFormat.AudioCodec.AAC)
            self._recorder.setMediaFormat(fmt)

            # --- Map quality ---
            quality_map = {
                "low": QMediaRecorder.Quality.LowQuality,
                "normal": QMediaRecorder.Quality.NormalQuality,
                "high": QMediaRecorder.Quality.HighQuality,
            }
            quality = quality_map.get(rec_cfg["quality"], QMediaRecorder.Quality.NormalQuality)
            self._recorder.setQuality(quality)

        except Exception:
            logger.warning("Recorder media format not fully set", exc_info=True)
        # 註冊錄影訊號
        if on_rec_state_changed:
            self._recorder.recorderStateChanged.connect(on_rec_state_changed)
        if on_rec_error:
            self._recorder.errorChanged.connect(on_rec_error)

        # 建立控制器
        self.photo = PhotoCapture(self._image, parent=self)
        self.burst = BurstShooter(self._image, parent=self)
        self.rec = VideoRecorder(self._recorder, parent=self)

        try:
            self._camera.start()
            logger.info("Camera started")
        except Exception:
            logger.exception("Camera start failed")
            raise

    def stop(self):
        """停止相機並釋放資源。"""
        # 先停止 recorder 與 camera
        try:
            if self.rec:
                self.rec.stop()
        except Exception:
            logger.warning("Recorder stop raised exception", exc_info=True)
        try:
            if self._camera:
                self._camera.stop()
        except Exception:
            logger.warning("Camera stop raised exception", exc_info=True)

        # 清理
        self.photo = None
        self.burst = None
        self.rec = None
        self._image = None
        self._recorder = None
        self._audio = None
        self._session = None
        self._camera = None

    def is_active(self) -> bool:
        """檢查相機目前是否處於活動狀態。"""
        if not self._camera:
            return False
        try:
            return self._camera.isActive()
        except Exception:
            from PySide6.QtMultimedia import QCamera as _QCam

            return self._camera.cameraState() == _QCam.ActiveState
