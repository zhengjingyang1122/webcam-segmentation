from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from PySide6.QtCore import QPoint, QRectF, Qt, Signal, QEvent
from PySide6.QtGui import QPainter, QTransform, QEventPoint
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget

from .utils import np_bgr_to_qpixmap

class ImageView(QGraphicsView):
    """A custom QGraphicsView for displaying images with zoom and pan support."""

    # Signals for drawing interaction
    drawing_started = Signal(int, int)  # x, y
    drawing_moved = Signal(int, int)    # x, y
    drawing_finished = Signal(int, int) # x, y

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
