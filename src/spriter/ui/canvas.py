# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Pixel canvas widget — renders the active frame with zoom, pan, and grid.

The canvas converts mouse events to canvas-space coordinates and dispatches
them to the currently active :class:`~spriter.tools.base.Tool`.  After each
stroke the composited image cache is invalidated and the widget is repainted.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QWheelEvent
from PyQt6.QtWidgets import QWidget

from ..commands.base import CommandStack
from ..core.sprite import Sprite
from ..tools.base import Tool


class CanvasWidget(QWidget):
    """Interactive pixel-art canvas.

    Args:
        sprite: The sprite document to display and edit.
        stack: The undo/redo command stack.
        parent: Optional Qt parent widget.
    """

    # Emitted whenever the cursor moves over the canvas; carries (canvas_x, canvas_y).
    cursor_moved = pyqtSignal(int, int)
    # Emitted whenever the zoom level changes.
    zoom_changed = pyqtSignal(float)

    # Supported discrete zoom levels (factor relative to 1 pixel = 1 px).
    ZOOM_LEVELS = (1, 2, 4, 8, 16, 32, 48, 64)

    def __init__(
        self,
        sprite: Sprite,
        stack: CommandStack,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._sprite = sprite
        self._stack = stack
        self._tool: Optional[Tool] = None

        self._zoom: float = 1.0
        self._pan: QPointF = QPointF(0.0, 0.0)

        # Composite image cache — invalidated on any pixel edit.
        self._composite_cache: Optional[np.ndarray] = None

        # Pan state
        self._panning: bool = False
        self._pan_last: QPointF = QPointF()
        self._space_held: bool = False

        # Active editing context
        self._active_layer: int = 0
        self._active_frame: int = 0

        self.show_grid: bool = True

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(64, 64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def zoom(self) -> float:
        """Current zoom level (1.0 = 100%)."""
        return self._zoom

    @zoom.setter
    def zoom(self, value: float) -> None:
        clamped = max(
            float(self.ZOOM_LEVELS[0]), min(float(self.ZOOM_LEVELS[-1]), value)
        )
        if clamped != self._zoom:
            self._zoom = clamped
            self.zoom_changed.emit(self._zoom)
            self.update()

    @property
    def active_layer(self) -> int:
        """Index of the layer that receives edits."""
        return self._active_layer

    @active_layer.setter
    def active_layer(self, value: int) -> None:
        self._active_layer = value

    @property
    def active_frame(self) -> int:
        """Index of the currently displayed frame."""
        return self._active_frame

    @active_frame.setter
    def active_frame(self, value: int) -> None:
        if value != self._active_frame:
            self._active_frame = value
            self.invalidate_cache()

    def set_tool(self, tool: Tool) -> None:
        """Replace the active drawing tool.

        Args:
            tool: The new tool to activate.
        """
        self._tool = tool
        if tool is not None:
            tool.layer_index = self._active_layer
            tool.frame_index = self._active_frame

    def invalidate_cache(self) -> None:
        """Discard the composite image cache and schedule a repaint."""
        self._composite_cache = None
        self.update()

    def fit_to_window(self) -> None:
        """Set zoom and pan so the canvas fills the widget as closely as possible."""
        if self._sprite.width == 0 or self._sprite.height == 0:
            return
        w_ratio = self.width() / self._sprite.width
        h_ratio = self.height() / self._sprite.height
        self.zoom = min(w_ratio, h_ratio)
        self._pan = QPointF(0.0, 0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_composite(self) -> np.ndarray:
        if self._composite_cache is None:
            if self._sprite.frame_count > 0 and self._sprite.layer_count > 0:
                from ..core.compositor import composite_frame

                self._composite_cache = composite_frame(
                    self._sprite, self._active_frame
                )
            else:
                self._composite_cache = np.zeros(
                    (self._sprite.height, self._sprite.width, 4), dtype=np.uint8
                )
        return self._composite_cache

    def _canvas_offset(self) -> QPointF:
        """Top-left corner of the canvas image in widget coordinates."""
        cw = self._sprite.width * self._zoom
        ch = self._sprite.height * self._zoom
        return QPointF(
            (self.width() - cw) / 2.0 + self._pan.x(),
            (self.height() - ch) / 2.0 + self._pan.y(),
        )

    def _widget_to_canvas(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert a widget-space point to canvas pixel coordinates."""
        offset = self._canvas_offset()
        cx = int((wx - offset.x()) / self._zoom)
        cy = int((wy - offset.y()) / self._zoom)
        return cx, cy

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(50, 50, 50))

        composite = self._get_composite()
        h, w = composite.shape[:2]
        if w == 0 or h == 0:
            return

        offset = self._canvas_offset()
        scaled_w = int(w * self._zoom)
        scaled_h = int(h * self._zoom)

        # Checkerboard — shows transparent pixels.
        self._paint_checkerboard(painter, offset, scaled_w, scaled_h)

        # Convert numpy RGBA → QImage.
        arr = np.ascontiguousarray(composite)
        image = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        scaled = image.scaled(
            scaled_w,
            scaled_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        painter.drawImage(offset, scaled)

        # Pixel grid — only visible at higher zoom levels.
        if self.show_grid and self._zoom >= 4.0:
            self._paint_grid(painter, offset, w, h)

        # Tool preview overlay.
        if self._tool is not None:
            overlay = self._tool.preview_overlay()
            if overlay is not None:
                ov_arr = np.ascontiguousarray(overlay)
                ov_image = QImage(
                    ov_arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888
                )
                scaled_ov = ov_image.scaled(
                    scaled_w,
                    scaled_h,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
                painter.setOpacity(0.75)
                painter.drawImage(offset, scaled_ov)
                painter.setOpacity(1.0)

        painter.end()

    def _paint_checkerboard(
        self, painter: QPainter, offset: QPointF, w: int, h: int
    ) -> None:
        check = max(4, int(self._zoom * 2))
        light = QColor(200, 200, 200)
        dark = QColor(150, 150, 150)
        ox, oy = int(offset.x()), int(offset.y())
        row = 0
        while row < h:
            col = 0
            while col < w:
                parity = (row // check + col // check) % 2
                color = light if parity == 0 else dark
                painter.fillRect(
                    ox + col,
                    oy + row,
                    min(check, w - col),
                    min(check, h - row),
                    color,
                )
                col += check
            row += check

    def _paint_grid(
        self, painter: QPainter, offset: QPointF, cols: int, rows: int
    ) -> None:
        pen = QPen(QColor(100, 100, 100, 140))
        pen.setWidth(1)
        painter.setPen(pen)
        ox, oy = offset.x(), offset.y()
        z = self._zoom
        for col in range(cols + 1):
            x = int(ox + col * z)
            painter.drawLine(x, int(oy), x, int(oy + rows * z))
        for row in range(rows + 1):
            y = int(oy + row * z)
            painter.drawLine(int(ox), y, int(ox + cols * z), y)

    # ------------------------------------------------------------------
    # Mouse / keyboard events
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._zoom_step(1)
            elif delta < 0:
                self._zoom_step(-1)
            event.accept()
        else:
            event.ignore()

    def _zoom_step(self, direction: int) -> None:
        """Step zoom up (+1) or down (-1) through ZOOM_LEVELS."""
        levels = self.ZOOM_LEVELS
        current = self._zoom
        try:
            idx = levels.index(int(current))
        except ValueError:
            # Not a snap level — find nearest.
            diffs = [abs(lv - current) for lv in levels]
            idx = diffs.index(min(diffs))
        new_idx = max(0, min(len(levels) - 1, idx + direction))
        self.zoom = float(levels[new_idx])

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        is_middle = event.button() == Qt.MouseButton.MiddleButton
        is_left = event.button() == Qt.MouseButton.LeftButton
        if is_middle or (is_left and self._space_held):
            self._panning = True
            self._pan_last = QPointF(event.pos())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif is_left and self._tool is not None:
            cx, cy = self._widget_to_canvas(event.pos().x(), event.pos().y())
            if 0 <= cx < self._sprite.width and 0 <= cy < self._sprite.height:
                self._tool.layer_index = self._active_layer
                self._tool.frame_index = self._active_frame
                self._tool.on_press(cx, cy)
                self.invalidate_cache()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        pos = QPointF(event.pos())
        cx, cy = self._widget_to_canvas(pos.x(), pos.y())
        if 0 <= cx < self._sprite.width and 0 <= cy < self._sprite.height:
            self.cursor_moved.emit(cx, cy)

        if self._panning:
            delta = pos - self._pan_last
            self._pan += delta
            self._pan_last = pos
            self.update()
        elif (event.buttons() & Qt.MouseButton.LeftButton) and self._tool is not None:
            self._tool.on_drag(cx, cy)
            self.invalidate_cache()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._panning and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self._panning = False
            if self._space_held:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.unsetCursor()
        elif event.button() == Qt.MouseButton.LeftButton and self._tool is not None:
            cx, cy = self._widget_to_canvas(event.pos().x(), event.pos().y())
            self._tool.on_release(cx, cy)
            self.invalidate_cache()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        if key == Qt.Key.Key_Space:
            self._space_held = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom_step(1)
        elif key == Qt.Key.Key_Minus:
            self._zoom_step(-1)
        elif key == Qt.Key.Key_0:
            self.zoom = 1.0
            self._pan = QPointF(0.0, 0.0)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space:
            self._space_held = False
            self.unsetCursor()
        else:
            super().keyReleaseEvent(event)
