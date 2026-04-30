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
from PyQt6.QtCore import QLine, QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QWheelEvent
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
    # Emitted when a tool samples a color (e.g. eyedropper); carries (r, g, b, a).
    color_sampled = pyqtSignal(object)

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

        # Onion skinning — show N previous / next frames as translucent overlays.
        self.onion_before: int = 0  # frames to show before active
        self.onion_after: int = 0  # frames to show after active
        self.onion_opacity: float = 0.3  # base opacity for ghost frames

        # Symmetry / mirror drawing mode.
        self.symmetry_h: bool = False  # mirror horizontally (left ↔ right)
        self.symmetry_v: bool = False  # mirror vertically   (top ↔ bottom)

        # Reference image overlay.
        self.reference_image: Optional[np.ndarray] = None  # RGBA H×W×4
        self.reference_opacity: float = 0.5

        # Tiling preview — renders the canvas in a 3×3 tile grid.
        self.tiling_preview: bool = False

        # Marching-ants animation — increments every 100 ms.
        self._sel_anim_offset: int = 0
        self._sel_timer = QTimer(self)
        self._sel_timer.setInterval(100)
        self._sel_timer.timeout.connect(self._tick_selection_anim)
        self._sel_timer.start()

        # Coalesce drag repaints — many mouseMove events per frame collapse
        # into a single paint, capped to ~60 fps.
        self._repaint_timer = QTimer(self)
        self._repaint_timer.setInterval(16)
        self._repaint_timer.setSingleShot(True)
        self._repaint_timer.timeout.connect(self.update)

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

    def _request_paint(self) -> None:
        """Schedule a repaint on the next vsync-ish tick (coalesces drag events)."""
        if not self._repaint_timer.isActive():
            self._repaint_timer.start()

    def fit_to_window(self) -> None:
        """Set zoom and pan so the canvas fills the widget as closely as possible."""
        if self._sprite.width == 0 or self._sprite.height == 0:
            return
        w_ratio = self.width() / self._sprite.width
        h_ratio = self.height() / self._sprite.height
        self.zoom = min(w_ratio, h_ratio)
        self._pan = QPointF(0.0, 0.0)

    def center_view(self) -> None:
        """Reset pan so the canvas is centered in the widget (zoom unchanged)."""
        self._pan = QPointF(0.0, 0.0)
        self.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_composite(self) -> np.ndarray:
        if self._composite_cache is None:
            if self._sprite.frame_count > 0 and self._sprite.layer_count > 0:
                from ..core.compositor import composite_frame

                frame_idx = min(self._active_frame, self._sprite.frame_count - 1)
                self._composite_cache = composite_frame(self._sprite, frame_idx)
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
        if self._zoom == 1.0 and scaled_w == w and scaled_h == h:
            painter.drawImage(offset, image)
        else:
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

        # Onion-skin overlays (previous / next frames).
        self._paint_onion_skin(painter, offset, scaled_w, scaled_h)

        # Reference image overlay.
        self._paint_reference_image(painter, offset, scaled_w, scaled_h)

        # Tool preview overlay.
        if self._tool is not None:
            overlay = self._tool.preview_overlay()
            if overlay is not None:
                ov_arr = np.ascontiguousarray(overlay)
                ov_image = QImage(
                    ov_arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888
                )
                painter.setOpacity(0.75)
                if self._zoom == 1.0 and scaled_w == w and scaled_h == h:
                    painter.drawImage(offset, ov_image)
                else:
                    scaled_ov = ov_image.scaled(
                        scaled_w,
                        scaled_h,
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.FastTransformation,
                    )
                    painter.drawImage(offset, scaled_ov)
                painter.setOpacity(1.0)

        # Tiling preview — 3×3 repeated copies surrounding the main canvas.
        if self.tiling_preview:
            self._paint_tiling_preview(painter, offset, scaled_w, scaled_h)

        # Selection overlay — committed mask and in-progress drag rectangle.
        self._paint_selection(painter, offset)
        self._paint_selection_preview(painter, offset)

        painter.end()

    def _paint_checkerboard(
        self, painter: QPainter, offset: QPointF, w: int, h: int
    ) -> None:
        check = max(4, int(self._zoom * 2))
        pixmap = self._get_checker_pixmap(check)
        # Align the brush so its origin matches the canvas top-left.
        ox, oy = int(offset.x()), int(offset.y())
        brush = QBrush(pixmap)
        from PyQt6.QtGui import QTransform

        brush.setTransform(QTransform().translate(ox, oy))
        painter.fillRect(ox, oy, w, h, brush)

    def _get_checker_pixmap(self, cell: int) -> QPixmap:
        """Return a 2-cell × 2-cell tiling checkerboard pixmap (cached by cell size)."""
        cache = getattr(self, "_checker_cache", None)
        if cache is None:
            cache = self._checker_cache = {}
        pm = cache.get(cell)
        if pm is not None:
            return pm
        size = cell * 2
        pm = QPixmap(size, size)
        light = QColor(200, 200, 200)
        dark = QColor(150, 150, 150)
        p = QPainter(pm)
        p.fillRect(0, 0, cell, cell, light)
        p.fillRect(cell, cell, cell, cell, light)
        p.fillRect(cell, 0, cell, cell, dark)
        p.fillRect(0, cell, cell, cell, dark)
        p.end()
        cache[cell] = pm
        return pm

    def _paint_onion_skin(
        self, painter: QPainter, offset: QPointF, scaled_w: int, scaled_h: int
    ) -> None:
        """Render previous / next frames as translucent ghost overlays."""
        total = self._sprite.frame_count
        if total <= 1 or (self.onion_before == 0 and self.onion_after == 0):
            return
        from ..core.compositor import composite_frame

        ghosts = []
        for step in range(1, self.onion_before + 1):
            fi = self._active_frame - step
            if 0 <= fi < total:
                ghosts.append((fi, self.onion_opacity * (1.0 - 0.15 * (step - 1))))
        for step in range(1, self.onion_after + 1):
            fi = self._active_frame + step
            if 0 <= fi < total:
                ghosts.append((fi, self.onion_opacity * (1.0 - 0.15 * (step - 1))))

        for fi, opacity in ghosts:
            ghost = composite_frame(self._sprite, fi)
            arr = np.ascontiguousarray(ghost)
            gh, gw = arr.shape[:2]
            gi = QImage(arr.data, gw, gh, gw * 4, QImage.Format.Format_RGBA8888)
            scaled_gi = gi.scaled(
                scaled_w,
                scaled_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            painter.setOpacity(opacity)
            painter.drawImage(offset, scaled_gi)
            painter.setOpacity(1.0)

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
        delta = event.angleDelta().y()
        direction = 1 if delta > 0 else -1 if delta < 0 else 0
        if direction != 0:
            self._zoom_step_at(direction, QPointF(event.position()))
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

    def _zoom_step_at(self, direction: int, anchor: QPointF) -> None:
        """Step zoom through ZOOM_LEVELS, keeping *anchor* (widget coords) fixed.

        Args:
            direction: +1 to zoom in, -1 to zoom out.
            anchor: Widget-space point that should remain stationary after zoom.
        """
        levels = self.ZOOM_LEVELS
        old_zoom = self._zoom
        try:
            idx = levels.index(int(old_zoom))
        except ValueError:
            diffs = [abs(lv - old_zoom) for lv in levels]
            idx = diffs.index(min(diffs))
        new_idx = max(0, min(len(levels) - 1, idx + direction))
        new_zoom = float(levels[new_idx])
        if new_zoom == old_zoom:
            return
        # Canvas coordinates of the point under the anchor before zoom.
        offset = self._canvas_offset()
        cx = (anchor.x() - offset.x()) / old_zoom
        cy = (anchor.y() - offset.y()) / old_zoom
        # Adjust pan so that same canvas point ends up under anchor.
        cw = self._sprite.width * new_zoom
        ch = self._sprite.height * new_zoom
        self._pan = QPointF(
            anchor.x() - cx * new_zoom - (self.width() - cw) / 2.0,
            anchor.y() - cy * new_zoom - (self.height() - ch) / 2.0,
        )
        self._zoom = new_zoom
        self.zoom_changed.emit(self._zoom)
        self.update()

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
                _prev_fg = self._tool.foreground
                self._tool.on_press(cx, cy)
                if self._tool.foreground != _prev_fg:
                    self.color_sampled.emit(self._tool.foreground)
                for mx, my in self._mirror_point(cx, cy)[1:]:
                    if 0 <= mx < self._sprite.width and 0 <= my < self._sprite.height:
                        self._tool.on_press(mx, my)
                self.invalidate_cache()
        elif event.button() == Qt.MouseButton.RightButton:
            # Right-click cancels any active selection.
            if self._sprite.selection_mask is not None or (
                self._tool is not None
                and self._tool.selection_preview_rect() is not None
            ):
                self._sprite.clear_selection()
                if self._tool is not None:
                    self._tool.cancel()
                self.update()

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
            _prev_fg = self._tool.foreground
            self._tool.on_drag(cx, cy)
            if self._tool.foreground != _prev_fg:
                self.color_sampled.emit(self._tool.foreground)
            for mx, my in self._mirror_point(cx, cy)[1:]:
                if 0 <= mx < self._sprite.width and 0 <= my < self._sprite.height:
                    self._tool.on_drag(mx, my)
            # No tool commits during on_drag — committed cel pixels are
            # unchanged.  Just request a coalesced repaint so the live
            # preview overlay refreshes.
            self._request_paint()

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
        elif key == Qt.Key.Key_Escape:
            # ESC cancels any active selection.
            if self._sprite.selection_mask is not None or (
                self._tool is not None
                and self._tool.selection_preview_rect() is not None
            ):
                self._sprite.clear_selection()
                if self._tool is not None:
                    self._tool.cancel()
                self.update()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space:
            self._space_held = False
            self.unsetCursor()
        else:
            super().keyReleaseEvent(event)

    # ------------------------------------------------------------------
    # Symmetry helper
    # ------------------------------------------------------------------

    def _mirror_point(self, cx: int, cy: int):
        """Return extra canvas positions to paint at for the active symmetry axes.

        Returns a list that always contains ``(cx, cy)`` plus any mirrored
        positions.  Callers should paint at every returned coordinate.
        """
        points = [(cx, cy)]
        mw = self._sprite.width - 1
        mh = self._sprite.height - 1
        if self.symmetry_h:
            points.append((mw - cx, cy))
        if self.symmetry_v:
            points.append((cx, mh - cy))
        if self.symmetry_h and self.symmetry_v:
            points.append((mw - cx, mh - cy))
        return points

    # ------------------------------------------------------------------
    # Phase 8 paint helpers
    # ------------------------------------------------------------------

    def _paint_reference_image(
        self,
        painter: QPainter,
        offset: QPointF,
        scaled_w: int,
        scaled_h: int,
    ) -> None:
        """Draw the reference overlay image at the configured opacity."""
        if self.reference_image is None:
            return
        ref = self.reference_image
        rh, rw = ref.shape[:2]
        ref_arr = np.ascontiguousarray(ref)
        ref_qi = QImage(ref_arr.data, rw, rh, rw * 4, QImage.Format.Format_RGBA8888)
        # Scale reference to match the canvas display size.
        scaled_ref = ref_qi.scaled(
            scaled_w,
            scaled_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.setOpacity(self.reference_opacity)
        painter.drawImage(offset, scaled_ref)
        painter.setOpacity(1.0)

    def _tick_selection_anim(self) -> None:
        """Advance the marching-ants dash offset and repaint if a selection exists."""
        has_sel = self._sprite.selection_mask is not None
        has_preview = (
            self._tool is not None and self._tool.selection_preview_rect() is not None
        )
        if has_sel or has_preview:
            self._sel_anim_offset = (self._sel_anim_offset + 1) % 8
            self.update()

    def _paint_selection(
        self,
        painter: QPainter,
        offset: QPointF,
    ) -> None:
        """Draw a marching-ants border around the committed selection mask."""
        mask = self._sprite.selection_mask
        if mask is None:
            return
        h, w = mask.shape
        z = self._zoom
        ox, oy = offset.x(), offset.y()
        anim = self._sel_anim_offset

        # Detect border edges using padded mask comparisons.
        padded = np.zeros((h + 2, w + 2), dtype=bool)
        padded[1 : h + 1, 1 : w + 1] = mask

        edge_top_r, edge_top_c = np.where(
            padded[1 : h + 1, 1 : w + 1] & ~padded[0:h, 1 : w + 1]
        )
        edge_bot_r, edge_bot_c = np.where(
            padded[1 : h + 1, 1 : w + 1] & ~padded[2 : h + 2, 1 : w + 1]
        )
        edge_left_r, edge_left_c = np.where(
            padded[1 : h + 1, 1 : w + 1] & ~padded[1 : h + 1, 0:w]
        )
        edge_right_r, edge_right_c = np.where(
            padded[1 : h + 1, 1 : w + 1] & ~padded[1 : h + 1, 2 : w + 2]
        )

        white_lines: list[QLine] = []
        black_lines: list[QLine] = []

        def _add_h(rows, cols, y_offset: int) -> None:
            for r, c in zip(rows, cols):
                color_idx = (r + c + anim) % 2
                x1 = int(ox + c * z)
                y = int(oy + r * z) + y_offset
                x2 = int(ox + (c + 1) * z) - 1
                (white_lines if color_idx == 0 else black_lines).append(
                    QLine(x1, y, x2, y)
                )

        def _add_v(rows, cols, x_offset: int) -> None:
            for r, c in zip(rows, cols):
                color_idx = (r + c + anim) % 2
                x = int(ox + c * z) + x_offset
                y1 = int(oy + r * z)
                y2 = int(oy + (r + 1) * z) - 1
                (white_lines if color_idx == 0 else black_lines).append(
                    QLine(x, y1, x, y2)
                )

        _add_h(edge_top_r, edge_top_c, 0)
        _add_h(edge_bot_r, edge_bot_c, int(z) - 1)
        _add_v(edge_left_r, edge_left_c, 0)
        _add_v(edge_right_r, edge_right_c, int(z) - 1)

        painter.setPen(QPen(QColor(255, 255, 255)))
        if white_lines:
            painter.drawLines(white_lines)
        painter.setPen(QPen(QColor(0, 0, 0)))
        if black_lines:
            painter.drawLines(black_lines)

    def _paint_selection_preview(
        self,
        painter: QPainter,
        offset: QPointF,
    ) -> None:
        """Draw the in-progress selection rectangle while the user is dragging."""
        if self._tool is None:
            return
        rect = self._tool.selection_preview_rect()
        if rect is None:
            return
        x0, y0, x1, y1 = rect
        z = self._zoom
        ox, oy = offset.x(), offset.y()
        px0 = int(ox + x0 * z)
        py0 = int(oy + y0 * z)
        px1 = int(ox + (x1 + 1) * z)
        py1 = int(oy + (y1 + 1) * z)
        anim = self._sel_anim_offset
        for color, d_off in (
            (QColor(255, 255, 255), anim),
            (QColor(0, 0, 0), anim + 4),
        ):
            pen = QPen(color)
            pen.setWidth(1)
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([4.0, 4.0])
            pen.setDashOffset(float(d_off))
            painter.setPen(pen)
            painter.drawRect(px0, py0, px1 - px0, py1 - py0)

    def _paint_tiling_preview(
        self,
        painter: QPainter,
        offset: QPointF,
        scaled_w: int,
        scaled_h: int,
    ) -> None:
        """Draw 8 surrounding tile copies of the current frame."""
        composite = self._get_composite()
        h, w = composite.shape[:2]
        arr = np.ascontiguousarray(composite)
        image = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        tile = image.scaled(
            scaled_w,
            scaled_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        painter.setOpacity(0.5)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue  # skip centre — already painted
                tile_offset = QPointF(
                    offset.x() + dc * scaled_w,
                    offset.y() + dr * scaled_h,
                )
                painter.drawImage(tile_offset, tile)
        painter.setOpacity(1.0)
