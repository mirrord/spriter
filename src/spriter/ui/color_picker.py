# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Color picker widget — foreground/background swatches with HSV sliders.

Layout
------
::

    ┌──────────────────────────────┐
    │   [FG swatch] [BG swatch]   │
    ├──────────────────────────────┤
    │  [SV square        ] [H │]  │
    ├──────────────────────────────┤
    │  H: ──────────────  [360]   │
    │  S: ──────────────  [255]   │
    │  V: ──────────────  [255]   │
    │  A: ──────────────  [255]   │
    ├──────────────────────────────┤
    │  R: [###] G: [###] B: [###] │
    │  Hex: [       #RRGGBBAA  ]  │
    ├──────────────────────────────┤
    │  [palette grid …]           │
    ├──────────────────────────────┤
    │  Recent: [■][■][■]…         │
    └──────────────────────────────┘

Clicking a swatch makes it "active"; the sliders then edit that swatch's
colour.  Emits :attr:`ColorPicker.foreground_changed` or
:attr:`ColorPicker.background_changed` whenever the active colour changes.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

Color = Tuple[int, int, int, int]  # RGBA 0-255


class _ColorSwatch(QWidget):
    """A clickable rectangle that shows a solid colour."""

    clicked = pyqtSignal()

    def __init__(self, color: Color, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._color = QColor(*color)
        self.setFixedSize(36, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    @property
    def color(self) -> QColor:
        return self._color

    @color.setter
    def color(self, value: QColor) -> None:
        self._color = value
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        from PyQt6.QtGui import QPainter

        p = QPainter(self)
        p.fillRect(self.rect(), self._color)
        p.setPen(Qt.GlobalColor.black)
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


# ---------------------------------------------------------------------------
# _HueStrip — vertical rainbow strip for selecting hue (H in HSV)
# ---------------------------------------------------------------------------


class _HueStrip(QWidget):
    """A vertical rainbow strip; clicking/dragging picks a hue (0–359)."""

    hue_changed = pyqtSignal(int)

    _W = 16
    _H = 150

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self._W, self._H)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._hue: int = 0
        self._gradient: Optional[QImage] = None

    def set_hue(self, h: int) -> None:
        self._hue = max(0, min(359, h))
        self.update()

    def _ensure_gradient(self) -> QImage:
        if self._gradient is not None:
            return self._gradient
        h = self._H
        # Build RGBA array: each row is a different hue
        hues = np.linspace(0, 359, h, dtype=np.float32)
        # HSV to RGB: S=1, V=1
        h6 = hues / 60.0
        i = np.floor(h6).astype(np.int32) % 6
        f = h6 - np.floor(h6)
        # p=0, q=1-f, t=f  (S=V=1)
        q = 1.0 - f
        t = f
        r = np.where(
            i == 0,
            1.0,
            np.where(
                i == 1,
                q,
                np.where(i == 2, 0.0, np.where(i == 3, 0.0, np.where(i == 4, t, 1.0))),
            ),
        )
        g = np.where(
            i == 0,
            t,
            np.where(
                i == 1,
                1.0,
                np.where(i == 2, 1.0, np.where(i == 3, q, np.where(i == 4, 0.0, 0.0))),
            ),
        )
        b = np.where(
            i == 0,
            0.0,
            np.where(
                i == 1,
                0.0,
                np.where(i == 2, f, np.where(i == 3, 1.0, np.where(i == 4, 1.0, q))),
            ),
        )
        # Shape: (H, W, 4)
        w = self._W
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = (r * 255).astype(np.uint8)[:, np.newaxis]
        rgba[:, :, 1] = (g * 255).astype(np.uint8)[:, np.newaxis]
        rgba[:, :, 2] = (b * 255).astype(np.uint8)[:, np.newaxis]
        rgba[:, :, 3] = 255
        raw = rgba.tobytes()
        img = QImage(raw, w, h, w * 4, QImage.Format.Format_RGBA8888)
        self._gradient = img.copy()  # detach from numpy buffer lifetime
        return self._gradient

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        img = self._ensure_gradient()
        p.drawImage(0, 0, img)
        # Cursor line
        y = int(self._hue / 359.0 * (self._H - 1))
        pen = QPen(Qt.GlobalColor.white, 2)
        p.setPen(pen)
        p.drawLine(0, y, self._W - 1, y)

    def _pick_from_y(self, y: float) -> None:
        h = max(0, min(359, int(y / (self._H - 1) * 359)))
        self._hue = h
        self.update()
        self.hue_changed.emit(h)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._pick_from_y(event.position().y())

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._pick_from_y(event.position().y())


# ---------------------------------------------------------------------------
# _SVSquare — 2-D saturation/value gradient square for current hue
# ---------------------------------------------------------------------------


def _hsv_to_rgb_array(hue: int, size: int) -> np.ndarray:
    """Return a (size, size, 3) uint8 array: S on x-axis, V on y-axis (top=high V)."""
    s_vals = np.linspace(0.0, 1.0, size, dtype=np.float32)  # x
    v_vals = np.linspace(1.0, 0.0, size, dtype=np.float32)  # y (top row = V=1)
    # Shape grids: (size, size)
    s_grid, v_grid = np.meshgrid(s_vals, v_vals)

    h6 = hue / 60.0
    i = int(h6) % 6
    f = h6 - int(h6)

    p = v_grid * (1.0 - s_grid)
    q = v_grid * (1.0 - f * s_grid)
    t = v_grid * (1.0 - (1.0 - f) * s_grid)

    if i == 0:
        r, g, b = v_grid, t, p
    elif i == 1:
        r, g, b = q, v_grid, p
    elif i == 2:
        r, g, b = p, v_grid, t
    elif i == 3:
        r, g, b = p, q, v_grid
    elif i == 4:
        r, g, b = t, p, v_grid
    else:
        r, g, b = v_grid, p, q

    rgb = np.stack(
        [
            (r * 255).astype(np.uint8),
            (g * 255).astype(np.uint8),
            (b * 255).astype(np.uint8),
        ],
        axis=2,
    )
    return rgb


class _SVSquare(QWidget):
    """Saturation/Value 2-D gradient square for the current hue."""

    sv_changed = pyqtSignal(int, int)  # saturation, value (0–255 each)

    _SIZE = 150

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self._SIZE, self._SIZE)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._hue: int = 0
        self._s: int = 255
        self._v: int = 255
        self._cached_hue: Optional[int] = None
        self._cached_img: Optional[QImage] = None

    def set_hue(self, h: int) -> None:
        self._hue = max(0, min(359, h))
        if self._cached_hue != self._hue:
            self._cached_img = None  # invalidate
        self.update()

    def set_sv(self, s: int, v: int) -> None:
        self._s = max(0, min(255, s))
        self._v = max(0, min(255, v))
        self.update()

    def _ensure_image(self) -> QImage:
        if self._cached_img is not None and self._cached_hue == self._hue:
            return self._cached_img
        sz = self._SIZE
        rgb = _hsv_to_rgb_array(self._hue, sz)
        # Add alpha channel
        alpha = np.full((sz, sz, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=2)
        raw = rgba.tobytes()
        img = QImage(raw, sz, sz, sz * 4, QImage.Format.Format_RGBA8888)
        self._cached_img = img.copy()
        self._cached_hue = self._hue
        return self._cached_img

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.drawImage(0, 0, self._ensure_image())
        # Crosshair at current S/V position
        sz = self._SIZE
        cx = int(self._s / 255.0 * (sz - 1))
        cy = int((255 - self._v) / 255.0 * (sz - 1))
        p.setPen(QPen(Qt.GlobalColor.white, 1))
        p.drawLine(cx - 4, cy, cx + 4, cy)
        p.drawLine(cx, cy - 4, cx, cy + 4)
        p.setPen(QPen(Qt.GlobalColor.black, 1))
        p.drawRect(cx - 4, cy - 4, 8, 8)

    def _pick_from_pos(self, x: float, y: float) -> None:
        sz = self._SIZE
        s = max(0, min(255, int(x / (sz - 1) * 255)))
        v = max(0, min(255, int((1.0 - y / (sz - 1)) * 255)))
        self._s = s
        self._v = v
        self.update()
        self.sv_changed.emit(s, v)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._pick_from_pos(pos.x(), pos.y())

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.buttons() & Qt.MouseButton.LeftButton:
            pos = event.position()
            self._pick_from_pos(pos.x(), pos.y())


# ---------------------------------------------------------------------------
# _PaletteButton — QToolButton with right-click context menu for palette slots
# ---------------------------------------------------------------------------


class _PaletteButton(QToolButton):
    """A palette slot button with right-click edit/delete actions."""

    set_color_requested = pyqtSignal(int)  # slot index
    delete_requested = pyqtSignal(int)  # slot index

    def __init__(
        self,
        index: int,
        color: Color,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.index = index
        self._color = color
        self.setFixedSize(20, 20)
        r, g, b = color[0], color[1], color[2]
        self.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #555;"
        )
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _build_context_menu(self) -> QMenu:
        menu = QMenu(self)
        set_act = QAction("Set to Foreground Color", menu)
        set_act.triggered.connect(lambda: self.set_color_requested.emit(self.index))
        menu.addAction(set_act)
        del_act = QAction("Delete Slot", menu)
        del_act.triggered.connect(lambda: self.delete_requested.emit(self.index))
        menu.addAction(del_act)
        return menu

    def _show_context_menu(self, pos: QPoint) -> None:
        menu = self._build_context_menu()
        menu.exec(self.mapToGlobal(pos))


class ColorPicker(QWidget):
    """Foreground / background colour selector with HSV sliders and hex input.

    Args:
        parent: Optional Qt parent widget.

    Signals:
        foreground_changed: Emitted when the foreground (primary) colour changes.
            Carries an ``(r, g, b, a)`` tuple.
        background_changed: Emitted when the background (secondary) colour changes.
            Carries an ``(r, g, b, a)`` tuple.
    """

    foreground_changed = pyqtSignal(object)  # tuple (r, g, b, a)
    background_changed = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._fg_color = QColor(0, 0, 0, 255)
        self._bg_color = QColor(255, 255, 255, 255)
        self._editing_fg = True  # True = sliders edit FG, False = BG
        self._updating = False  # guard against signal loops

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Swatches ────────────────────────────────────────────────
        swatch_row = QHBoxLayout()
        self._bg_swatch = _ColorSwatch((255, 255, 255, 255))
        self._fg_swatch = _ColorSwatch((0, 0, 0, 255))
        swatch_row.addWidget(self._fg_swatch)
        swatch_row.addWidget(self._bg_swatch)
        swatch_row.addStretch()
        root.addLayout(swatch_row)

        self._fg_swatch.clicked.connect(lambda: self._set_editing_fg(True))
        self._bg_swatch.clicked.connect(lambda: self._set_editing_fg(False))

        # ── Gradient picker: SV square + H strip (Aseprite-style) ────
        self._sv_square = _SVSquare()
        self._hue_strip = _HueStrip()
        gradient_row = QHBoxLayout()
        gradient_row.setSpacing(4)
        gradient_row.addWidget(self._sv_square)
        gradient_row.addWidget(self._hue_strip)
        gradient_row.addStretch()
        root.addLayout(gradient_row)

        # ── HSV + Alpha sliders ──────────────────────────────────────
        slider_frame = QFrame()
        slider_frame.setFrameShape(QFrame.Shape.StyledPanel)
        slider_grid = QGridLayout(slider_frame)
        slider_grid.setContentsMargins(4, 4, 4, 4)
        slider_grid.setSpacing(2)

        self._h_slider, self._h_spin = self._make_slider_row(
            slider_grid, 0, "H", 0, 359
        )
        self._s_slider, self._s_spin = self._make_slider_row(
            slider_grid, 1, "S", 0, 255
        )
        self._v_slider, self._v_spin = self._make_slider_row(
            slider_grid, 2, "V", 0, 255
        )
        self._a_slider, self._a_spin = self._make_slider_row(
            slider_grid, 3, "A", 0, 255
        )
        root.addWidget(slider_frame)

        # ── RGB + Hex inputs ─────────────────────────────────────────
        rgb_row = QHBoxLayout()
        self._r_spin = self._make_channel_spin("R", rgb_row)
        self._g_spin = self._make_channel_spin("G", rgb_row)
        self._b_spin = self._make_channel_spin("B", rgb_row)
        root.addLayout(rgb_row)

        hex_row = QHBoxLayout()
        hex_row.addWidget(QLabel("Hex:"))
        self._hex_edit = QLineEdit()
        self._hex_edit.setMaxLength(9)
        self._hex_edit.setPlaceholderText("#RRGGBBAA")
        hex_row.addWidget(self._hex_edit)
        root.addLayout(hex_row)

        # ── Palette grid (first 16 standard web colours) ─────────────
        self._palette_frame = QFrame()
        self._palette_frame.setFrameShape(QFrame.Shape.StyledPanel)
        palette_grid = QGridLayout(self._palette_frame)
        palette_grid.setSpacing(2)
        palette_grid.setContentsMargins(2, 2, 2, 2)
        self._palette_buttons: List[_PaletteButton] = []
        self._palette_colors: list = [(r, g, b, 255) for r, g, b in _DEFAULT_PALETTE]
        self._rebuild_palette_grid(self._palette_colors)
        root.addWidget(self._palette_frame)

        # ── Recent colours row ───────────────────────────────────────
        self._recent_colors: List[Color] = []
        recent_frame = QFrame()
        recent_frame.setFrameShape(QFrame.Shape.StyledPanel)
        recent_row = QHBoxLayout(recent_frame)
        recent_row.setContentsMargins(2, 2, 2, 2)
        recent_row.setSpacing(2)
        recent_row.addWidget(QLabel("Recent:"))
        self._recent_buttons: List[QToolButton] = []
        for _ in range(16):
            btn = QToolButton()
            btn.setFixedSize(16, 16)
            btn.setStyleSheet("background-color: #333; border: 1px solid #555;")
            btn.setEnabled(False)
            recent_row.addWidget(btn)
            self._recent_buttons.append(btn)
        recent_row.addStretch()
        root.addWidget(recent_frame)

        root.addStretch()

        # Connect slider/spin changes
        for widget in (
            self._h_slider,
            self._s_slider,
            self._v_slider,
            self._a_slider,
            self._h_spin,
            self._s_spin,
            self._v_spin,
            self._a_spin,
        ):
            widget.valueChanged.connect(self._on_hsva_changed)

        for spin in (self._r_spin, self._g_spin, self._b_spin):
            spin.valueChanged.connect(self._on_rgb_changed)

        self._hex_edit.editingFinished.connect(self._on_hex_edited)

        # Connect gradient picker signals
        self._hue_strip.hue_changed.connect(self._on_hue_strip_changed)
        self._sv_square.sv_changed.connect(self._on_sv_square_changed)

        # Initialise display to FG (black).
        self._refresh_controls()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def foreground(self) -> Color:
        """Foreground colour as an ``(r, g, b, a)`` tuple."""
        return (
            self._fg_color.red(),
            self._fg_color.green(),
            self._fg_color.blue(),
            self._fg_color.alpha(),
        )

    @foreground.setter
    def foreground(self, color: Color) -> None:
        self._fg_color = QColor(*color)
        self._fg_swatch.color = self._fg_color
        if self._editing_fg:
            self._refresh_controls()
        self.foreground_changed.emit(color)
        self.push_recent(color)

    @property
    def background(self) -> Color:
        """Background colour as an ``(r, g, b, a)`` tuple."""
        return (
            self._bg_color.red(),
            self._bg_color.green(),
            self._bg_color.blue(),
            self._bg_color.alpha(),
        )

    @background.setter
    def background(self, color: Color) -> None:
        self._bg_color = QColor(*color)
        self._bg_swatch.color = self._bg_color
        if not self._editing_fg:
            self._refresh_controls()
        self.background_changed.emit(color)

    def swap_colors(self) -> None:
        """Swap the foreground and background colours."""
        self._fg_color, self._bg_color = self._bg_color, self._fg_color
        self._fg_swatch.color = self._fg_color
        self._bg_swatch.color = self._bg_color
        fg = (
            self._fg_color.red(),
            self._fg_color.green(),
            self._fg_color.blue(),
            self._fg_color.alpha(),
        )
        bg = (
            self._bg_color.red(),
            self._bg_color.green(),
            self._bg_color.blue(),
            self._bg_color.alpha(),
        )
        self.foreground_changed.emit(fg)
        self.background_changed.emit(bg)
        self._refresh_controls()

    def load_palette(self, colors: list) -> None:
        """Replace the palette grid with the given colour list.

        Args:
            colors: List of ``(R, G, B)`` or ``(R, G, B, A)`` tuples.
        """
        self._palette_colors = [
            (c[0], c[1], c[2], c[3] if len(c) > 3 else 255) for c in colors
        ]
        self._rebuild_palette_grid(self._palette_colors)

    def push_recent(self, color: Color) -> None:
        """Prepend *color* to the recent-colours row (max 16, newest first).

        Duplicate entries are removed before insertion.

        Args:
            color: ``(R, G, B, A)`` tuple.
        """
        normalized: Color = (
            color[0],
            color[1],
            color[2],
            color[3] if len(color) > 3 else 255,
        )
        # Remove existing duplicate
        self._recent_colors = [c for c in self._recent_colors if c != normalized]
        self._recent_colors.insert(0, normalized)
        self._recent_colors = self._recent_colors[:16]
        self._refresh_recent_buttons()

    def _refresh_recent_buttons(self) -> None:
        for i, btn in enumerate(self._recent_buttons):
            if i < len(self._recent_colors):
                color = self._recent_colors[i]
                r, g, b = color[0], color[1], color[2]
                btn.setStyleSheet(
                    f"background-color: rgb({r},{g},{b}); border: 1px solid #555;"
                )
                btn.setEnabled(True)
                # Disconnect old connections before reconnecting
                try:
                    btn.clicked.disconnect()
                except (RuntimeError, TypeError):
                    pass
                btn.clicked.connect(lambda _, c=color: self._apply_palette_color(c))
            else:
                btn.setStyleSheet("background-color: #333; border: 1px solid #555;")
                btn.setEnabled(False)
                try:
                    btn.clicked.disconnect()
                except (RuntimeError, TypeError):
                    pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _active_color(self) -> QColor:
        return self._fg_color if self._editing_fg else self._bg_color

    def _set_editing_fg(self, fg: bool) -> None:
        self._editing_fg = fg
        # Highlight the active swatch
        self._fg_swatch.setStyleSheet("border: 2px solid white;" if fg else "")
        self._bg_swatch.setStyleSheet("border: 2px solid white;" if not fg else "")
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        """Push the active colour values into all controls."""
        self._updating = True
        try:
            c = self._active_color()
            h, s, v, a = c.hsvHue(), c.hsvSaturation(), c.value(), c.alpha()
            h = max(0, h)  # QColor returns -1 for achromatic

            self._h_slider.setValue(h)
            self._h_spin.setValue(h)
            self._s_slider.setValue(s)
            self._s_spin.setValue(s)
            self._v_slider.setValue(v)
            self._v_spin.setValue(v)
            self._a_slider.setValue(a)
            self._a_spin.setValue(a)

            self._r_spin.setValue(c.red())
            self._g_spin.setValue(c.green())
            self._b_spin.setValue(c.blue())

            self._hex_edit.setText(
                f"#{c.red():02x}{c.green():02x}{c.blue():02x}{c.alpha():02x}"
            )
            # Sync gradient widgets
            self._sv_square.set_hue(h)
            self._sv_square.set_sv(s, v)
            self._hue_strip.set_hue(h)
        finally:
            self._updating = False

    def _apply_color(self, color: QColor) -> None:
        if self._editing_fg:
            self._fg_color = color
            self._fg_swatch.color = color
            self.foreground_changed.emit(
                (color.red(), color.green(), color.blue(), color.alpha())
            )
        else:
            self._bg_color = color
            self._bg_swatch.color = color
            self.background_changed.emit(
                (color.red(), color.green(), color.blue(), color.alpha())
            )

    def _on_hsva_changed(self, _value: int) -> None:
        if self._updating:
            return
        h = self._h_spin.value()
        s = self._s_spin.value()
        v = self._v_spin.value()
        a = self._a_spin.value()
        color = QColor.fromHsv(h, s, v, a)
        self._updating = True
        try:
            self._r_spin.setValue(color.red())
            self._g_spin.setValue(color.green())
            self._b_spin.setValue(color.blue())
            self._hex_edit.setText(
                f"#{color.red():02x}{color.green():02x}{color.blue():02x}{color.alpha():02x}"
            )
            # Sync partner sliders/spins
            for slider, spin in zip(
                (self._h_slider, self._s_slider, self._v_slider, self._a_slider),
                (self._h_spin, self._s_spin, self._v_spin, self._a_spin),
            ):
                slider.blockSignals(True)
                spin.blockSignals(True)
            self._h_slider.setValue(h)
            self._s_slider.setValue(s)
            self._v_slider.setValue(v)
            self._a_slider.setValue(a)
            for slider, spin in zip(
                (self._h_slider, self._s_slider, self._v_slider, self._a_slider),
                (self._h_spin, self._s_spin, self._v_spin, self._a_spin),
            ):
                slider.blockSignals(False)
                spin.blockSignals(False)
        finally:
            self._updating = False
        self._apply_color(color)

    def _on_rgb_changed(self, _value: int) -> None:
        if self._updating:
            return
        color = QColor(
            self._r_spin.value(),
            self._g_spin.value(),
            self._b_spin.value(),
            self._a_spin.value(),
        )
        self._updating = True
        try:
            h = max(0, color.hsvHue())
            self._h_slider.setValue(h)
            self._h_spin.setValue(h)
            self._s_slider.setValue(color.hsvSaturation())
            self._s_spin.setValue(color.hsvSaturation())
            self._v_slider.setValue(color.value())
            self._v_spin.setValue(color.value())
            self._hex_edit.setText(
                f"#{color.red():02x}{color.green():02x}{color.blue():02x}{color.alpha():02x}"
            )
        finally:
            self._updating = False
        self._apply_color(color)

    def _on_hex_edited(self) -> None:
        text = self._hex_edit.text().strip().lstrip("#")
        match = re.fullmatch(r"([0-9a-fA-F]{6})([0-9a-fA-F]{2})?", text)
        if not match:
            return
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
        a = int(text[6:8], 16) if len(text) >= 8 else 255
        color = QColor(r, g, b, a)
        self._updating = True
        try:
            self._r_spin.setValue(r)
            self._g_spin.setValue(g)
            self._b_spin.setValue(b)
            self._a_spin.setValue(a)
            h = max(0, color.hsvHue())
            self._h_slider.setValue(h)
            self._h_spin.setValue(h)
            self._s_slider.setValue(color.hsvSaturation())
            self._s_spin.setValue(color.hsvSaturation())
            self._v_slider.setValue(color.value())
            self._v_spin.setValue(color.value())
        finally:
            self._updating = False
        self._apply_color(color)
        self.push_recent((color.red(), color.green(), color.blue(), color.alpha()))

    def _on_hue_strip_changed(self, hue: int) -> None:
        """Hue strip was clicked/dragged — update the H spin (which triggers _on_hsva_changed)."""
        if not self._updating:
            self._h_spin.setValue(hue)

    def _on_sv_square_changed(self, s: int, v: int) -> None:
        """SV square was clicked/dragged — update S/V spins."""
        if not self._updating:
            self._s_spin.setValue(s)
            self._v_spin.setValue(v)

    def _rebuild_palette_grid(self, colors: list) -> None:
        """Clear and repopulate the palette button grid."""
        layout = self._palette_frame.layout()
        for btn in self._palette_buttons:
            layout.removeWidget(btn)
            btn.deleteLater()
        self._palette_buttons.clear()
        for i, color in enumerate(colors):
            r, g, b = color[0], color[1], color[2]
            a = color[3] if len(color) > 3 else 255
            c: Color = (r, g, b, a)
            btn = _PaletteButton(i, c)
            btn.clicked.connect(lambda _, col=c: self._apply_palette_color(col))
            btn.set_color_requested.connect(self._on_palette_set_color)
            btn.delete_requested.connect(self._on_palette_delete)
            layout.addWidget(btn, i // 8, i % 8)
            self._palette_buttons.append(btn)

    def _apply_palette_color(self, color: Color) -> None:
        qc = QColor(*color)
        self._updating = True
        try:
            self._a_spin.setValue(color[3])
            self._a_slider.setValue(color[3])
        finally:
            self._updating = False
        # Apply via RGB setter to update all controls
        self._r_spin.setValue(qc.red())
        self._g_spin.setValue(qc.green())
        self._b_spin.setValue(qc.blue())
        self.push_recent(color)

    def _on_palette_set_color(self, index: int) -> None:
        """Set palette slot *index* to the current foreground colour."""
        if 0 <= index < len(self._palette_colors):
            self._palette_colors[index] = self.foreground
            self._rebuild_palette_grid(self._palette_colors)

    def _on_palette_delete(self, index: int) -> None:
        """Remove palette slot *index*."""
        if 0 <= index < len(self._palette_colors):
            del self._palette_colors[index]
            self._rebuild_palette_grid(self._palette_colors)

    # ------------------------------------------------------------------
    # Widget creation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_slider_row(
        grid: QGridLayout, row: int, label: str, lo: int, hi: int
    ) -> tuple:
        grid.addWidget(QLabel(label), row, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setFixedWidth(52)
        # Keep slider and spin in sync without recursion
        slider.valueChanged.connect(
            lambda v, s=spin: (
                s.blockSignals(True),
                s.setValue(v),
                s.blockSignals(False),
            )
        )
        spin.valueChanged.connect(
            lambda v, sl=slider: (
                sl.blockSignals(True),
                sl.setValue(v),
                sl.blockSignals(False),
            )
        )
        grid.addWidget(slider, row, 1)
        grid.addWidget(spin, row, 2)
        return slider, spin

    @staticmethod
    def _make_channel_spin(label: str, layout: QHBoxLayout) -> QSpinBox:
        layout.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(0, 255)
        spin.setFixedWidth(52)
        layout.addWidget(spin)
        return spin


# A small default palette (16 swatches).
_DEFAULT_PALETTE = [
    (0, 0, 0),
    (255, 255, 255),
    (128, 128, 128),
    (192, 192, 192),
    (255, 0, 0),
    (128, 0, 0),
    (255, 128, 0),
    (128, 64, 0),
    (255, 255, 0),
    (128, 128, 0),
    (0, 255, 0),
    (0, 128, 0),
    (0, 255, 255),
    (0, 128, 128),
    (0, 0, 255),
    (0, 0, 128),
]
