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
    │  H: ──────────────  [360]   │
    │  S: ──────────────  [255]   │
    │  V: ──────────────  [255]   │
    │  A: ──────────────  [255]   │
    ├──────────────────────────────┤
    │  R: [###] G: [###] B: [###] │
    │  Hex: [       #RRGGBBAA  ]  │
    ├──────────────────────────────┤
    │  [palette grid …]           │
    └──────────────────────────────┘

Clicking a swatch makes it "active"; the sliders then edit that swatch's
colour.  Emits :attr:`ColorPicker.foreground_changed` or
:attr:`ColorPicker.background_changed` whenever the active colour changes.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
        palette_frame = QFrame()
        palette_frame.setFrameShape(QFrame.Shape.StyledPanel)
        palette_grid = QGridLayout(palette_frame)
        palette_grid.setSpacing(2)
        palette_grid.setContentsMargins(2, 2, 2, 2)
        self._palette_buttons: list = []
        for i, (r, g, b) in enumerate(_DEFAULT_PALETTE):
            btn = QToolButton()
            btn.setFixedSize(20, 20)
            btn.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #555;"
            )
            btn.clicked.connect(
                lambda _, c=(r, g, b, 255): self._apply_palette_color(c)
            )
            palette_grid.addWidget(btn, i // 8, i % 8)
            self._palette_buttons.append(btn)
        root.addWidget(palette_frame)
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
