# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tool palette sidebar and tool-options bar.

:class:`ToolBar` is a vertical QWidget with one :class:`~PyQt6.QtWidgets.QToolButton`
per drawing tool.  Below the buttons a compact options strip exposes
brush size, opacity, and flood-fill tolerance.

Selecting a button emits :attr:`ToolBar.tool_changed` with the tool's
canonical name (e.g. ``"pencil"``).
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# Ordered list of (name, label) for each tool button.
_TOOLS = [
    ("pencil", "✏ Pencil"),
    ("eraser", "⌫ Eraser"),
    ("line", "/ Line"),
    ("rectangle", "□ Rect"),
    ("ellipse", "○ Ellipse"),
    ("fill", "⛽ Fill"),
    ("eyedropper", "🔍 Eyedrop"),
    ("select", "⬚ Select"),
    ("move", "✥ Move"),
    ("text", "T Text"),
]


class ToolBar(QWidget):
    """Vertical tool palette with options strip.

    Signals:
        tool_changed: Emitted when the active tool changes.
            Carries the tool's name string (e.g. ``"pencil"``).
        brush_size_changed: Emitted when brush size changes.
        opacity_changed: Emitted when opacity changes.
        tolerance_changed: Emitted when fill tolerance changes.
    """

    tool_changed = pyqtSignal(str)
    brush_size_changed = pyqtSignal(int)
    opacity_changed = pyqtSignal(int)
    tolerance_changed = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_tool: str = "pencil"
        self._buttons: Dict[str, QToolButton] = {}
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        # Tool buttons
        for name, label in _TOOLS:
            btn = QToolButton(self)
            btn.setText(label)
            btn.setToolTip(name.capitalize())
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda checked, n=name: self._on_tool_clicked(n))
            self._buttons[name] = btn
            self._button_group.addButton(btn)
            root.addWidget(btn)

        # Select pencil by default.
        self._buttons["pencil"].setChecked(True)

        # Separator
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        # Options strip
        root.addWidget(self._make_option_row("Brush", self._make_brush_spin()))
        root.addWidget(self._make_option_row("Opacity", self._make_opacity_slider()))
        root.addWidget(self._make_option_row("Tolerance", self._make_tolerance_spin()))

        root.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_tool(self) -> str:
        """Name of the currently selected tool."""
        return self._current_tool

    def select_tool(self, name: str) -> None:
        """Programmatically select a tool by name.

        Args:
            name: Tool name (e.g. ``"pencil"``).

        Raises:
            KeyError: If *name* is not a recognised tool.
        """
        btn = self._buttons[name]
        btn.setChecked(True)
        self._on_tool_clicked(name)

    @property
    def brush_size(self) -> int:
        """Current brush size in pixels."""
        return self._brush_spin.value()

    @property
    def opacity(self) -> int:
        """Current opacity (0–255)."""
        return self._opacity_slider.value()

    @property
    def tolerance(self) -> int:
        """Current flood-fill tolerance (0–255)."""
        return self._tolerance_spin.value()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_tool_clicked(self, name: str) -> None:
        self._current_tool = name
        self.tool_changed.emit(name)

    @staticmethod
    def _make_option_row(label_text: str, widget: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label_text))
        layout.addWidget(widget, 1)
        return row

    def _make_brush_spin(self) -> QSpinBox:
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(1, 64)
        self._brush_spin.setValue(1)
        self._brush_spin.valueChanged.connect(self.brush_size_changed)
        return self._brush_spin

    def _make_opacity_slider(self) -> QSlider:
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 255)
        self._opacity_slider.setValue(255)
        self._opacity_slider.valueChanged.connect(self.opacity_changed)
        return self._opacity_slider

    def _make_tolerance_spin(self) -> QSpinBox:
        self._tolerance_spin = QSpinBox()
        self._tolerance_spin.setRange(0, 255)
        self._tolerance_spin.setValue(32)
        self._tolerance_spin.valueChanged.connect(self.tolerance_changed)
        return self._tolerance_spin
