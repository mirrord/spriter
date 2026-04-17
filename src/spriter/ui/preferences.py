# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Preferences dialog (Phase 8).

A tabbed :class:`PreferencesDialog` exposes all :class:`~spriter.core.settings.Settings`
fields to the user.  Changes are applied to the passed-in :class:`Settings`
object when the user clicks **OK**.

Tabs
----
* **Canvas** — default size, grid colour, checker colours
* **Editor** — undo depth, autosave interval, theme
* **Shortcuts** — per-tool key bindings

Usage::

    from spriter.core.settings import Settings
    from spriter.ui.preferences import PreferencesDialog

    settings = Settings.load()
    dlg = PreferencesDialog(settings, parent=window)
    if dlg.exec():
        settings.save()
"""

from __future__ import annotations

from typing import Dict

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.settings import Settings


class PreferencesDialog(QDialog):
    """Multi-tab preferences dialog that edits a :class:`Settings` instance.

    Args:
        settings: The settings object to read defaults from and write to on accept.
        parent: Optional Qt parent.
    """

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(400)
        self._settings = settings

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_canvas_tab(), "Canvas")
        self._tabs.addTab(self._build_editor_tab(), "Editor")
        self._tabs.addTab(self._build_shortcuts_tab(), "Shortcuts")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_canvas_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._canvas_w = QSpinBox()
        self._canvas_w.setRange(1, 4096)
        self._canvas_w.setValue(self._settings.default_canvas_width)
        form.addRow("Default width (px):", self._canvas_w)

        self._canvas_h = QSpinBox()
        self._canvas_h.setRange(1, 4096)
        self._canvas_h.setValue(self._settings.default_canvas_height)
        form.addRow("Default height (px):", self._canvas_h)

        # Grid color as R,G,B,A text field.
        self._grid_color = QLineEdit(
            ",".join(str(v) for v in self._settings.grid_color)
        )
        form.addRow("Grid colour (R,G,B,A):", self._grid_color)

        self._checker_light = QLineEdit(
            ",".join(str(v) for v in self._settings.checker_light)
        )
        form.addRow("Checker light (R,G,B):", self._checker_light)

        self._checker_dark = QLineEdit(
            ",".join(str(v) for v in self._settings.checker_dark)
        )
        form.addRow("Checker dark (R,G,B):", self._checker_dark)

        return w

    def _build_editor_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._undo_depth = QSpinBox()
        self._undo_depth.setRange(1, 1000)
        self._undo_depth.setValue(self._settings.max_undo_depth)
        form.addRow("Max undo depth:", self._undo_depth)

        self._autosave = QSpinBox()
        self._autosave.setRange(0, 600_000)
        self._autosave.setSingleStep(5_000)
        self._autosave.setSpecialValueText("Disabled")
        self._autosave.setValue(self._settings.autosave_interval_ms)
        form.addRow("Autosave interval (ms):", self._autosave)

        self._theme = QComboBox()
        self._theme.addItems(["dark", "light"])
        idx = self._theme.findText(self._settings.theme)
        if idx >= 0:
            self._theme.setCurrentIndex(idx)
        form.addRow("Theme:", self._theme)

        return w

    def _build_shortcuts_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._shortcut_edits: Dict[str, QLineEdit] = {}
        for tool, key in self._settings.keybindings.items():
            edit = QLineEdit(key)
            edit.setMaxLength(1)
            self._shortcut_edits[tool] = edit
            form.addRow(f"{tool.capitalize()}:", edit)

        return w

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        """Write widget values back to the settings object."""
        self._settings.default_canvas_width = self._canvas_w.value()
        self._settings.default_canvas_height = self._canvas_h.value()
        self._settings.max_undo_depth = self._undo_depth.value()
        self._settings.autosave_interval_ms = self._autosave.value()
        self._settings.theme = self._theme.currentText()

        try:
            parts = [int(v.strip()) for v in self._grid_color.text().split(",")]
            if len(parts) == 4:
                self._settings.grid_color = tuple(parts)  # type: ignore[assignment]
        except ValueError:
            pass

        try:
            parts = [int(v.strip()) for v in self._checker_light.text().split(",")]
            if len(parts) == 3:
                self._settings.checker_light = tuple(parts)  # type: ignore[assignment]
        except ValueError:
            pass

        try:
            parts = [int(v.strip()) for v in self._checker_dark.text().split(",")]
            if len(parts) == 3:
                self._settings.checker_dark = tuple(parts)  # type: ignore[assignment]
        except ValueError:
            pass

        for tool, edit in self._shortcut_edits.items():
            val = edit.text().strip().upper()
            if val:
                self._settings.keybindings[tool] = val

        self.accept()
