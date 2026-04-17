# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Layers panel — list view with thumbnails, visibility, lock, opacity, and
blend mode controls.

:class:`LayersPanel` wraps a :class:`~PyQt6.QtWidgets.QListWidget` that shows
one row per layer (top → bottom).  Drag-to-reorder is handled via Qt's
built-in internal-move drag-drop, with a :class:`~spriter.commands.layer_ops.MoveLayerCommand`
pushed on drop so the action is undoable.

Each item shows:
* A 24×24 thumbnail of the layer's first-frame pixels
* A visibility (eye) toggle button
* A lock toggle button
* The layer name (editable on double-click)

Below the list: opacity QSlider, blend mode QComboBox, and Add / Delete
/ Duplicate / Merge Down / Flatten action buttons.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..commands.base import CommandStack
from ..commands.layer_ops import (
    AddLayerCommand,
    DuplicateLayerCommand,
    FlattenCommand,
    MergeLayerDownCommand,
    MoveLayerCommand,
    RemoveLayerCommand,
)
from ..core.layer import BlendMode
from ..core.sprite import Sprite

_BLEND_MODE_LABELS = {
    BlendMode.NORMAL: "Normal",
    BlendMode.MULTIPLY: "Multiply",
    BlendMode.SCREEN: "Screen",
    BlendMode.OVERLAY: "Overlay",
    BlendMode.DARKEN: "Darken",
    BlendMode.LIGHTEN: "Lighten",
}
_LABEL_TO_MODE = {v: k for k, v in _BLEND_MODE_LABELS.items()}

_THUMB_SIZE = 24  # thumbnail side length in pixels


def _make_thumbnail(sprite: Sprite, layer_idx: int, frame_idx: int) -> QPixmap:
    """Render a small thumbnail QPixmap for the given layer / frame cel."""
    cel = sprite.get_cel(layer_idx, frame_idx)
    if cel.pixels is None:
        pix = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
        pix.fill(QColor(80, 80, 80))
        return pix

    arr = cel.pixels
    h, w = arr.shape[:2]
    arr_c = np.ascontiguousarray(arr)
    img = QImage(arr_c.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    scaled = img.scaled(
        _THUMB_SIZE,
        _THUMB_SIZE,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )
    return QPixmap.fromImage(scaled)


class LayersPanel(QWidget):
    """Layer-management dock panel.

    Args:
        sprite: The sprite document to manage.
        stack: The undo/redo command stack.
        parent: Optional Qt parent widget.

    Signals:
        active_layer_changed: Emitted when the user selects a different layer.
            Carries the new layer index.
        layer_visibility_changed: Emitted when a visibility toggle is flipped.
            Carries ``(layer_index, new_visible_state)``.
        layers_modified: Emitted after any structural change (add/remove/merge/
            flatten/reorder) so the canvas can refresh.
    """

    active_layer_changed = pyqtSignal(int)
    layer_visibility_changed = pyqtSignal(int, bool)
    layers_modified = pyqtSignal()

    def __init__(
        self,
        sprite: Sprite,
        stack: CommandStack,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._sprite = sprite
        self._stack = stack
        self._active_layer: int = 0
        self._refreshing: bool = False

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Layer list ────────────────────────────────────────────────
        self._list = QListWidget(self)
        self._list.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        root.addWidget(self._list, 1)

        # ── Blend mode + opacity ──────────────────────────────────────
        blend_row = QHBoxLayout()
        blend_row.addWidget(QLabel("Blend:"))
        self._blend_combo = QComboBox()
        for label in _BLEND_MODE_LABELS.values():
            self._blend_combo.addItem(label)
        self._blend_combo.currentTextChanged.connect(self._on_blend_changed)
        blend_row.addWidget(self._blend_combo, 1)
        root.addLayout(blend_row)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity:"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 255)
        self._opacity_slider.setValue(255)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider, 1)
        self._opacity_label = QLabel("255")
        self._opacity_label.setFixedWidth(28)
        opacity_row.addWidget(self._opacity_label)
        root.addLayout(opacity_row)

        # ── Bottom action buttons ─────────────────────────────────────
        btn_row = QHBoxLayout()
        for label, slot in (
            ("+", self._add_layer),
            ("−", self._remove_layer),
            ("⧉", self._duplicate_layer),
            ("↓ Merge", self._merge_down),
            ("⊞ Flatten", self._flatten),
        ):
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        root.addLayout(btn_row)

        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active_layer(self) -> int:
        """Index of the currently selected layer."""
        return self._active_layer

    def refresh(self) -> None:
        """Rebuild the list from the sprite's current layer state."""
        self._refreshing = True
        self._list.clear()
        # Show layers top-to-bottom (highest index first for visual naturalness).
        for li in reversed(range(self._sprite.layer_count)):
            layer = self._sprite.layers[li]
            # Thumbnail from frame 0 if available.
            frame_idx = 0 if self._sprite.frame_count > 0 else -1
            if frame_idx >= 0:
                icon = QIcon(_make_thumbnail(self._sprite, li, frame_idx))
            else:
                pix = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
                pix.fill(QColor(80, 80, 80))
                icon = QIcon(pix)

            eye = "👁" if layer.visible else " "
            lock = "🔒" if layer.locked else " "
            item = QListWidgetItem(icon, f"{eye} {lock}  {layer.name}")
            item.setData(Qt.ItemDataRole.UserRole, li)  # store actual layer index
            self._list.addItem(item)

        # Select the active layer's corresponding list row.
        self._select_row_for_layer(self._active_layer)
        self._refreshing = False
        self._update_controls()

    # ------------------------------------------------------------------
    # Layer actions
    # ------------------------------------------------------------------

    def _add_layer(self) -> None:
        cmd = AddLayerCommand(
            self._sprite, name=f"Layer {self._sprite.layer_count + 1}"
        )
        self._stack.push(cmd)
        self._active_layer = self._sprite.layer_count - 1
        self.refresh()
        self.layers_modified.emit()

    def _remove_layer(self) -> None:
        if self._sprite.layer_count <= 1:
            QMessageBox.warning(
                self, "Cannot Remove", "A sprite must have at least one layer."
            )
            return
        cmd = RemoveLayerCommand(self._sprite, self._active_layer)
        self._stack.push(cmd)
        self._active_layer = max(0, self._active_layer - 1)
        self.refresh()
        self.layers_modified.emit()

    def _duplicate_layer(self) -> None:
        cmd = DuplicateLayerCommand(self._sprite, self._active_layer)
        self._stack.push(cmd)
        self._active_layer = self._active_layer + 1
        self.refresh()
        self.layers_modified.emit()

    def _merge_down(self) -> None:
        if self._active_layer <= 0:
            QMessageBox.warning(self, "Cannot Merge", "No layer below to merge into.")
            return
        cmd = MergeLayerDownCommand(self._sprite, self._active_layer)
        self._stack.push(cmd)
        self._active_layer = max(0, self._active_layer - 1)
        self.refresh()
        self.layers_modified.emit()

    def _flatten(self) -> None:
        cmd = FlattenCommand(self._sprite)
        self._stack.push(cmd)
        self._active_layer = 0
        self.refresh()
        self.layers_modified.emit()

    # ------------------------------------------------------------------
    # List interactions
    # ------------------------------------------------------------------

    def _on_row_changed(self, row: int) -> None:
        if self._refreshing or row < 0:
            return
        item = self._list.item(row)
        if item is None:
            return
        layer_idx = item.data(Qt.ItemDataRole.UserRole)
        if layer_idx is None:
            return
        self._active_layer = layer_idx
        self._update_controls()
        self.active_layer_changed.emit(layer_idx)

    def _on_rows_moved(self, parent, src_start, src_end, dst_parent, dst_row) -> None:
        """Handle drag-to-reorder inside the list."""
        if self._refreshing:
            return
        # Translate list-row positions back into layer indices.
        # In the list, row 0 = layer N-1 (top), so list-row → layer-index math:
        n = self._sprite.layer_count
        from_layer = n - 1 - src_start
        # dst_row in Qt is the insertion point (0…n); clamp to valid range.
        to_layer = max(
            0, min(n - 1, n - 1 - (dst_row if dst_row <= src_start else dst_row - 1))
        )
        if from_layer != to_layer:
            cmd = MoveLayerCommand(self._sprite, from_layer, to_layer)
            self._stack.push(cmd)
            self._active_layer = to_layer
            self.refresh()
            self.layers_modified.emit()

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.addAction("Add Layer", self._add_layer)
        menu.addAction("Duplicate Layer", self._duplicate_layer)
        menu.addAction("Merge Down", self._merge_down)
        menu.addAction("Flatten", self._flatten)
        menu.addSeparator()
        menu.addAction("Delete Layer", self._remove_layer)
        menu.exec(self._list.mapToGlobal(pos))

    def _toggle_visibility(self, layer_idx: int) -> None:
        layer = self._sprite._layers[layer_idx]  # type: ignore[attr-defined]
        layer.visible = not layer.visible
        self.refresh()
        self.layer_visibility_changed.emit(layer_idx, layer.visible)
        self.layers_modified.emit()

    # ------------------------------------------------------------------
    # Controls sync
    # ------------------------------------------------------------------

    def _update_controls(self) -> None:
        if not (0 <= self._active_layer < self._sprite.layer_count):
            return
        layer = self._sprite.layers[self._active_layer]
        self._opacity_slider.blockSignals(True)
        self._opacity_slider.setValue(layer.opacity)
        self._opacity_slider.blockSignals(False)
        self._opacity_label.setText(str(layer.opacity))

        mode_label = _BLEND_MODE_LABELS.get(layer.blend_mode, "Normal")
        self._blend_combo.blockSignals(True)
        self._blend_combo.setCurrentText(mode_label)
        self._blend_combo.blockSignals(False)

    def _on_opacity_changed(self, value: int) -> None:
        if self._refreshing:
            return
        if 0 <= self._active_layer < self._sprite.layer_count:
            self._sprite._layers[self._active_layer].opacity = value  # type: ignore[attr-defined]
            self._opacity_label.setText(str(value))
            self.layers_modified.emit()

    def _on_blend_changed(self, label: str) -> None:
        if self._refreshing:
            return
        mode = _LABEL_TO_MODE.get(label)
        if mode is not None and 0 <= self._active_layer < self._sprite.layer_count:
            self._sprite._layers[self._active_layer].blend_mode = mode  # type: ignore[attr-defined]
            self.layers_modified.emit()

    def _select_row_for_layer(self, layer_idx: int) -> None:
        """Select the list row that corresponds to *layer_idx*."""
        n = self._sprite.layer_count
        row = n - 1 - layer_idx  # inverted mapping
        if 0 <= row < self._list.count():
            self._list.setCurrentRow(row)
