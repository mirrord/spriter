# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""View, layer, frame, animation and transform slots for :class:`MainWindow`.

Grouped into :class:`EditMixin` to keep the core window focused on lifecycle
and wiring.  The Replace-Color dialog used by one of these slots lives here too.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..commands.base import CompositeCommand
from ..commands.frame_ops import (
    AddFrameCommand,
    DuplicateFrameCommand,
    RemoveFrameCommand,
)
from ..commands.transform import (
    AdjustmentCommand,
    AutocropCommand,
    CanvasResizeCommand,
    CropToSelectionCommand,
    FlipCommand,
    InvertColorsCommand,
    OutlineCommand,
    ReplaceColorCommand,
    RotateCommand,
    ScaleCommand,
    ScaleSelectionCommand,
    ShiftCommand,
)
from ..core.animation import LoopMode
from ..core.sprite import Sprite
from .main_window_base import _MainWindowProtocol
from .preview import PreviewWindow


class EditMixin(_MainWindowProtocol):
    """Menu slots for view controls, layers, frames, animation and transforms."""

    def _zoom_in(self) -> None:
        if self._canvas:
            self._canvas._zoom_step(1)

    def _zoom_out(self) -> None:
        if self._canvas:
            self._canvas._zoom_step(-1)

    def _fit(self) -> None:
        if self._canvas:
            self._canvas.fit_to_window()

    def _center_view(self) -> None:
        if self._canvas:
            self._canvas.center_view()

    def _toggle_grid(self) -> None:
        self._set_grid_visible(self._grid_action.isChecked())

    def _on_toolbar_grid_toggled(self, checked: bool) -> None:
        self._set_grid_visible(checked)

    def _set_grid_visible(self, visible: bool) -> None:
        """Toggle the pixel grid, keeping the menu action and toolbar button in sync."""
        if self._canvas:
            self._canvas.show_grid = visible
            self._canvas.update()
        if self._grid_action.isChecked() != visible:
            self._grid_action.setChecked(visible)
        if self._toolbar and self._toolbar._grid_button.isChecked() != visible:
            self._toolbar._grid_button.setChecked(visible)

    def _add_layer(self) -> None:
        if self._layers_panel:
            self._layers_panel._add_layer()

    def _delete_layer(self) -> None:
        if self._layers_panel:
            self._layers_panel._remove_layer()

    def _duplicate_layer(self) -> None:
        if self._layers_panel:
            self._layers_panel._duplicate_layer()

    def _merge_down(self) -> None:
        if self._layers_panel:
            self._layers_panel._merge_down()

    def _flatten(self) -> None:
        if self._layers_panel:
            self._layers_panel._flatten()

    def _rename_layer(self) -> None:
        if self._layers_panel:
            self._layers_panel._rename_layer()

    def _set_layer_role_foreground(self) -> None:
        if self._layers_panel:
            from ..core.layer import LayerRole

            self._layers_panel._set_layer_role(LayerRole.FOREGROUND)

    def _set_layer_role_background(self) -> None:
        if self._layers_panel:
            from ..core.layer import LayerRole

            self._layers_panel._set_layer_role(LayerRole.BACKGROUND)

    def _set_layer_role_normal(self) -> None:
        if self._layers_panel:
            from ..core.layer import LayerRole

            self._layers_panel._set_layer_role(LayerRole.NORMAL)

    def _add_frame(self) -> None:
        if self._sprite is None:
            return
        cmd = AddFrameCommand(self._sprite)
        self._stack.push(cmd)
        if self._canvas:
            self._canvas.invalidate_cache()
        if self._timeline:
            self._timeline.refresh()
        self._refresh_undo_redo_labels()

    def _delete_frame(self) -> None:
        if self._sprite is None or self._sprite.frame_count <= 1:
            QMessageBox.warning(
                self, "Cannot Delete", "A sprite must have at least one frame."
            )
            return
        fi = self._canvas.active_frame if self._canvas else 0
        cmd = RemoveFrameCommand(self._sprite, fi)
        self._stack.push(cmd)
        new_fi = max(0, fi - 1)
        if self._canvas:
            self._canvas.active_frame = new_fi
            self._canvas.invalidate_cache()
        if self._timeline:
            self._timeline.set_active_frame(new_fi)
            self._timeline.refresh()
        self._refresh_undo_redo_labels()

    def _duplicate_frame(self) -> None:
        if self._sprite is None:
            return
        fi = self._canvas.active_frame if self._canvas else 0
        cmd = DuplicateFrameCommand(self._sprite, fi)
        self._stack.push(cmd)
        if self._canvas:
            self._canvas.active_frame = fi + 1
            self._canvas.invalidate_cache()
        if self._timeline:
            self._timeline.set_active_frame(fi + 1)
            self._timeline.refresh()
        self._refresh_undo_redo_labels()

    def _move_frame_left(self) -> None:
        if self._timeline:
            self._timeline._move_frame_left()

    def _move_frame_right(self) -> None:
        if self._timeline:
            self._timeline._move_frame_right()

    def _on_timeline_frame_selected(self, frame_index: int) -> None:
        if self._canvas:
            self._canvas.active_frame = frame_index
            self._canvas.invalidate_cache()
            if self._canvas._tool:
                self._canvas._tool.frame_index = frame_index

    def _on_animation_changed(self, timeline_index: int) -> None:
        """Handle the active animation timeline changing in the timeline panel."""
        if self._sprite is None:
            return
        if self._canvas:
            self._canvas.active_frame = 0
            self._canvas.invalidate_cache()
            if self._canvas._tool:
                self._canvas._tool.frame_index = 0
            self._canvas.update()
        if self._preview is not None:
            self._preview.set_sprite(self._sprite)

    # ------------------------------------------------------------------
    # Animation menu actions
    # ------------------------------------------------------------------

    def _show_preview(self) -> None:
        if self._sprite is None:
            return
        if self._preview is None:
            self._preview = PreviewWindow(self._sprite, self)
        else:
            self._preview.set_sprite(self._sprite)
        self._preview.show()
        self._preview.raise_()

    def _set_loop_mode(self, mode: LoopMode) -> None:
        if self._sprite:
            self._sprite.animation.loop_mode = mode

    def _toggle_onion_skin(self) -> None:
        if self._canvas is None:
            return
        enabled = self._onion_action.isChecked()
        self._canvas.onion_before = 1 if enabled else 0
        self._canvas.onion_after = 1 if enabled else 0
        self._canvas.invalidate_cache()

    def _prompt_onion_depth(self) -> None:
        if self._canvas is None:
            return
        before, ok1 = QInputDialog.getInt(
            self,
            "Onion Skin Depth",
            "Frames before active:",
            self._canvas.onion_before,
            0,
            10,
        )
        if not ok1:
            return
        after, ok2 = QInputDialog.getInt(
            self,
            "Onion Skin Depth",
            "Frames after active:",
            self._canvas.onion_after,
            0,
            10,
        )
        if ok2:
            self._canvas.onion_before = before
            self._canvas.onion_after = after
            self._canvas.invalidate_cache()

    # ------------------------------------------------------------------
    # Transform menu actions
    # ------------------------------------------------------------------

    def _active_layer_frame(self):
        li = self._layers_panel.active_layer if self._layers_panel else 0
        fi = self._canvas.active_frame if self._canvas else 0
        return li, fi

    def _push_transform(self, cmd) -> None:
        self._stack.push(cmd)
        if self._canvas:
            self._canvas.invalidate_cache()
        if self._timeline:
            self._timeline.refresh()
        self._unsaved = True
        self._refresh_undo_redo_labels()

    def _flip_h(self) -> None:
        if self._sprite is None:
            return
        li, fi = self._active_layer_frame()
        self._push_transform(FlipCommand(self._sprite, li, fi, horizontal=True))

    def _flip_v(self) -> None:
        if self._sprite is None:
            return
        li, fi = self._active_layer_frame()
        self._push_transform(FlipCommand(self._sprite, li, fi, horizontal=False))

    def _rotate(self, angle: float) -> None:
        if self._sprite is None:
            return
        li, fi = self._active_layer_frame()
        self._push_transform(RotateCommand(self._sprite, li, fi, angle))

    def _prompt_canvas_resize(self) -> None:
        if self._sprite is None:
            return
        w, ok1 = QInputDialog.getInt(
            self, "Canvas Size", "New width (px):", self._sprite.width, 1, 4096
        )
        if not ok1:
            return
        h, ok2 = QInputDialog.getInt(
            self, "Canvas Size", "New height (px):", self._sprite.height, 1, 4096
        )
        if ok2:
            self._push_transform(CanvasResizeCommand(self._sprite, w, h))
            self._status_canvas.setText(f"{self._sprite.width}×{self._sprite.height}")

    def _prompt_scale(self) -> None:
        if self._sprite is None:
            return
        w, ok1 = QInputDialog.getInt(
            self, "Scale Image", "New width (px):", self._sprite.width, 1, 4096
        )
        if not ok1:
            return
        h, ok2 = QInputDialog.getInt(
            self, "Scale Image", "New height (px):", self._sprite.height, 1, 4096
        )
        if ok2:
            self._push_transform(ScaleCommand(self._sprite, w, h))
            self._status_canvas.setText(f"{self._sprite.width}×{self._sprite.height}")

    def _crop_to_selection(self) -> None:
        if self._sprite is None:
            return
        mask = self._sprite.selection_mask
        if mask is None or not bool(mask.any()):
            QMessageBox.information(
                self,
                "Crop to Selection",
                "No active selection. Make a selection first.",
            )
            return
        try:
            cmd = CropToSelectionCommand(self._sprite)
        except ValueError as exc:
            QMessageBox.warning(self, "Crop to Selection", str(exc))
            return
        self._push_transform(cmd)
        self._status_canvas.setText(f"{self._sprite.width}×{self._sprite.height}")

    def _autocrop(self) -> None:
        if self._sprite is None:
            return
        try:
            cmd = AutocropCommand(self._sprite)
        except ValueError as exc:
            QMessageBox.information(self, "Autocrop", str(exc))
            return
        self._push_transform(cmd)
        self._status_canvas.setText(f"{self._sprite.width}×{self._sprite.height}")

    def _prompt_scale_selection(self) -> None:
        if self._sprite is None:
            return
        mask = self._sprite.selection_mask
        if mask is None or not bool(mask.any()):
            QMessageBox.information(
                self,
                "Scale Selection",
                "No active selection. Make a selection first.",
            )
            return
        # Use selection bounding box as the current size baseline.
        import numpy as _np

        sel_rows = _np.any(mask, axis=1)
        sel_cols = _np.any(mask, axis=0)
        cur_h = int(_np.where(sel_rows)[0][-1] - _np.where(sel_rows)[0][0] + 1)
        cur_w = int(_np.where(sel_cols)[0][-1] - _np.where(sel_cols)[0][0] + 1)
        w, ok1 = QInputDialog.getInt(
            self, "Scale Selection", "New width (px):", cur_w, 1, 4096
        )
        if not ok1:
            return
        h, ok2 = QInputDialog.getInt(
            self, "Scale Selection", "New height (px):", cur_h, 1, 4096
        )
        if not ok2:
            return
        li, fi = self._active_layer_frame()
        try:
            cmd = ScaleSelectionCommand(self._sprite, li, fi, w, h)
        except ValueError as exc:
            QMessageBox.warning(self, "Scale Selection", str(exc))
            return
        self._push_transform(cmd)

    def _prompt_shift(self) -> None:
        if self._sprite is None:
            return
        dx, ok1 = QInputDialog.getInt(
            self,
            "Shift",
            "Horizontal offset (px):",
            0,
            -self._sprite.width,
            self._sprite.width,
        )
        if not ok1:
            return
        dy, ok2 = QInputDialog.getInt(
            self,
            "Shift",
            "Vertical offset (px):",
            0,
            -self._sprite.height,
            self._sprite.height,
        )
        if ok2:
            li, fi = self._active_layer_frame()
            self._push_transform(ShiftCommand(self._sprite, li, fi, dx, dy))

    def _apply_outline(self) -> None:
        if self._sprite is None:
            return
        li, fi = self._active_layer_frame()
        self._push_transform(OutlineCommand(self._sprite, li, fi))

    def _prompt_replace_color(self) -> None:
        if self._sprite is None or self._color_picker is None:
            return
        fg = tuple(int(c) for c in self._color_picker.foreground)
        bg = tuple(int(c) for c in self._color_picker.background)
        if fg == bg:
            QMessageBox.information(
                self,
                "Replace Color",
                "Foreground and background colors are identical \u2014 nothing to replace.",
            )
            return
        dlg = _ReplaceColorDialog(fg, bg, self._sprite, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        old_color, new_color = dlg.color_pair()
        tolerance = dlg.tolerance()
        targets = self._scope_targets(dlg.scope())
        _li_active, _fi_active = self._active_layer_frame()
        cmds = [
            ReplaceColorCommand(
                self._sprite, li, fi, old_color, new_color, tolerance=tolerance
            )
            for (li, fi) in targets
        ]
        if not cmds:
            return
        if len(cmds) == 1:
            self._push_transform(cmds[0])
        else:
            self._push_transform(CompositeCommand(cmds, description="Replace Color"))

    def _invert_colors(self) -> None:
        if self._sprite is None:
            return
        # Decide frame scope.
        all_frames = False
        if self._sprite.frame_count > 1:
            choice = QMessageBox.question(
                self,
                "Invert Colors",
                "This sprite has multiple frames. Invert colors on all frames?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if choice == QMessageBox.StandardButton.Cancel:
                return
            all_frames = choice == QMessageBox.StandardButton.Yes
        # Decide layer scope.
        all_layers = False
        if self._sprite.layer_count > 1:
            choice = QMessageBox.question(
                self,
                "Invert Colors",
                "This sprite has multiple layers. Invert colors on all layers?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if choice == QMessageBox.StandardButton.Cancel:
                return
            all_layers = choice == QMessageBox.StandardButton.Yes
        li_active, fi_active = self._active_layer_frame()
        frames = range(self._sprite.frame_count) if all_frames else [fi_active]
        layers = range(self._sprite.layer_count) if all_layers else [li_active]
        cmds = [
            InvertColorsCommand(self._sprite, li, fi, respect_selection=False)
            for fi in frames
            for li in layers
        ]
        if not cmds:
            return
        if len(cmds) == 1:
            self._push_transform(cmds[0])
        else:
            self._push_transform(CompositeCommand(cmds, description="Invert Colors"))

    def _invert_colors_selection(self) -> None:
        if self._sprite is None:
            return
        mask = self._sprite.selection_mask
        if mask is None or not bool(mask.any()):
            QMessageBox.information(
                self,
                "Invert Colors in Selection",
                "No active selection. Make a selection first.",
            )
            return
        li, fi = self._active_layer_frame()
        self._push_transform(
            InvertColorsCommand(self._sprite, li, fi, respect_selection=True)
        )

    def _scope_targets(self, scope: str) -> list:
        """Return a list of (layer_index, frame_index) for the given scope."""
        if self._sprite is None:
            return []
        li, fi = self._active_layer_frame()
        if scope == "active":
            return [(li, fi)]
        if scope == "frame":
            return [(layer_idx, fi) for layer_idx in range(self._sprite.layer_count)]
        if scope == "all":
            return [
                (layer_idx, frame_idx)
                for frame_idx in range(self._sprite.frame_count)
                for layer_idx in range(self._sprite.layer_count)
            ]
        return [(li, fi)]

    def _prompt_adjust_brightness(self) -> None:
        if self._sprite is None:
            return
        val, ok = QInputDialog.getDouble(
            self, "Brightness", "Brightness factor (1.0 = no change):", 1.0, 0.0, 5.0, 2
        )
        if ok:
            li, fi = self._active_layer_frame()
            self._push_transform(
                AdjustmentCommand(self._sprite, li, fi, brightness=val)
            )

    def _prompt_adjust_hue(self) -> None:
        if self._sprite is None:
            return
        val, ok = QInputDialog.getDouble(
            self, "Hue / Saturation", "Hue rotation (degrees):", 0.0, -180.0, 180.0, 1
        )
        if ok:
            li, fi = self._active_layer_frame()
            self._push_transform(AdjustmentCommand(self._sprite, li, fi, hue=val))


# ---------------------------------------------------------------------------
# Replace Color dialog
# ---------------------------------------------------------------------------


class _ReplaceColorDialog(QDialog):
    """Dialog for the Transform > Replace Color action.

    Shows the current foreground and background swatches, lets the user pick
    direction (FG\u2192BG or BG\u2192FG), the cel scope, and a tolerance.
    """

    def __init__(
        self,
        fg: tuple,
        bg: tuple,
        sprite: Sprite,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Replace Color")
        self._fg = tuple(int(c) for c in fg)
        self._bg = tuple(int(c) for c in bg)

        layout = QVBoxLayout(self)

        # Color swatches with labels.
        swatch_row = QHBoxLayout()
        swatch_row.addWidget(QLabel("Foreground:"))
        swatch_row.addWidget(self._make_swatch(self._fg))
        swatch_row.addWidget(QLabel(self._hex_label(self._fg)))
        swatch_row.addSpacing(20)
        swatch_row.addWidget(QLabel("Background:"))
        swatch_row.addWidget(self._make_swatch(self._bg))
        swatch_row.addWidget(QLabel(self._hex_label(self._bg)))
        swatch_row.addStretch(1)
        layout.addLayout(swatch_row)

        # Direction.
        layout.addWidget(QLabel("Direction:"))
        self._dir_fg_to_bg = QRadioButton("Replace Foreground with Background")
        self._dir_bg_to_fg = QRadioButton("Replace Background with Foreground")
        self._dir_fg_to_bg.setChecked(True)
        self._dir_group = QButtonGroup(self)
        self._dir_group.addButton(self._dir_fg_to_bg)
        self._dir_group.addButton(self._dir_bg_to_fg)
        layout.addWidget(self._dir_fg_to_bg)
        layout.addWidget(self._dir_bg_to_fg)

        # Scope.
        layout.addWidget(QLabel("Apply to:"))
        self._scope_active = QRadioButton("Active layer + frame")
        self._scope_frame = QRadioButton("All layers in current frame")
        self._scope_all = QRadioButton("All layers in all frames")
        self._scope_active.setChecked(True)
        self._scope_group = QButtonGroup(self)
        self._scope_group.addButton(self._scope_active)
        self._scope_group.addButton(self._scope_frame)
        self._scope_group.addButton(self._scope_all)
        layout.addWidget(self._scope_active)
        if sprite.layer_count > 1:
            layout.addWidget(self._scope_frame)
        if sprite.frame_count > 1:
            layout.addWidget(self._scope_all)

        # Tolerance.
        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Tolerance:"))
        self._tol_spin = QDoubleSpinBox()
        self._tol_spin.setRange(0.0, 510.0)
        self._tol_spin.setDecimals(1)
        self._tol_spin.setSingleStep(1.0)
        self._tol_spin.setValue(0.0)
        tol_row.addWidget(self._tol_spin)
        tol_row.addStretch(1)
        layout.addLayout(tol_row)

        # Buttons.
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _make_swatch(color: tuple) -> QFrame:
        sw = QFrame()
        sw.setFixedSize(28, 20)
        sw.setFrameShape(QFrame.Shape.Box)
        r, g, b, a = (list(color) + [255, 255, 255, 255])[:4]
        sw.setStyleSheet(
            f"background-color: rgba({r}, {g}, {b}, {a}); border: 1px solid #444;"
        )
        return sw

    @staticmethod
    def _hex_label(color: tuple) -> str:
        r, g, b, a = (list(color) + [255, 255, 255, 255])[:4]
        return f"#{r:02X}{g:02X}{b:02X} (a={a})"

    def color_pair(self) -> tuple:
        """Return ``(old_color, new_color)`` based on the chosen direction."""
        if self._dir_fg_to_bg.isChecked():
            return self._fg, self._bg
        return self._bg, self._fg

    def tolerance(self) -> float:
        return float(self._tol_spin.value())

    def scope(self) -> str:
        if self._scope_all.isChecked():
            return "all"
        if self._scope_frame.isChecked():
            return "frame"
        return "active"
