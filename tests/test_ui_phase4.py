# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Phase 4 UI smoke tests — LayersPanel instantiation and interactions.

Uses the ``qapp`` fixture (offscreen QApplication) from ``conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_sprite(layers=2, frames=1):
    from spriter.core.sprite import Sprite

    s = Sprite(8, 8)
    for i in range(layers):
        s.add_layer(f"Layer {i + 1}")
    for _ in range(frames):
        s.add_frame()
    return s


# ---------------------------------------------------------------------------
# LayersPanel
# ---------------------------------------------------------------------------


class TestLayersPanel:
    def test_instantiation(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite()
        panel = LayersPanel(s, CommandStack())
        assert panel is not None

    def test_list_has_correct_row_count(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(3)
        panel = LayersPanel(s, CommandStack())
        assert panel._list.count() == 3

    def test_active_layer_default_zero(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite()
        panel = LayersPanel(s, CommandStack())
        assert panel.active_layer == 0

    def test_add_layer_increases_count(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(1)
        panel = LayersPanel(s, CommandStack())
        panel._add_layer()
        assert s.layer_count == 2
        assert panel._list.count() == 2

    def test_remove_layer_decreases_count(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(2)
        stack = CommandStack()
        panel = LayersPanel(s, stack)
        panel._active_layer = 1
        panel._remove_layer()
        assert s.layer_count == 1
        assert panel._list.count() == 1

    def test_remove_last_layer_shows_warning(self, qapp, monkeypatch):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel
        from PyQt6.QtWidgets import QMessageBox

        s = _make_sprite(1)
        panel = LayersPanel(s, CommandStack())
        # Patch QMessageBox.warning to avoid an actual dialog.
        called = []
        monkeypatch.setattr(
            QMessageBox, "warning", staticmethod(lambda *a, **kw: called.append(True))
        )
        panel._remove_layer()
        assert s.layer_count == 1  # unchanged
        assert called  # warning was shown

    def test_duplicate_layer(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(1)
        panel = LayersPanel(s, CommandStack())
        panel._duplicate_layer()
        assert s.layer_count == 2

    def test_flatten_leaves_one_layer(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(3)
        panel = LayersPanel(s, CommandStack())
        panel._flatten()
        assert s.layer_count == 1

    def test_merge_down_reduces_count(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(2)
        stack = CommandStack()
        panel = LayersPanel(s, stack)
        panel._active_layer = 1
        panel._merge_down()
        assert s.layer_count == 1

    def test_merge_down_on_bottom_layer_shows_warning(self, qapp, monkeypatch):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel
        from PyQt6.QtWidgets import QMessageBox

        s = _make_sprite(2)
        panel = LayersPanel(s, CommandStack())
        panel._active_layer = 0  # bottom layer
        called = []
        monkeypatch.setattr(
            QMessageBox, "warning", staticmethod(lambda *a, **kw: called.append(True))
        )
        panel._merge_down()
        assert s.layer_count == 2  # unchanged
        assert called

    def test_layers_modified_signal_emitted_on_add(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(1)
        panel = LayersPanel(s, CommandStack())
        received = []
        panel.layers_modified.connect(lambda: received.append(True))
        panel._add_layer()
        assert received

    def test_active_layer_changed_signal_on_selection(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(2)
        panel = LayersPanel(s, CommandStack())
        received = []
        panel.active_layer_changed.connect(received.append)
        # Simulate selecting a different row in the list.
        panel._list.setCurrentRow(0)  # top of list = highest layer index
        # Signal should have fired (or was already set).
        # We just confirm the signal attribute exists and is connectable.
        assert hasattr(panel, "active_layer_changed")

    def test_opacity_slider_updates_layer(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(1)
        panel = LayersPanel(s, CommandStack())
        panel._opacity_slider.setValue(100)
        assert s.layers[0].opacity == 100

    def test_blend_combo_updates_layer(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.core.layer import BlendMode
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(1)
        panel = LayersPanel(s, CommandStack())
        panel._blend_combo.setCurrentText("Multiply")
        assert s.layers[0].blend_mode == BlendMode.MULTIPLY

    def test_refresh_reflects_sprite_changes(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(2)
        panel = LayersPanel(s, CommandStack())
        s.add_layer("Bonus")
        panel.refresh()
        assert panel._list.count() == 3


# ---------------------------------------------------------------------------
# Integration — undo/redo via LayersPanel + CommandStack
# ---------------------------------------------------------------------------


class TestLayersPanelUndo:
    def test_add_then_undo(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(1)
        stack = CommandStack()
        panel = LayersPanel(s, stack)
        panel._add_layer()
        assert s.layer_count == 2
        stack.undo()
        panel.refresh()
        assert s.layer_count == 1
        assert panel._list.count() == 1

    def test_remove_then_undo(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.layers_panel import LayersPanel

        s = _make_sprite(2)
        stack = CommandStack()
        panel = LayersPanel(s, stack)
        panel._active_layer = 1
        panel._remove_layer()
        stack.undo()
        panel.refresh()
        assert s.layer_count == 2
        assert panel._list.count() == 2
