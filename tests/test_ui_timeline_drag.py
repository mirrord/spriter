# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for animated click-drag timeline frame reordering."""

from __future__ import annotations

import numpy as np


def _make_sprite(w: int = 8, h: int = 8, layers: int = 1, frames: int = 1):
    from spriter.core.sprite import Sprite

    s = Sprite(w, h)
    for i in range(layers):
        s.add_layer(f"Layer {i + 1}")
    for _ in range(frames):
        s.add_frame()
    return s


def _make_stack():
    from spriter.commands.base import CommandStack

    return CommandStack()


def _paint_distinct(sprite, frames: int) -> None:
    """Give each frame a unique red value so order can be verified."""
    for fi in range(frames):
        pixels = np.zeros((sprite.height, sprite.width, 4), dtype=np.uint8)
        pixels[:, :] = [fi * 40 + 10, 0, 0, 255]
        sprite.set_cel_pixels(0, fi, pixels)


# ---------------------------------------------------------------------------
# Pure geometry helpers
# ---------------------------------------------------------------------------


class TestGeometryHelpers:
    def test_cell_span_matches_width_plus_spacing(self, qapp):
        from spriter.ui.timeline import TimelinePanel, _FrameCell

        panel = TimelinePanel(_make_sprite(frames=3), _make_stack())
        expected = _FrameCell._CELL_W + panel._strip_layout.spacing()
        assert panel._cell_span() == expected

    def test_target_positions_open_gap_and_close_origin(self, qapp):
        from spriter.ui.timeline import TimelinePanel

        panel = TimelinePanel(_make_sprite(frames=4), _make_stack())
        # Simulate a settled layout: home x = index * span.
        from PyQt6.QtCore import QPoint

        span = panel._cell_span()
        panel._home_positions = [QPoint(i * span, 0) for i in range(4)]

        # Drag frame 0 to slot 2. Remaining frames 1,2,3 fill slots {0,1,3}
        # in order, reserving slot 2 for the dragged frame.
        targets = panel._compute_target_positions(source=0, hover=2)
        assert targets == {
            1: 0 * span,
            2: 1 * span,
            3: 3 * span,
        }

    def test_target_positions_drag_to_start(self, qapp):
        from PyQt6.QtCore import QPoint

        from spriter.ui.timeline import TimelinePanel

        panel = TimelinePanel(_make_sprite(frames=3), _make_stack())
        span = panel._cell_span()
        panel._home_positions = [QPoint(i * span, 0) for i in range(3)]

        # Drag frame 2 to slot 0: frames 0,1 shift right to slots 1,2.
        targets = panel._compute_target_positions(source=2, hover=0)
        assert targets == {0: 1 * span, 1: 2 * span}

    def test_insertion_index_maps_cursor_x_to_nearest_slot(self, qapp):
        from PyQt6.QtCore import QPoint

        from spriter.ui.timeline import TimelinePanel

        panel = TimelinePanel(_make_sprite(frames=4), _make_stack())
        span = panel._cell_span()
        panel._home_positions = [QPoint(i * span, 0) for i in range(4)]

        assert panel._insertion_index_for_x(0) == 0
        assert panel._insertion_index_for_x(span * 2 + 3) == 2
        # Beyond the last slot clamps to n-1.
        assert panel._insertion_index_for_x(span * 99) == 3
        # Before the first slot clamps to 0.
        assert panel._insertion_index_for_x(-500) == 0

    def test_insertion_index_none_without_home_positions(self, qapp):
        from spriter.ui.timeline import TimelinePanel

        panel = TimelinePanel(_make_sprite(frames=3), _make_stack())
        panel._home_positions = []
        assert panel._insertion_index_for_x(50) is None


# ---------------------------------------------------------------------------
# Drag lifecycle
# ---------------------------------------------------------------------------


class TestDragLifecycle:
    def _mouse_event(self, etype, x=0.0, y=0.0):
        from PyQt6.QtCore import QPointF, Qt
        from PyQt6.QtGui import QMouseEvent

        return QMouseEvent(
            etype,
            QPointF(x, y),
            QPointF(x, y),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )

    def test_begin_drag_captures_home_and_lifts(self, qapp):
        from PyQt6.QtGui import QMouseEvent

        from spriter.ui.timeline import TimelinePanel

        panel = TimelinePanel(_make_sprite(frames=3), _make_stack())
        panel.resize(300, 100)
        panel._drag_source = 1
        press = self._mouse_event(QMouseEvent.Type.MouseButtonPress)
        panel._begin_drag(press)

        assert panel._dragging is True
        assert len(panel._home_positions) == 3
        assert panel._drag_indicator == 1

    def test_drop_reorders_and_updates_active(self, qapp):
        from PyQt6.QtGui import QMouseEvent

        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=3)
        _paint_distinct(s, 3)
        panel = TimelinePanel(s, _make_stack())
        panel.resize(300, 100)

        # Drag frame 0 → slot 2.
        panel._drag_source = 0
        panel._dragging = True
        panel._drag_indicator = 2

        release = self._mouse_event(QMouseEvent.Type.MouseButtonRelease)
        panel.eventFilter(panel._cells[0], release)

        assert panel._active_frame == 2
        # Original frame 0 (red=10) is now last.
        last = s.get_cel(0, 2).pixels
        assert last is not None
        assert int(last[0, 0, 0]) == 10
        # Drag state cleared.
        assert panel._drag_source is None
        assert panel._dragging is False

    def test_drop_reorder_is_undoable(self, qapp):
        from PyQt6.QtGui import QMouseEvent

        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=3)
        _paint_distinct(s, 3)
        stack = _make_stack()
        panel = TimelinePanel(s, stack)
        panel.resize(300, 100)

        panel._drag_source = 0
        panel._dragging = True
        panel._drag_indicator = 2
        release = self._mouse_event(QMouseEvent.Type.MouseButtonRelease)
        panel.eventFilter(panel._cells[0], release)

        stack.undo()
        first = s.get_cel(0, 0).pixels
        assert first is not None
        assert int(first[0, 0, 0]) == 10  # back at the front

    def test_drop_without_move_does_not_push(self, qapp):
        from PyQt6.QtGui import QMouseEvent

        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=3)
        stack = _make_stack()
        panel = TimelinePanel(s, stack)
        panel.resize(300, 100)

        panel._drag_source = 1
        panel._dragging = True
        panel._drag_indicator = 1  # dropped on itself
        release = self._mouse_event(QMouseEvent.Type.MouseButtonRelease)
        panel.eventFilter(panel._cells[1], release)

        assert not stack.can_undo
        assert panel._dragging is False

    def test_animate_displaced_creates_animations(self, qapp):
        from PyQt6.QtCore import QPoint

        from spriter.ui.timeline import TimelinePanel

        panel = TimelinePanel(_make_sprite(frames=4), _make_stack())
        panel.resize(400, 100)
        span = panel._cell_span()
        panel._home_positions = [QPoint(i * span, 0) for i in range(4)]
        panel._drag_source = 0

        panel._animate_displaced(hover=2)
        # One animation per non-dragged cell.
        assert set(panel._cell_anims.keys()) == {1, 2, 3}

    def test_reset_drag_state_stops_animations(self, qapp):
        from PyQt6.QtCore import QPoint

        from spriter.ui.timeline import TimelinePanel

        panel = TimelinePanel(_make_sprite(frames=3), _make_stack())
        panel.resize(300, 100)
        span = panel._cell_span()
        panel._home_positions = [QPoint(i * span, 0) for i in range(3)]
        panel._drag_source = 0
        panel._animate_displaced(hover=1)
        assert panel._cell_anims

        panel._reset_drag_state()
        assert panel._cell_anims == {}
        assert panel._home_positions == []
        assert panel._drag_source is None


# ---------------------------------------------------------------------------
# Existing-path compatibility (threshold gate)
# ---------------------------------------------------------------------------


class TestThresholdGate:
    def test_no_drag_under_threshold(self, qapp):
        from PyQt6.QtCore import QPointF, Qt
        from PyQt6.QtGui import QMouseEvent

        from spriter.ui.timeline import TimelinePanel

        panel = TimelinePanel(_make_sprite(frames=2), _make_stack())
        cell0 = panel._cells[0]

        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(10.0, 10.0),
            QPointF(10.0, 10.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel.eventFilter(cell0, press)
        assert panel._drag_source == 0
        assert not panel._dragging

        move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(12.0, 10.0),
            QPointF(12.0, 10.0),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        panel.eventFilter(cell0, move)
        assert not panel._dragging
