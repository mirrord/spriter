# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Phase 5 tests — Animation model, TimelinePanel, PreviewWindow, onion skinning.

Uses the ``qapp`` fixture (offscreen QApplication) from ``conftest.py``.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sprite(layers: int = 1, frames: int = 3):
    from spriter.core.sprite import Sprite

    s = Sprite(8, 8)
    for i in range(layers):
        s.add_layer(f"Layer {i + 1}")
    for _ in range(frames):
        s.add_frame()
    return s


# ---------------------------------------------------------------------------
# AnimationTag
# ---------------------------------------------------------------------------


class TestAnimationTag:
    def test_basic_creation(self):
        from spriter.core.animation import AnimationTag, LoopMode

        tag = AnimationTag("walk", 0, 3)
        assert tag.name == "walk"
        assert tag.from_frame == 0
        assert tag.to_frame == 3
        assert tag.loop_mode == LoopMode.LOOP

    def test_custom_color_and_mode(self):
        from spriter.core.animation import AnimationTag, LoopMode

        tag = AnimationTag(
            "idle", 4, 7, color=(0, 255, 0), loop_mode=LoopMode.PING_PONG
        )
        assert tag.color == (0, 255, 0)
        assert tag.loop_mode == LoopMode.PING_PONG

    def test_invalid_negative_from_frame(self):
        from spriter.core.animation import AnimationTag

        with pytest.raises(ValueError, match="from_frame"):
            AnimationTag("x", -1, 0)

    def test_invalid_to_before_from(self):
        from spriter.core.animation import AnimationTag

        with pytest.raises(ValueError, match="to_frame"):
            AnimationTag("x", 3, 1)

    def test_repr(self):
        from spriter.core.animation import AnimationTag

        tag = AnimationTag("run", 0, 2)
        assert "run" in repr(tag)
        assert "0..2" in repr(tag)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


class TestAnimation:
    def test_default_fps(self):
        from spriter.core.animation import Animation

        anim = Animation()
        assert anim.default_fps == 12

    def test_custom_fps(self):
        from spriter.core.animation import Animation

        anim = Animation(default_fps=24)
        assert anim.default_fps == 24

    def test_invalid_fps(self):
        from spriter.core.animation import Animation

        with pytest.raises(ValueError):
            Animation(default_fps=0)

    def test_loop_mode_default(self):
        from spriter.core.animation import Animation, LoopMode

        anim = Animation()
        assert anim.loop_mode == LoopMode.LOOP

    def test_add_and_list_tags(self):
        from spriter.core.animation import Animation

        anim = Animation()
        tag = anim.add_tag("walk", 0, 3)
        assert tag.name == "walk"
        assert len(anim.tags) == 1
        assert anim.tags[0] is tag

    def test_add_multiple_tags(self):
        from spriter.core.animation import Animation

        anim = Animation()
        anim.add_tag("walk", 0, 3)
        anim.add_tag("idle", 4, 7)
        assert len(anim.tags) == 2

    def test_remove_tag(self):
        from spriter.core.animation import Animation

        anim = Animation()
        anim.add_tag("walk", 0, 3)
        anim.remove_tag("walk")
        assert len(anim.tags) == 0

    def test_remove_missing_tag_raises(self):
        from spriter.core.animation import Animation

        anim = Animation()
        with pytest.raises(KeyError):
            anim.remove_tag("nonexistent")

    def test_tags_returns_copy(self):
        from spriter.core.animation import Animation

        anim = Animation()
        anim.add_tag("walk", 0, 3)
        tags_copy = anim.tags
        tags_copy.clear()
        assert len(anim.tags) == 1  # original unaffected

    def test_get_frame_duration_uses_sprite_frame(self):
        from spriter.core.animation import Animation

        s = _make_sprite(frames=2)
        s.frames[0].duration_ms = 200
        anim = Animation()
        assert anim.get_frame_duration_ms(s, 0) == 200

    def test_get_frame_duration_fallback(self):
        from spriter.core.animation import Animation
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)  # no frames
        anim = Animation(default_fps=10)
        # No frames → fallback to 1000 // 10 = 100
        assert anim.get_frame_duration_ms(s, 0) == 100

    def test_next_frame_loop_wraps(self):
        from spriter.core.animation import Animation, LoopMode

        anim = Animation(loop_mode=LoopMode.LOOP)
        assert anim.next_frame(2, 3) == 0  # wrap around

    def test_next_frame_loop_normal(self):
        from spriter.core.animation import Animation, LoopMode

        anim = Animation(loop_mode=LoopMode.LOOP)
        assert anim.next_frame(1, 3) == 2

    def test_next_frame_one_shot_stops(self):
        from spriter.core.animation import Animation, LoopMode

        anim = Animation(loop_mode=LoopMode.ONE_SHOT)
        assert anim.next_frame(2, 3) == 2  # stays at last frame

    def test_next_frame_one_shot_advances(self):
        from spriter.core.animation import Animation, LoopMode

        anim = Animation(loop_mode=LoopMode.ONE_SHOT)
        assert anim.next_frame(1, 3) == 2

    def test_next_frame_single_frame(self):
        from spriter.core.animation import Animation

        anim = Animation()
        assert anim.next_frame(0, 1) == 0

    def test_next_frame_pingpong_wraps(self):
        from spriter.core.animation import Animation, LoopMode

        anim = Animation(loop_mode=LoopMode.PING_PONG)
        # PING_PONG uses the same wrap as LOOP for next_frame;
        # direction tracking is the caller's responsibility.
        assert anim.next_frame(2, 3) == 0


# ---------------------------------------------------------------------------
# Sprite.animation attribute
# ---------------------------------------------------------------------------


class TestSpriteAnimation:
    def test_sprite_has_animation(self):
        from spriter.core.animation import Animation
        from spriter.core.sprite import Sprite

        s = Sprite(8, 8)
        assert isinstance(s.animation, Animation)

    def test_animation_is_independent_per_sprite(self):
        from spriter.core.sprite import Sprite

        s1 = Sprite(4, 4)
        s2 = Sprite(4, 4)
        s1.animation.default_fps = 24
        assert s2.animation.default_fps == 12  # unchanged


# ---------------------------------------------------------------------------
# TimelinePanel
# ---------------------------------------------------------------------------


class TestTimelinePanel:
    def test_instantiation(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=3)
        panel = TimelinePanel(s, CommandStack())
        assert panel is not None

    def test_cell_count_matches_frames(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=4)
        panel = TimelinePanel(s, CommandStack())
        assert len(panel._cells) == 4

    def test_active_frame_default(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=3)
        panel = TimelinePanel(s, CommandStack())
        assert panel.active_frame == 0

    def test_set_active_frame(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=3)
        panel = TimelinePanel(s, CommandStack())
        panel.set_active_frame(2)
        assert panel.active_frame == 2

    def test_frame_selected_signal(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=3)
        panel = TimelinePanel(s, CommandStack())
        received = []
        panel.frame_selected.connect(received.append)
        # Simulate a cell click directly.
        panel._on_cell_clicked(1)
        assert received == [1]

    def test_add_frame_button_increases_count(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=2)
        panel = TimelinePanel(s, CommandStack())
        panel._add_frame()
        assert s.frame_count == 3
        assert len(panel._cells) == 3

    def test_remove_frame_decreases_count(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=3)
        panel = TimelinePanel(s, CommandStack())
        panel._active_frame = 1
        panel._remove_frame()
        assert s.frame_count == 2
        assert len(panel._cells) == 2

    def test_remove_last_frame_shows_warning(self, qapp, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox

        from spriter.commands.base import CommandStack
        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=1)
        panel = TimelinePanel(s, CommandStack())
        warned = []
        monkeypatch.setattr(
            QMessageBox, "warning", staticmethod(lambda *a, **kw: warned.append(1))
        )
        panel._remove_frame()
        assert s.frame_count == 1
        assert warned

    def test_duplicate_frame(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=2)
        panel = TimelinePanel(s, CommandStack())
        panel._duplicate_frame()
        assert s.frame_count == 3

    def test_refresh_rebuilds_cells(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.timeline import TimelinePanel

        s = _make_sprite(frames=2)
        panel = TimelinePanel(s, CommandStack())
        s.add_frame()
        panel.refresh()
        assert len(panel._cells) == 3


# ---------------------------------------------------------------------------
# PreviewWindow
# ---------------------------------------------------------------------------


class TestPreviewWindow:
    def test_instantiation(self, qapp):
        from spriter.ui.preview import PreviewWindow

        s = _make_sprite(layers=1, frames=3)
        pw = PreviewWindow(s)
        assert pw is not None

    def test_initial_frame_zero(self, qapp):
        from spriter.ui.preview import PreviewWindow

        s = _make_sprite(frames=3)
        pw = PreviewWindow(s)
        assert pw.current_frame == 0

    def test_not_playing_initially(self, qapp):
        from spriter.ui.preview import PreviewWindow

        s = _make_sprite(frames=3)
        pw = PreviewWindow(s)
        assert not pw.is_playing

    def test_step_forward(self, qapp):
        from spriter.ui.preview import PreviewWindow

        s = _make_sprite(frames=3)
        pw = PreviewWindow(s)
        pw.step_forward()
        assert pw.current_frame == 1

    def test_step_backward_wraps(self, qapp):
        from spriter.ui.preview import PreviewWindow

        s = _make_sprite(frames=3)
        pw = PreviewWindow(s)
        pw.step_backward()
        assert pw.current_frame == 2  # wraps to last frame

    def test_step_forward_wraps(self, qapp):
        from spriter.ui.preview import PreviewWindow

        s = _make_sprite(frames=3)
        pw = PreviewWindow(s)
        pw.step_forward()
        pw.step_forward()
        pw.step_forward()
        assert pw.current_frame == 0  # wraps back to 0

    def test_play_sets_playing(self, qapp):
        from spriter.ui.preview import PreviewWindow

        s = _make_sprite(frames=3)
        pw = PreviewWindow(s)
        pw.play()
        assert pw.is_playing
        pw.pause()  # cleanup

    def test_pause_stops_playing(self, qapp):
        from spriter.ui.preview import PreviewWindow

        s = _make_sprite(frames=3)
        pw = PreviewWindow(s)
        pw.play()
        pw.pause()
        assert not pw.is_playing

    def test_toggle_play(self, qapp):
        from spriter.ui.preview import PreviewWindow

        s = _make_sprite(frames=3)
        pw = PreviewWindow(s)
        pw.toggle_play()
        assert pw.is_playing
        pw.toggle_play()
        assert not pw.is_playing

    def test_frame_changed_signal_on_step(self, qapp):
        from spriter.ui.preview import PreviewWindow

        s = _make_sprite(frames=3)
        pw = PreviewWindow(s)
        received = []
        pw.frame_changed.connect(received.append)
        pw.step_forward()
        assert received == [1]

    def test_set_sprite(self, qapp):
        from spriter.core.sprite import Sprite
        from spriter.ui.preview import PreviewWindow

        s = _make_sprite(frames=3)
        pw = PreviewWindow(s)
        pw.step_forward()
        s2 = Sprite(4, 4)
        s2.add_layer("L")
        s2.add_frame()
        pw.set_sprite(s2)
        assert pw.current_frame == 0


# ---------------------------------------------------------------------------
# Canvas onion skinning
# ---------------------------------------------------------------------------


class TestOnionSkinning:
    def test_canvas_has_onion_attrs(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite(frames=3)
        canvas = CanvasWidget(s, CommandStack())
        assert hasattr(canvas, "onion_before")
        assert hasattr(canvas, "onion_after")
        assert hasattr(canvas, "onion_opacity")

    def test_onion_defaults_zero(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite(frames=3)
        canvas = CanvasWidget(s, CommandStack())
        assert canvas.onion_before == 0
        assert canvas.onion_after == 0

    def test_onion_can_be_set(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite(frames=3)
        canvas = CanvasWidget(s, CommandStack())
        canvas.onion_before = 2
        canvas.onion_after = 1
        assert canvas.onion_before == 2
        assert canvas.onion_after == 1
