# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Phase 6 tests — Transform commands: pixel-exact verification and undo.

All tests are headless (no Qt required) except the MainWindow smoke tests.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sprite_with_pixels(pixels: np.ndarray):
    """Create a 1-layer, 1-frame sprite with the given pixel buffer."""
    from spriter.core.sprite import Sprite

    h, w = pixels.shape[:2]
    s = Sprite(w, h)
    s.add_layer("L")
    s.add_frame()
    s.set_cel_pixels(0, 0, pixels)
    return s


def _solid(color, w=4, h=4) -> np.ndarray:
    buf = np.zeros((h, w, 4), dtype=np.uint8)
    buf[:, :] = color
    return buf


def _get_pixels(sprite):
    return sprite.get_cel(0, 0).pixels.copy()


# ---------------------------------------------------------------------------
# FlipCommand
# ---------------------------------------------------------------------------


class TestFlipCommand:
    def test_flip_horizontal(self):
        from spriter.commands.transform import FlipCommand

        # Create a 4-wide pattern where column 0 is red, rest is transparent.
        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[:, 0] = (255, 0, 0, 255)  # left column red
        s = _sprite_with_pixels(pixels)
        cmd = FlipCommand(s, 0, 0, horizontal=True)
        cmd.execute()
        result = _get_pixels(s)
        # After horizontal flip the rightmost column should be red.
        assert np.all(result[:, 3] == [255, 0, 0, 255])
        assert np.all(result[:, 0] == [0, 0, 0, 0])

    def test_flip_vertical(self):
        from spriter.commands.transform import FlipCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[0, :] = (0, 255, 0, 255)  # top row green
        s = _sprite_with_pixels(pixels)
        cmd = FlipCommand(s, 0, 0, horizontal=False)
        cmd.execute()
        result = _get_pixels(s)
        # After vertical flip the bottom row should be green.
        assert np.all(result[3, :] == [0, 255, 0, 255])
        assert np.all(result[0, :] == [0, 0, 0, 0])

    def test_flip_h_undo(self):
        from spriter.commands.transform import FlipCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[:, 0] = (255, 0, 0, 255)
        s = _sprite_with_pixels(pixels.copy())
        cmd = FlipCommand(s, 0, 0, horizontal=True)
        cmd.execute()
        cmd.undo()
        result = _get_pixels(s)
        assert np.array_equal(result, pixels)

    def test_flip_v_undo(self):
        from spriter.commands.transform import FlipCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[0, :] = (0, 255, 0, 255)
        s = _sprite_with_pixels(pixels.copy())
        cmd = FlipCommand(s, 0, 0, horizontal=False)
        cmd.execute()
        cmd.undo()
        result = _get_pixels(s)
        assert np.array_equal(result, pixels)

    def test_flip_with_selection_mask(self):
        from spriter.commands.transform import FlipCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[:, 0] = (255, 0, 0, 255)
        s = _sprite_with_pixels(pixels.copy())
        # Only select the left half.
        mask = np.zeros((4, 4), dtype=bool)
        mask[:, :2] = True
        s.set_selection(mask)
        cmd = FlipCommand(s, 0, 0, horizontal=True)
        cmd.execute()
        result = _get_pixels(s)
        # The red column was in the left half of the selection; after flipping
        # within the selection it moved to column 1 (the other selected column).
        assert np.all(result[:, 1] == [255, 0, 0, 255])

    def test_description(self):
        from spriter.commands.transform import FlipCommand
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)
        assert "Horizontal" in FlipCommand(s, 0, 0, horizontal=True).description
        assert "Vertical" in FlipCommand(s, 0, 0, horizontal=False).description


# ---------------------------------------------------------------------------
# RotateCommand
# ---------------------------------------------------------------------------


class TestRotateCommand:
    def test_rotate_90_cw(self):
        from spriter.commands.transform import RotateCommand

        # Top row is blue before rotation; should become right column after 90° CW.
        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[0, :] = (0, 0, 255, 255)
        s = _sprite_with_pixels(pixels)
        cmd = RotateCommand(s, 0, 0, 90)
        cmd.execute()
        result = _get_pixels(s)
        assert np.all(result[:, 3] == [0, 0, 255, 255])

    def test_rotate_180(self):
        from spriter.commands.transform import RotateCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[0, 0] = (255, 0, 0, 255)
        s = _sprite_with_pixels(pixels)
        cmd = RotateCommand(s, 0, 0, 180)
        cmd.execute()
        result = _get_pixels(s)
        # (0,0) should move to (3,3) after 180°.
        assert np.all(result[3, 3] == [255, 0, 0, 255])
        assert np.all(result[0, 0] == [0, 0, 0, 0])

    def test_rotate_270_cw_equals_90_ccw(self):
        from spriter.commands.transform import RotateCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[0, :] = (0, 255, 0, 255)  # top row green
        s = _sprite_with_pixels(pixels)
        cmd = RotateCommand(s, 0, 0, 270)
        cmd.execute()
        result = _get_pixels(s)
        # 270° CW (= 90° CCW): top row → left column.
        assert np.all(result[:, 0] == [0, 255, 0, 255])

    def test_rotate_undo(self):
        from spriter.commands.transform import RotateCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[0, 0] = (255, 128, 0, 255)
        s = _sprite_with_pixels(pixels.copy())
        cmd = RotateCommand(s, 0, 0, 90)
        cmd.execute()
        cmd.undo()
        assert np.array_equal(_get_pixels(s), pixels)

    def test_description(self):
        from spriter.commands.transform import RotateCommand
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)
        assert "90" in RotateCommand(s, 0, 0, 90).description


# ---------------------------------------------------------------------------
# ShiftCommand
# ---------------------------------------------------------------------------


class TestShiftCommand:
    def test_shift_right(self):
        from spriter.commands.transform import ShiftCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[:, 0] = (255, 0, 0, 255)  # left column red
        s = _sprite_with_pixels(pixels)
        cmd = ShiftCommand(s, 0, 0, dx=1, dy=0)
        cmd.execute()
        result = _get_pixels(s)
        assert np.all(result[:, 1] == [255, 0, 0, 255])

    def test_shift_down(self):
        from spriter.commands.transform import ShiftCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[0, :] = (0, 255, 0, 255)  # top row green
        s = _sprite_with_pixels(pixels)
        cmd = ShiftCommand(s, 0, 0, dx=0, dy=1)
        cmd.execute()
        result = _get_pixels(s)
        assert np.all(result[1, :] == [0, 255, 0, 255])

    def test_shift_wraps_right(self):
        from spriter.commands.transform import ShiftCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[:, 3] = (0, 0, 255, 255)  # rightmost column blue
        s = _sprite_with_pixels(pixels)
        cmd = ShiftCommand(s, 0, 0, dx=1, dy=0)
        cmd.execute()
        result = _get_pixels(s)
        # Shifted right by 1, column 3 wraps to column 0.
        assert np.all(result[:, 0] == [0, 0, 255, 255])

    def test_shift_undo(self):
        from spriter.commands.transform import ShiftCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[:, 0] = (255, 0, 0, 255)
        s = _sprite_with_pixels(pixels.copy())
        cmd = ShiftCommand(s, 0, 0, dx=2, dy=1)
        cmd.execute()
        cmd.undo()
        assert np.array_equal(_get_pixels(s), pixels)

    def test_description(self):
        from spriter.commands.transform import ShiftCommand
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)
        assert "3" in ShiftCommand(s, 0, 0, 3, 5).description


# ---------------------------------------------------------------------------
# OutlineCommand
# ---------------------------------------------------------------------------


class TestOutlineCommand:
    def test_outline_adds_border_pixels(self):
        from spriter.commands.transform import OutlineCommand

        pixels = np.zeros((5, 5, 4), dtype=np.uint8)
        pixels[2, 2] = (255, 255, 255, 255)  # single white pixel in centre
        s = _sprite_with_pixels(pixels)
        cmd = OutlineCommand(s, 0, 0, outline_color=(0, 0, 0, 255))
        cmd.execute()
        result = _get_pixels(s)
        # The four adjacent pixels should be black.
        assert np.all(result[1, 2] == [0, 0, 0, 255])
        assert np.all(result[3, 2] == [0, 0, 0, 255])
        assert np.all(result[2, 1] == [0, 0, 0, 255])
        assert np.all(result[2, 3] == [0, 0, 0, 255])
        # Original pixel unchanged.
        assert np.all(result[2, 2] == [255, 255, 255, 255])

    def test_outline_does_not_overwrite_opaque_pixels(self):
        from spriter.commands.transform import OutlineCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[1, 1] = (255, 0, 0, 255)
        pixels[1, 2] = (0, 255, 0, 255)  # already opaque neighbour
        s = _sprite_with_pixels(pixels.copy())
        cmd = OutlineCommand(s, 0, 0, outline_color=(0, 0, 255, 255))
        cmd.execute()
        result = _get_pixels(s)
        # Existing opaque pixel (1,2) must not be overwritten.
        assert np.all(result[1, 2] == [0, 255, 0, 255])

    def test_outline_undo(self):
        from spriter.commands.transform import OutlineCommand

        pixels = np.zeros((5, 5, 4), dtype=np.uint8)
        pixels[2, 2] = (255, 255, 255, 255)
        s = _sprite_with_pixels(pixels.copy())
        cmd = OutlineCommand(s, 0, 0)
        cmd.execute()
        cmd.undo()
        assert np.array_equal(_get_pixels(s), pixels)

    def test_description(self):
        from spriter.commands.transform import OutlineCommand
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)
        assert OutlineCommand(s, 0, 0).description == "Outline"


# ---------------------------------------------------------------------------
# ReplaceColorCommand
# ---------------------------------------------------------------------------


class TestReplaceColorCommand:
    def test_exact_replace(self):
        from spriter.commands.transform import ReplaceColorCommand

        pixels = _solid((255, 0, 0, 255), 4, 4)
        s = _sprite_with_pixels(pixels)
        cmd = ReplaceColorCommand(s, 0, 0, (255, 0, 0, 255), (0, 0, 255, 255))
        cmd.execute()
        result = _get_pixels(s)
        assert np.all(result == [0, 0, 255, 255])

    def test_no_match_leaves_pixels(self):
        from spriter.commands.transform import ReplaceColorCommand

        pixels = _solid((255, 0, 0, 255), 4, 4)
        s = _sprite_with_pixels(pixels.copy())
        cmd = ReplaceColorCommand(s, 0, 0, (0, 0, 255, 255), (255, 255, 0, 255))
        cmd.execute()
        assert np.array_equal(_get_pixels(s), pixels)

    def test_tolerance_replace(self):
        from spriter.commands.transform import ReplaceColorCommand

        pixels = np.zeros((2, 2, 4), dtype=np.uint8)
        pixels[0, 0] = (100, 0, 0, 255)
        pixels[0, 1] = (110, 0, 0, 255)  # within tolerance 20
        pixels[1, 0] = (200, 0, 0, 255)  # outside tolerance 20
        s = _sprite_with_pixels(pixels.copy())
        cmd = ReplaceColorCommand(
            s, 0, 0, (100, 0, 0, 255), (0, 255, 0, 255), tolerance=20.0
        )
        cmd.execute()
        result = _get_pixels(s)
        assert np.all(result[0, 0] == [0, 255, 0, 255])
        assert np.all(result[0, 1] == [0, 255, 0, 255])
        assert np.all(result[1, 0] == [200, 0, 0, 255])  # unchanged

    def test_selection_scoped(self):
        from spriter.commands.transform import ReplaceColorCommand

        pixels = _solid((255, 0, 0, 255), 4, 4)
        s = _sprite_with_pixels(pixels.copy())
        # Only first two columns selected.
        mask = np.zeros((4, 4), dtype=bool)
        mask[:, :2] = True
        s.set_selection(mask)
        cmd = ReplaceColorCommand(s, 0, 0, (255, 0, 0, 255), (0, 0, 255, 255))
        cmd.execute()
        result = _get_pixels(s)
        assert np.all(result[:, 0] == [0, 0, 255, 255])
        assert np.all(result[:, 2] == [255, 0, 0, 255])  # outside selection unchanged

    def test_undo(self):
        from spriter.commands.transform import ReplaceColorCommand

        pixels = _solid((255, 0, 0, 255), 4, 4)
        s = _sprite_with_pixels(pixels.copy())
        cmd = ReplaceColorCommand(s, 0, 0, (255, 0, 0, 255), (0, 0, 255, 255))
        cmd.execute()
        cmd.undo()
        assert np.array_equal(_get_pixels(s), pixels)

    def test_description(self):
        from spriter.commands.transform import ReplaceColorCommand
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)
        cmd = ReplaceColorCommand(s, 0, 0, (0, 0, 0, 255), (255, 255, 255, 255))
        assert cmd.description == "Replace Color"


# ---------------------------------------------------------------------------
# AdjustmentCommand
# ---------------------------------------------------------------------------


class TestAdjustmentCommand:
    def test_brightness_increases_values(self):
        from spriter.commands.transform import AdjustmentCommand

        pixels = _solid((100, 100, 100, 255), 4, 4)
        s = _sprite_with_pixels(pixels.copy())
        cmd = AdjustmentCommand(s, 0, 0, brightness=2.0)
        cmd.execute()
        result = _get_pixels(s)
        # Each RGB channel should be brighter.
        assert int(result[0, 0, 0]) > 100

    def test_brightness_undo(self):
        from spriter.commands.transform import AdjustmentCommand

        pixels = _solid((100, 100, 100, 255), 4, 4)
        s = _sprite_with_pixels(pixels.copy())
        cmd = AdjustmentCommand(s, 0, 0, brightness=2.0)
        cmd.execute()
        cmd.undo()
        assert np.array_equal(_get_pixels(s), pixels)

    def test_neutral_adjustment_is_noop(self):
        from spriter.commands.transform import AdjustmentCommand

        pixels = _solid((80, 120, 200, 255), 4, 4)
        s = _sprite_with_pixels(pixels.copy())
        cmd = AdjustmentCommand(s, 0, 0)  # all neutral
        cmd.execute()
        result = _get_pixels(s)
        # Neutral adjustments should not change pixel values significantly.
        diff = np.abs(result.astype(int) - pixels.astype(int))
        assert diff.max() <= 2  # allow ≤2 rounding error from PIL round-trips

    def test_description(self):
        from spriter.commands.transform import AdjustmentCommand
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)
        assert AdjustmentCommand(s, 0, 0).description == "Adjust"


# ---------------------------------------------------------------------------
# CanvasResizeCommand
# ---------------------------------------------------------------------------


class TestCanvasResizeCommand:
    def test_extend_canvas(self):
        from spriter.commands.transform import CanvasResizeCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[0, 0] = (255, 0, 0, 255)
        s = _sprite_with_pixels(pixels)
        cmd = CanvasResizeCommand(s, 8, 8)
        cmd.execute()
        assert s.width == 8
        assert s.height == 8
        # Original pixel should still be at (0,0).
        result = _get_pixels(s)
        assert np.all(result[0, 0] == [255, 0, 0, 255])

    def test_extend_with_offset(self):
        from spriter.commands.transform import CanvasResizeCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[0, 0] = (0, 255, 0, 255)
        s = _sprite_with_pixels(pixels)
        cmd = CanvasResizeCommand(s, 8, 8, offset_x=2, offset_y=2)
        cmd.execute()
        result = _get_pixels(s)
        # Original (0,0) moves to (2,2).
        assert np.all(result[2, 2] == [0, 255, 0, 255])

    def test_crop(self):
        from spriter.commands.transform import CanvasResizeCommand

        pixels = np.zeros((8, 8, 4), dtype=np.uint8)
        pixels[0, 0] = (255, 0, 0, 255)
        pixels[7, 7] = (0, 0, 255, 255)
        s = _sprite_with_pixels(pixels)
        cmd = CanvasResizeCommand(s, 4, 4)
        cmd.execute()
        assert s.width == 4
        assert s.height == 4
        result = _get_pixels(s)
        # Pixel at (0,0) survives.
        assert np.all(result[0, 0] == [255, 0, 0, 255])
        # Pixel at original (7,7) is cropped away.
        assert result.shape == (4, 4, 4)

    def test_undo_restores_size_and_pixels(self):
        from spriter.commands.transform import CanvasResizeCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[1, 1] = (0, 255, 0, 255)
        s = _sprite_with_pixels(pixels.copy())
        original_w, original_h = s.width, s.height
        cmd = CanvasResizeCommand(s, 8, 8)
        cmd.execute()
        cmd.undo()
        assert s.width == original_w
        assert s.height == original_h
        assert np.array_equal(_get_pixels(s), pixels)

    def test_description(self):
        from spriter.commands.transform import CanvasResizeCommand
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)
        cmd = CanvasResizeCommand(s, 8, 8)
        assert "8" in cmd.description


# ---------------------------------------------------------------------------
# ScaleCommand
# ---------------------------------------------------------------------------


class TestScaleCommand:
    def test_scale_up(self):
        from spriter.commands.transform import ScaleCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[0, 0] = (255, 0, 0, 255)
        s = _sprite_with_pixels(pixels)
        cmd = ScaleCommand(s, 8, 8)
        cmd.execute()
        assert s.width == 8
        assert s.height == 8
        result = _get_pixels(s)
        # After 2× nearest-neighbour scale the top-left 2×2 block should be red.
        assert np.all(result[0, 0] == [255, 0, 0, 255])

    def test_scale_down(self):
        from spriter.commands.transform import ScaleCommand

        pixels = np.zeros((8, 8, 4), dtype=np.uint8)
        s = _sprite_with_pixels(pixels)
        cmd = ScaleCommand(s, 4, 4)
        cmd.execute()
        assert s.width == 4 and s.height == 4

    def test_scale_undo(self):
        from spriter.commands.transform import ScaleCommand

        pixels = np.zeros((4, 4, 4), dtype=np.uint8)
        pixels[0, 0] = (128, 64, 32, 255)
        s = _sprite_with_pixels(pixels.copy())
        cmd = ScaleCommand(s, 8, 8)
        cmd.execute()
        cmd.undo()
        assert s.width == 4 and s.height == 4
        assert np.array_equal(_get_pixels(s), pixels)

    def test_description(self):
        from spriter.commands.transform import ScaleCommand
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)
        assert "8" in ScaleCommand(s, 8, 8).description


# ---------------------------------------------------------------------------
# Sprite resize helpers
# ---------------------------------------------------------------------------


class TestSpriteResizeHelpers:
    def test_resize_canvas_larger(self):
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)
        s.add_layer("L")
        s.add_frame()
        s.resize_canvas(8, 8)
        assert s.width == 8 and s.height == 8

    def test_resize_canvas_invalid_size(self):
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)
        with pytest.raises(ValueError):
            s.resize_canvas(0, 8)

    def test_scale_pixels_changes_size(self):
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)
        s.add_layer("L")
        s.add_frame()
        s.scale_pixels(8, 8)
        assert s.width == 8 and s.height == 8

    def test_scale_pixels_invalid_size(self):
        from spriter.core.sprite import Sprite

        s = Sprite(4, 4)
        with pytest.raises(ValueError):
            s.scale_pixels(4, 0)


# ---------------------------------------------------------------------------
# MainWindow transform menu smoke test
# ---------------------------------------------------------------------------


class TestMainWindowTransforms:
    def test_flip_h_action(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._flip_h()
        assert win._canvas is not None
        win._unsaved = False
        win.close()

    def test_rotate_action(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._rotate(90)
        assert win._canvas is not None
        win._unsaved = False
        win.close()

    def test_canvas_resize_applies(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        from spriter.commands.transform import CanvasResizeCommand

        assert win._sprite is not None
        win._push_transform(CanvasResizeCommand(win._sprite, 16, 16))
        assert win._sprite.width == 16
        win._unsaved = False
        win.close()
