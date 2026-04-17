# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Phase 8 tests — Polish & Quality-of-Life.

Tests cover: Settings save/load, PreferencesDialog, recent files, symmetry
mode, pixel-perfect stroke, drag-and-drop wiring, reference image, tiling
preview flag, and MainWindow smoke tests for the new features.

Qt widget tests require the ``qapp`` fixture from conftest.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sprite(w: int = 8, h: int = 8, frames: int = 1, layers: int = 1):
    from spriter.core.sprite import Sprite

    s = Sprite(w, h)
    for i in range(layers):
        s.add_layer(f"Layer {i + 1}")
    for _ in range(frames):
        s.add_frame()
    return s


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_defaults(self):
        from spriter.core.settings import Settings

        s = Settings()
        assert s.default_canvas_width == 32
        assert s.default_canvas_height == 32
        assert s.max_undo_depth == 100
        assert s.theme == "dark"
        assert s.recent_files == []
        assert "pencil" in s.keybindings

    def test_save_and_load_roundtrip(self, tmp_path):
        from spriter.core.settings import Settings

        s = Settings()
        s.default_canvas_width = 64
        s.default_canvas_height = 128
        s.max_undo_depth = 50
        s.theme = "light"
        cfg = tmp_path / "settings.json"
        s.save(cfg)

        loaded = Settings.load(cfg)
        assert loaded.default_canvas_width == 64
        assert loaded.default_canvas_height == 128
        assert loaded.max_undo_depth == 50
        assert loaded.theme == "light"

    def test_load_missing_file_returns_defaults(self, tmp_path):
        from spriter.core.settings import Settings

        s = Settings.load(tmp_path / "nonexistent.json")
        assert s.default_canvas_width == 32

    def test_load_corrupt_file_returns_defaults(self, tmp_path):
        from spriter.core.settings import Settings

        bad = tmp_path / "bad.json"
        bad.write_text("NOT VALID JSON", encoding="utf-8")
        s = Settings.load(bad)
        assert s.default_canvas_width == 32

    def test_to_dict_contains_all_keys(self):
        from spriter.core.settings import Settings

        d = Settings().to_dict()
        for key in [
            "default_canvas_width",
            "default_canvas_height",
            "max_undo_depth",
            "autosave_interval_ms",
            "grid_color",
            "checker_light",
            "checker_dark",
            "theme",
            "recent_files",
            "keybindings",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_from_dict_ignores_unknown_keys(self):
        from spriter.core.settings import Settings

        d = Settings().to_dict()
        d["unknown_future_key"] = True
        # Should not raise.
        s = Settings.from_dict(d)
        assert s.default_canvas_width == 32

    def test_add_recent_file_prepends(self):
        from spriter.core.settings import Settings

        s = Settings()
        s.add_recent_file("/path/a.spriter")
        s.add_recent_file("/path/b.spriter")
        assert s.recent_files[0] == "/path/b.spriter"
        assert s.recent_files[1] == "/path/a.spriter"

    def test_add_recent_file_no_duplicates(self):
        from spriter.core.settings import Settings

        s = Settings()
        s.add_recent_file("/file.spriter")
        s.add_recent_file("/file.spriter")
        assert s.recent_files.count("/file.spriter") == 1
        assert len(s.recent_files) == 1

    def test_recent_files_trimmed_to_max(self):
        from spriter.core.settings import Settings

        s = Settings()
        s.max_recent_files = 3
        for i in range(5):
            s.add_recent_file(f"/file{i}.spriter")
        assert len(s.recent_files) == 3

    def test_keybindings_save_load(self, tmp_path):
        from spriter.core.settings import Settings

        s = Settings()
        s.keybindings["pencil"] = "P"
        cfg = tmp_path / "kb.json"
        s.save(cfg)
        loaded = Settings.load(cfg)
        assert loaded.keybindings["pencil"] == "P"

    def test_grid_color_roundtrip(self, tmp_path):
        from spriter.core.settings import Settings

        s = Settings()
        s.grid_color = (10, 20, 30, 200)
        cfg = tmp_path / "s.json"
        s.save(cfg)
        loaded = Settings.load(cfg)
        assert loaded.grid_color == (10, 20, 30, 200)


# ---------------------------------------------------------------------------
# PreferencesDialog
# ---------------------------------------------------------------------------


class TestPreferencesDialog:
    def test_instantiation(self, qapp):
        from spriter.core.settings import Settings
        from spriter.ui.preferences import PreferencesDialog

        dlg = PreferencesDialog(Settings())
        assert dlg is not None

    def test_has_three_tabs(self, qapp):
        from spriter.core.settings import Settings
        from spriter.ui.preferences import PreferencesDialog

        dlg = PreferencesDialog(Settings())
        assert dlg._tabs.count() == 3

    def test_tab_labels(self, qapp):
        from spriter.core.settings import Settings
        from spriter.ui.preferences import PreferencesDialog

        dlg = PreferencesDialog(Settings())
        labels = [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())]
        assert "Canvas" in labels
        assert "Editor" in labels
        assert "Shortcuts" in labels

    def test_canvas_width_reflects_settings(self, qapp):
        from spriter.core.settings import Settings
        from spriter.ui.preferences import PreferencesDialog

        s = Settings()
        s.default_canvas_width = 64
        dlg = PreferencesDialog(s)
        assert dlg._canvas_w.value() == 64

    def test_accept_updates_settings(self, qapp):
        from spriter.core.settings import Settings
        from spriter.ui.preferences import PreferencesDialog

        s = Settings()
        dlg = PreferencesDialog(s)
        dlg._canvas_w.setValue(128)
        dlg._on_accept()
        assert s.default_canvas_width == 128

    def test_undo_depth_reflected(self, qapp):
        from spriter.core.settings import Settings
        from spriter.ui.preferences import PreferencesDialog

        s = Settings()
        s.max_undo_depth = 50
        dlg = PreferencesDialog(s)
        assert dlg._undo_depth.value() == 50

    def test_theme_combo_items(self, qapp):
        from spriter.core.settings import Settings
        from spriter.ui.preferences import PreferencesDialog

        dlg = PreferencesDialog(Settings())
        items = [dlg._theme.itemText(i) for i in range(dlg._theme.count())]
        assert "dark" in items
        assert "light" in items

    def test_shortcut_edits_present_for_all_tools(self, qapp):
        from spriter.core.settings import Settings
        from spriter.ui.preferences import PreferencesDialog

        s = Settings()
        dlg = PreferencesDialog(s)
        for tool in s.keybindings:
            assert tool in dlg._shortcut_edits


# ---------------------------------------------------------------------------
# Pixel-perfect stroke
# ---------------------------------------------------------------------------


class TestPixelPerfect:
    def _make_stroke(self, points):
        """Draw a sequence of drag points and return the resulting pixel buffer."""
        from spriter.commands.base import CommandStack
        from spriter.tools.pencil import PencilTool

        s = _make_sprite(16, 16)
        stack = CommandStack()
        tool = PencilTool(s, stack)
        tool.foreground = (255, 0, 0, 255)
        tool.pixel_perfect = True

        x0, y0 = points[0]
        tool.on_press(x0, y0)
        for x, y in points[1:]:
            tool.on_drag(x, y)
        tool.on_release(*points[-1])

        return s.get_cel(0, 0).pixels.copy()

    def test_pixel_perfect_attribute_default(self):
        from spriter.commands.base import CommandStack
        from spriter.tools.pencil import PencilTool

        s = _make_sprite()
        tool = PencilTool(s, CommandStack())
        assert tool.pixel_perfect is False

    def test_pixel_perfect_no_diagonal_corners(self):
        """L-shaped corners should be removed in pixel-perfect mode."""
        from spriter.tools.pencil import _remove_l_corners

        # Hand-crafted diagonal stroke: (0,0)→(1,1)→(2,1) — (1,1) is an L-corner.
        pts = [(0, 0), (1, 1), (2, 1)]
        result = _remove_l_corners(pts)
        assert (1, 1) not in result

    def test_remove_l_corners_keeps_endpoints(self):
        from spriter.tools.pencil import _remove_l_corners

        pts = [(0, 0), (1, 1), (2, 1)]
        result = _remove_l_corners(pts)
        assert result[0] == (0, 0)
        assert result[-1] == (2, 1)

    def test_remove_l_corners_short_stroke(self):
        from spriter.tools.pencil import _remove_l_corners

        # Fewer than 3 points — nothing should be removed.
        pts = [(0, 0), (1, 0)]
        assert _remove_l_corners(pts) == pts

    def test_pixel_perfect_paints_pixels(self):
        """With pixel_perfect=True, a straight drag should still paint pixels."""
        pixels = self._make_stroke([(0, 0), (1, 0), (2, 0), (3, 0)])
        # Top row should have at least some red pixels.
        assert np.any(pixels[0, :, 0] == 255)

    def test_non_pixel_perfect_same_output_for_straight_line(self):
        """Straight horizontal strokes should be identical regardless of mode."""
        from spriter.commands.base import CommandStack
        from spriter.tools.pencil import PencilTool

        s1 = _make_sprite(16, 16)
        s2 = _make_sprite(16, 16)
        stack = CommandStack()

        for sprite, pp in [(s1, False), (s2, True)]:
            tool = PencilTool(sprite, stack)
            tool.foreground = (255, 0, 0, 255)
            tool.pixel_perfect = pp
            tool.on_press(0, 0)
            tool.on_drag(3, 0)
            tool.on_release(3, 0)

        p1 = s1.get_cel(0, 0).pixels
        p2 = s2.get_cel(0, 0).pixels
        assert np.array_equal(p1, p2)


# ---------------------------------------------------------------------------
# Canvas: symmetry mode
# ---------------------------------------------------------------------------


class TestCanvasSymmetry:
    def test_symmetry_h_attribute(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite()
        c = CanvasWidget(s, CommandStack())
        assert c.symmetry_h is False
        c.symmetry_h = True
        assert c.symmetry_h is True

    def test_symmetry_v_attribute(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite()
        c = CanvasWidget(s, CommandStack())
        assert c.symmetry_v is False

    def test_mirror_point_no_symmetry(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite(8, 8)
        c = CanvasWidget(s, CommandStack())
        pts = c._mirror_point(2, 3)
        assert pts == [(2, 3)]

    def test_mirror_point_horizontal(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite(8, 8)
        c = CanvasWidget(s, CommandStack())
        c.symmetry_h = True
        pts = c._mirror_point(2, 3)
        assert (2, 3) in pts
        assert (5, 3) in pts  # mirror: 7 - 2 = 5

    def test_mirror_point_vertical(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite(8, 8)
        c = CanvasWidget(s, CommandStack())
        c.symmetry_v = True
        pts = c._mirror_point(2, 1)
        assert (2, 1) in pts
        assert (2, 6) in pts  # mirror: 7 - 1 = 6

    def test_mirror_point_both_axes_four_points(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite(8, 8)
        c = CanvasWidget(s, CommandStack())
        c.symmetry_h = True
        c.symmetry_v = True
        pts = c._mirror_point(1, 2)
        assert len(pts) == 4


# ---------------------------------------------------------------------------
# Canvas: reference image and tiling preview flags
# ---------------------------------------------------------------------------


class TestCanvasReferenceAndTiling:
    def test_reference_image_default_none(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite()
        c = CanvasWidget(s, CommandStack())
        assert c.reference_image is None

    def test_reference_image_assignable(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite(8, 8)
        c = CanvasWidget(s, CommandStack())
        ref = np.zeros((8, 8, 4), dtype=np.uint8)
        c.reference_image = ref
        assert c.reference_image is not None

    def test_reference_opacity_default(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite()
        c = CanvasWidget(s, CommandStack())
        assert c.reference_opacity == 0.5

    def test_tiling_preview_default_false(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite()
        c = CanvasWidget(s, CommandStack())
        assert c.tiling_preview is False

    def test_tiling_preview_toggle(self, qapp):
        from spriter.commands.base import CommandStack
        from spriter.ui.canvas import CanvasWidget

        s = _make_sprite()
        c = CanvasWidget(s, CommandStack())
        c.tiling_preview = True
        assert c.tiling_preview is True


# ---------------------------------------------------------------------------
# MainWindow smoke tests (Phase 7+8)
# ---------------------------------------------------------------------------


class TestMainWindowPhase8:
    def test_instantiation(self, qapp):
        from spriter.ui.main_window import MainWindow

        w = MainWindow()
        assert w is not None
        w._unsaved = False
        w.close()

    def test_has_export_menu(self, qapp):
        from spriter.ui.main_window import MainWindow

        w = MainWindow()
        menu_titles = [
            w.menuBar().actions()[i].text()
            for i in range(w.menuBar().actions().__len__())
        ]
        # File menu should contain Export sub-menu
        file_menu = w.menuBar().actions()[0].menu()
        assert file_menu is not None
        sub_texts = [a.text() for a in file_menu.actions()]
        assert any("Export" in t for t in sub_texts)
        w._unsaved = False
        w.close()

    def test_has_recent_files_menu(self, qapp):
        from spriter.ui.main_window import MainWindow

        w = MainWindow()
        file_menu = w.menuBar().actions()[0].menu()
        assert file_menu is not None
        sub_texts = [a.text() for a in file_menu.actions()]
        assert any("Recent" in t for t in sub_texts)
        w._unsaved = False
        w.close()

    def test_has_preferences_menu(self, qapp):
        from spriter.ui.main_window import MainWindow

        w = MainWindow()
        menu_titles = [a.text() for a in w.menuBar().actions()]
        assert any("Preferences" in t for t in menu_titles)
        w._unsaved = False
        w.close()

    def test_settings_loaded(self, qapp):
        from spriter.ui.main_window import MainWindow

        w = MainWindow()
        assert w._settings is not None
        w._unsaved = False
        w.close()

    def test_autosave_timer_present(self, qapp):
        from spriter.ui.main_window import MainWindow

        w = MainWindow()
        assert w._autosave_timer is not None
        w._unsaved = False
        w.close()

    def test_symmetry_toggle_h(self, qapp):
        from spriter.ui.main_window import MainWindow

        w = MainWindow()
        w._sym_h_action.setChecked(True)
        w._toggle_sym_h()
        assert w._canvas.symmetry_h is True
        w._unsaved = False
        w.close()

    def test_symmetry_toggle_v(self, qapp):
        from spriter.ui.main_window import MainWindow

        w = MainWindow()
        w._sym_v_action.setChecked(True)
        w._toggle_sym_v()
        assert w._canvas.symmetry_v is True
        w._unsaved = False
        w.close()

    def test_tiling_preview_toggle(self, qapp):
        from spriter.ui.main_window import MainWindow

        w = MainWindow()
        w._tiling_action.setChecked(True)
        w._toggle_tiling()
        assert w._canvas.tiling_preview is True
        w._unsaved = False
        w.close()

    def test_clear_reference_image(self, qapp):
        from spriter.ui.main_window import MainWindow

        w = MainWindow()
        w._canvas.reference_image = np.zeros((8, 8, 4), dtype=np.uint8)
        w._clear_reference_image()
        assert w._canvas.reference_image is None
        w._unsaved = False
        w.close()

    def test_recent_files_menu_refreshes(self, qapp):
        from spriter.ui.main_window import MainWindow

        w = MainWindow()
        w._settings.add_recent_file("/some/fake.spriter")
        w._refresh_recent_menu()
        actions = w._recent_menu.actions()
        texts = [a.text() for a in actions]
        assert any("/some/fake.spriter" in t for t in texts)
        w._unsaved = False
        w.close()
