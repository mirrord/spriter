# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the "remember last file-dialog directory" feature.

When the user opens, imports, exports, or saves a file, the directory
containing the chosen file becomes the default starting location for the
next file dialog (and is persisted to ``settings.json``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Settings-level tests (no Qt required)
# ---------------------------------------------------------------------------


class TestSettingsLastDirectory:
    def test_default_is_empty(self):
        from spriter.core.settings import Settings

        assert Settings().last_directory == ""

    def test_remember_path_records_parent(self, tmp_path):
        from spriter.core.settings import Settings

        s = Settings()
        target = tmp_path / "sub" / "sprite.spriter"
        target.parent.mkdir()
        target.write_text("{}", encoding="utf-8")

        s.remember_path(str(target))
        assert Path(s.last_directory) == target.parent.resolve()

    def test_remember_path_ignores_blank(self):
        from spriter.core.settings import Settings

        s = Settings()
        s.last_directory = "/some/dir"
        s.remember_path("")
        assert s.last_directory == "/some/dir"

    def test_remember_directory_records_directory(self, tmp_path):
        from spriter.core.settings import Settings

        s = Settings()
        s.remember_directory(str(tmp_path))
        assert Path(s.last_directory) == tmp_path.resolve()

    def test_remember_directory_ignores_blank(self):
        from spriter.core.settings import Settings

        s = Settings()
        s.last_directory = "/persisted"
        s.remember_directory("")
        assert s.last_directory == "/persisted"

    def test_persisted_via_to_dict_roundtrip(self, tmp_path):
        from spriter.core.settings import Settings

        s = Settings()
        s.last_directory = str(tmp_path)
        data = s.to_dict()
        assert data["last_directory"] == str(tmp_path)

        loaded = Settings.from_dict(data)
        assert loaded.last_directory == str(tmp_path)

    def test_save_load_roundtrip(self, tmp_path):
        from spriter.core.settings import Settings

        cfg = tmp_path / "settings.json"
        s = Settings()
        s.last_directory = str(tmp_path / "art")
        s.save(cfg)

        loaded = Settings.load(cfg)
        assert loaded.last_directory == str(tmp_path / "art")

    def test_load_missing_field_defaults_to_empty(self):
        from spriter.core.settings import Settings

        s = Settings.from_dict({})
        assert s.last_directory == ""


# ---------------------------------------------------------------------------
# MainWindow integration tests
# ---------------------------------------------------------------------------


def _make_window(tmp_path):
    """Create a MainWindow with an isolated settings file."""
    from spriter.ui.main_window import MainWindow
    from spriter.core.settings import Settings

    win = MainWindow()
    win._unsaved = False
    # Redirect persistence to a tmp file so the user's real config is untouched.
    cfg = tmp_path / "settings.json"
    win._settings = Settings()
    win._settings.save(cfg)
    # Monkey-patch save to use the tmp file.
    win._settings.save = lambda path=cfg: Settings.save(win._settings, path)  # type: ignore[method-assign]
    return win, cfg


class TestDialogDirHelper:
    def test_returns_empty_when_unset(self, qapp, tmp_path):
        win, _ = _make_window(tmp_path)
        try:
            assert win._dialog_dir() == ""
        finally:
            win.close()

    def test_returns_last_directory_when_set(self, qapp, tmp_path):
        win, _ = _make_window(tmp_path)
        try:
            win._settings.last_directory = str(tmp_path)
            assert win._dialog_dir() == str(tmp_path)
        finally:
            win.close()

    def test_falls_back_when_directory_missing(self, qapp, tmp_path):
        win, _ = _make_window(tmp_path)
        try:
            win._settings.last_directory = str(tmp_path / "does-not-exist")
            assert win._dialog_dir() == ""
        finally:
            win.close()


class TestOpenProjectRemembersDir:
    def test_open_via_dialog_updates_last_directory(self, qapp, tmp_path):
        from spriter.io.project_io import save as save_project
        from spriter.core.sprite import Sprite

        sprite = Sprite(8, 8)
        sprite.add_layer("L")
        sprite.add_frame()
        proj = tmp_path / "proj.spriter"
        save_project(sprite, str(proj))

        win, cfg = _make_window(tmp_path)
        try:
            with patch(
                "spriter.ui.main_window.QFileDialog.getOpenFileName",
                return_value=(str(proj), ""),
            ):
                win.open_project()
            assert Path(win._settings.last_directory) == tmp_path.resolve()
            # And persisted to disk
            from spriter.core.settings import Settings

            reloaded = Settings.load(cfg)
            assert Path(reloaded.last_directory) == tmp_path.resolve()
        finally:
            win.close()

    def test_open_via_recent_path_also_updates(self, qapp, tmp_path):
        from spriter.io.project_io import save as save_project
        from spriter.core.sprite import Sprite

        sprite = Sprite(8, 8)
        sprite.add_layer("L")
        sprite.add_frame()
        sub = tmp_path / "nested"
        sub.mkdir()
        proj = sub / "proj.spriter"
        save_project(sprite, str(proj))

        win, _ = _make_window(tmp_path)
        try:
            win.open_project(str(proj))
            assert Path(win._settings.last_directory) == sub.resolve()
        finally:
            win.close()

    def test_open_dialog_uses_current_last_directory(self, qapp, tmp_path):
        win, _ = _make_window(tmp_path)
        captured = {}

        def fake_dialog(parent, caption, directory, *args, **kwargs):
            captured["dir"] = directory
            return ("", "")

        try:
            win._settings.last_directory = str(tmp_path)
            with patch(
                "spriter.ui.main_window.QFileDialog.getOpenFileName",
                side_effect=fake_dialog,
            ):
                win.open_project()
            assert captured["dir"] == str(tmp_path)
        finally:
            win.close()

    def test_cancelled_dialog_does_not_change_last_directory(self, qapp, tmp_path):
        win, _ = _make_window(tmp_path)
        try:
            win._settings.last_directory = str(tmp_path)
            with patch(
                "spriter.ui.main_window.QFileDialog.getOpenFileName",
                return_value=("", ""),
            ):
                win.open_project()
            assert win._settings.last_directory == str(tmp_path)
        finally:
            win.close()


class TestSaveAsRemembersDir:
    def test_save_as_updates_last_directory(self, qapp, tmp_path):
        from spriter.core.sprite import Sprite

        win, _ = _make_window(tmp_path)
        try:
            win._sprite = Sprite(8, 8)
            win._sprite.add_layer("L")
            win._sprite.add_frame()
            target = tmp_path / "saved.spriter"
            with patch(
                "spriter.ui.main_window.QFileDialog.getSaveFileName",
                return_value=(str(target), ""),
            ):
                ok = win.save_as_project()
            assert ok
            assert Path(win._settings.last_directory) == tmp_path.resolve()
        finally:
            win.close()


class TestImportPngRemembersDir:
    def test_import_png_updates_last_directory(self, qapp, tmp_path):
        import numpy as np
        from PIL import Image

        sub = tmp_path / "imports"
        sub.mkdir()
        img_path = sub / "art.png"
        Image.fromarray(np.zeros((4, 4, 4), dtype=np.uint8), mode="RGBA").save(img_path)

        win, _ = _make_window(tmp_path)
        try:
            with patch(
                "spriter.ui.main_window.QFileDialog.getOpenFileName",
                return_value=(str(img_path), ""),
            ), patch(
                "spriter.ui.main_window.import_png",
                side_effect=RuntimeError("stop after remember"),
            ), patch(
                "spriter.ui.main_window.QMessageBox.critical"
            ):
                win._import_png()
            assert Path(win._settings.last_directory) == sub.resolve()
        finally:
            win.close()


class TestExportFramePngRemembersDir:
    def test_export_frame_png_updates_last_directory(self, qapp, tmp_path):
        from spriter.core.sprite import Sprite

        win, _ = _make_window(tmp_path)
        try:
            win._sprite = Sprite(8, 8)
            win._sprite.add_layer("L")
            win._sprite.add_frame()
            sub = tmp_path / "exports"
            sub.mkdir()
            target = sub / "frame.png"
            with patch(
                "spriter.ui.main_window.QFileDialog.getSaveFileName",
                return_value=(str(target), ""),
            ), patch("spriter.ui.main_window.export_frame") as mock_export:
                win._export_frame_png()
            mock_export.assert_called_once()
            assert Path(win._settings.last_directory) == sub.resolve()
        finally:
            win.close()


class TestExportAllFramesRemembersDir:
    def test_export_all_frames_updates_last_directory(self, qapp, tmp_path):
        from spriter.core.sprite import Sprite

        win, _ = _make_window(tmp_path)
        try:
            win._sprite = Sprite(8, 8)
            win._sprite.add_layer("L")
            win._sprite.add_frame()
            sub = tmp_path / "frames-out"
            sub.mkdir()
            with patch(
                "spriter.ui.main_window.QFileDialog.getExistingDirectory",
                return_value=str(sub),
            ), patch("spriter.ui.main_window.export_all_frames") as mock_export:
                win._export_all_frames_png()
            mock_export.assert_called_once()
            assert Path(win._settings.last_directory) == sub.resolve()
        finally:
            win.close()


class TestSubsequentDialogUsesRememberedDir:
    def test_export_then_import_dialogs_default_to_remembered_dir(self, qapp, tmp_path):
        """An import dialog opened after an export should default to the
        directory used for the export."""
        from spriter.core.sprite import Sprite

        win, _ = _make_window(tmp_path)
        try:
            win._sprite = Sprite(8, 8)
            win._sprite.add_layer("L")
            win._sprite.add_frame()
            sub = tmp_path / "shared"
            sub.mkdir()
            target = sub / "frame.png"

            with patch(
                "spriter.ui.main_window.QFileDialog.getSaveFileName",
                return_value=(str(target), ""),
            ), patch("spriter.ui.main_window.export_frame"):
                win._export_frame_png()

            captured = {}

            def fake_open(parent, caption, directory, *args, **kwargs):
                captured["dir"] = directory
                return ("", "")

            with patch(
                "spriter.ui.main_window.QFileDialog.getOpenFileName",
                side_effect=fake_open,
            ), patch(
                "spriter.ui.main_window.import_png",
                side_effect=RuntimeError("stop"),
            ), patch(
                "spriter.ui.main_window.QMessageBox.critical"
            ):
                win._import_png()

            assert Path(captured["dir"]) == sub.resolve()
        finally:
            win.close()
