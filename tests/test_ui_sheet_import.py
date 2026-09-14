# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""UI tests for the inconsistent-spacing sprite-sheet import prompt."""

from unittest.mock import patch

from PyQt6.QtWidgets import QMessageBox


def _small_sprite():
    from spriter.core.sprite import Sprite

    sprite = Sprite(8, 8)
    sprite.add_layer("Background")
    sprite.add_frame()
    return sprite


class TestImportSheetPrompt:
    def test_yes_uses_auto_detection(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        sprite = _small_sprite()

        with patch(
            "spriter.ui.main_window.QFileDialog.getOpenFileName",
            return_value=("sheet.png", ""),
        ), patch(
            "spriter.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ), patch(
            "spriter.ui.main_window.import_sheet_auto",
            return_value=sprite,
        ) as mock_auto, patch(
            "spriter.ui.main_window.import_sheet"
        ) as mock_grid, patch("spriter.ui.main_window.MainWindow._rebuild_ui"):
            win._import_sheet()

        mock_auto.assert_called_once_with("sheet.png")
        mock_grid.assert_not_called()
        assert win._sprite is sprite
        win._unsaved = False
        win.close()

    def test_no_uses_fixed_grid(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        sprite = _small_sprite()

        with patch(
            "spriter.ui.main_window.QFileDialog.getOpenFileName",
            return_value=("sheet.png", ""),
        ), patch(
            "spriter.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ), patch(
            "spriter.ui.main_window.QInputDialog.getInt",
            side_effect=[(8, True), (8, True), (0, True)],
        ), patch("spriter.ui.main_window.import_sheet_auto") as mock_auto, patch(
            "spriter.ui.main_window.import_sheet",
            return_value=sprite,
        ) as mock_grid, patch("spriter.ui.main_window.MainWindow._rebuild_ui"):
            win._import_sheet()

        mock_auto.assert_not_called()
        mock_grid.assert_called_once()
        assert win._sprite is sprite
        win._unsaved = False
        win.close()

    def test_auto_import_error_shows_dialog(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        original = win._sprite

        with patch(
            "spriter.ui.main_window.QFileDialog.getOpenFileName",
            return_value=("sheet.png", ""),
        ), patch(
            "spriter.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ), patch(
            "spriter.ui.main_window.import_sheet_auto",
            side_effect=ValueError("no frames"),
        ), patch("spriter.ui.main_window.QMessageBox.critical") as mock_critical:
            win._import_sheet()

        mock_critical.assert_called_once()
        assert win._sprite is original
        win.close()
