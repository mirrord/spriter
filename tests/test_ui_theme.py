# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the application theming system (blue / red-orange ember theme)."""

from __future__ import annotations

from spriter.ui import theme


class TestThemeRegistry:
    def test_theme_names_and_default(self):
        assert theme.DEFAULT_THEME == "ember"
        for name in ("ember", "dark", "light"):
            assert name in theme.THEME_NAMES

    def test_build_palette_returns_qpalette(self):
        from PyQt6.QtGui import QPalette

        for name in theme.THEME_NAMES:
            assert isinstance(theme.build_palette(name), QPalette)

    def test_unknown_theme_falls_back_to_ember(self):
        # Fallback palette/stylesheet match the ember theme.
        assert theme.build_stylesheet("does-not-exist") == ""
        pal = theme.build_palette("does-not-exist")
        ember = theme.build_palette("ember")
        from PyQt6.QtGui import QPalette

        role = QPalette.ColorRole.Highlight
        assert pal.color(role).name() == ember.color(role).name()

    def test_ember_stylesheet_has_accents(self):
        qss = theme.build_stylesheet("ember")
        assert qss  # non-empty
        # Blue primary + red-orange secondary accent hex codes are present.
        assert "#3b82f6" in qss
        assert "#f2521b" in qss

    def test_plain_themes_have_empty_stylesheet(self):
        assert theme.build_stylesheet("dark") == ""
        assert theme.build_stylesheet("light") == ""

    def test_ember_highlight_is_blue(self):
        from PyQt6.QtGui import QPalette

        pal = theme.build_palette("ember")
        assert pal.color(QPalette.ColorRole.Highlight).name() == "#3b82f6"


class TestApplyTheme:
    def test_apply_theme_sets_palette_and_stylesheet(self, qapp):
        theme.apply_theme(qapp, "ember")
        assert qapp.styleSheet()  # ember sets a non-empty stylesheet
        theme.apply_theme(qapp, "dark")
        assert qapp.styleSheet() == ""

    def test_apply_every_theme_does_not_raise(self, qapp):
        for name in theme.THEME_NAMES:
            theme.apply_theme(qapp, name)


class TestPreferencesIntegration:
    def test_theme_combo_includes_ember(self, qapp):
        from spriter.core.settings import Settings
        from spriter.ui.preferences import PreferencesDialog

        dlg = PreferencesDialog(Settings())
        items = [dlg._theme.itemText(i) for i in range(dlg._theme.count())]
        assert "ember" in items
        assert "dark" in items
        assert "light" in items
