# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Shared pytest fixtures for all test modules.

Sets up a session-scoped :class:`~PyQt6.QtWidgets.QApplication` so that widget
tests (Phases 3/4) can instantiate UI components without a visible display.
``QT_QPA_PLATFORM=offscreen`` is applied before any Qt code is imported so that
running under headless environments (CI, virtualised) works without a real
X/Wayland/Win32 display server.
"""

from __future__ import annotations

import os

import pytest

# Must be set before PyQt6 is imported the first time.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication for widget smoke tests."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
