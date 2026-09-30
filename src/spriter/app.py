# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Application entry point.

Usage::

    # From the command line (after `pip install spriter`):
    spriter

    # Or directly:
    python -m spriter
"""

from __future__ import annotations

import sys


def main() -> None:
    """Launch the Spriter GUI application."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("Spriter")
    app.setOrganizationName("Spriter")

    from .core.settings import Settings
    from .ui import theme

    settings = Settings.load()
    theme.apply_theme(app, settings.theme)

    from .ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    # Open a file passed as a command-line argument (e.g. via "Open with…"
    # or by dragging a file onto the executable).
    if len(sys.argv) > 1:
        window.open_project(sys.argv[1])

    sys.exit(app.exec())


def _apply_dark_theme(app) -> None:
    """Apply the dark theme (kept for backward compatibility)."""
    from .ui import theme

    theme.apply_theme(app, "dark")


if __name__ == "__main__":
    main()
