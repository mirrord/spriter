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
    import os

    # Use offscreen rendering when no display is available (e.g. CI).
    # The real Windows/macOS/Linux platform plugins take precedence when
    # QT_QPA_PLATFORM is not set.
    os.environ.setdefault("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", ""))

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("Spriter")
    app.setOrganizationName("Spriter")

    _apply_dark_theme(app)

    from .ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def _apply_dark_theme(app) -> None:
    """Apply a dark Fusion palette to the application."""
    from PyQt6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    palette = QPalette()
    dark = QColor(45, 45, 45)
    darker = QColor(35, 35, 35)
    text = QColor(220, 220, 220)
    highlight = QColor(70, 130, 180)
    disabled = QColor(120, 120, 120)

    palette.setColor(QPalette.ColorRole.Window, dark)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, darker)
    palette.setColor(QPalette.ColorRole.AlternateBase, dark)
    palette.setColor(QPalette.ColorRole.ToolTipBase, dark)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, dark)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled)
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled
    )
    app.setPalette(palette)


if __name__ == "__main__":
    main()
