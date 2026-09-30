# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Application theming.

Themes bundle a :class:`~PyQt6.QtGui.QPalette` with a global Qt Style Sheet
(QSS) so the whole application can be restyled from one place.  The flagship
**ember** theme pairs a dark, blue-tinted base with an electric-blue primary
accent and a red-orange secondary accent for an eye-catching look.

Usage::

    from PyQt6.QtWidgets import QApplication
    from spriter.ui import theme

    app = QApplication([])
    theme.apply_theme(app, "ember")
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QColor, QPalette

# Ordered so the first entry is the default presented in the UI.
THEME_NAMES: list[str] = ["ember", "dark", "light"]
DEFAULT_THEME: str = "ember"


@dataclass(frozen=True)
class Palette:
    """A flat set of colours a theme is built from (hex ``#rrggbb`` strings)."""

    window: str  # Main window / panel background
    base: str  # Input / canvas-adjacent background (usually darker)
    alt_base: str  # Alternating rows / secondary surface
    text: str  # Primary foreground text
    dim_text: str  # Disabled / secondary text
    button: str  # Button face
    border: str  # Subtle panel / control borders
    accent: str  # Primary accent (blue): selection, focus, headers
    accent_alt: str  # Secondary accent (red-orange): active tool, hover, warn
    bright_text: str  # Error / warning foreground


_EMBER = Palette(
    window="#1b2029",
    base="#12151b",
    alt_base="#232a36",
    text="#e6eaf0",
    dim_text="#6c7889",
    button="#232a36",
    border="#313c4d",
    accent="#3b82f6",
    accent_alt="#f2521b",
    bright_text="#ff5a47",
)

_DARK = Palette(
    window="#2d2d2d",
    base="#232323",
    alt_base="#2d2d2d",
    text="#dcdcdc",
    dim_text="#787878",
    button="#2d2d2d",
    border="#3c3c3c",
    accent="#4682b4",
    accent_alt="#c85a32",
    bright_text="#ff5050",
)

_LIGHT = Palette(
    window="#f2f2f2",
    base="#ffffff",
    alt_base="#e9e9e9",
    text="#1e1e1e",
    dim_text="#9a9a9a",
    button="#e4e4e4",
    border="#c4c4c4",
    accent="#2563eb",
    accent_alt="#e04a1a",
    bright_text="#c81e1e",
)

_PALETTES: dict[str, Palette] = {
    "ember": _EMBER,
    "dark": _DARK,
    "light": _LIGHT,
}


def _colors(name: str) -> Palette:
    return _PALETTES.get(name, _EMBER)


def build_palette(name: str) -> QPalette:
    """Return a :class:`QPalette` for the named theme.

    Args:
        name: One of :data:`THEME_NAMES`; unknown names fall back to *ember*.
    """
    c = _colors(name)
    p = QPalette()
    window = QColor(c.window)
    base = QColor(c.base)
    alt = QColor(c.alt_base)
    text = QColor(c.text)
    dim = QColor(c.dim_text)
    button = QColor(c.button)
    accent = QColor(c.accent)

    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, alt)
    p.setColor(QPalette.ColorRole.ToolTipBase, window)
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, button)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor(c.bright_text))
    p.setColor(QPalette.ColorRole.Link, accent)
    p.setColor(QPalette.ColorRole.Highlight, accent)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))

    disabled = QPalette.ColorGroup.Disabled
    p.setColor(disabled, QPalette.ColorRole.Text, dim)
    p.setColor(disabled, QPalette.ColorRole.ButtonText, dim)
    p.setColor(disabled, QPalette.ColorRole.WindowText, dim)
    return p


def build_stylesheet(name: str) -> str:
    """Return a global QSS stylesheet for the named theme.

    The *ember* theme returns a rich stylesheet that restyles menus, docks,
    buttons, sliders, scrollbars and tabs with the blue / red-orange accents.
    The plainer *dark* and *light* themes return an empty string and rely on
    their palette alone.

    Args:
        name: One of :data:`THEME_NAMES`.
    """
    if name != "ember":
        return ""
    c = _EMBER
    return _EMBER_QSS.format(
        window=c.window,
        base=c.base,
        alt_base=c.alt_base,
        text=c.text,
        dim_text=c.dim_text,
        button=c.button,
        border=c.border,
        accent=c.accent,
        accent_alt=c.accent_alt,
    )


def apply_theme(app, name: str) -> None:
    """Apply the named theme's style, palette and stylesheet to *app*.

    Args:
        app: The :class:`~PyQt6.QtWidgets.QApplication` instance.
        name: One of :data:`THEME_NAMES`; unknown names fall back to *ember*.
    """
    app.setStyle("Fusion")
    app.setPalette(build_palette(name))
    app.setStyleSheet(build_stylesheet(name))


# Placeholders are filled by :func:`build_stylesheet`.  Literal braces in the
# QSS are doubled so ``str.format`` leaves them intact.
_EMBER_QSS = """
QWidget {{
    background-color: {window};
    color: {text};
}}
QMainWindow::separator {{
    background-color: {accent};
    width: 2px;
    height: 2px;
}}
QMainWindow::separator:hover {{
    background-color: {accent_alt};
}}

/* Menu bar: blue selection, orange hover underline feel */
QMenuBar {{
    background-color: {base};
    border-bottom: 2px solid {accent};
    padding: 2px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{
    background-color: {accent};
    color: #ffffff;
}}
QMenuBar::item:pressed {{
    background-color: {accent_alt};
    color: #ffffff;
}}
QMenu {{
    background-color: {window};
    border: 1px solid {border};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 24px 5px 20px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {accent};
    color: #ffffff;
}}
QMenu::separator {{
    height: 1px;
    background: {border};
    margin: 4px 8px;
}}

/* Dock panels: gradient title bar with an orange left accent */
QDockWidget {{
    titlebar-close-icon: none;
    border: 1px solid {border};
}}
QDockWidget::title {{
    text-align: left;
    padding: 6px 10px;
    border-left: 3px solid {accent_alt};
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {accent}, stop:0.6 {alt_base}, stop:1 {window});
    color: #ffffff;
    font-weight: bold;
}}

/* Buttons: rounded, blue border, orange on hover / press */
QPushButton, QToolButton {{
    background-color: {button};
    border: 1px solid {border};
    border-radius: 5px;
    padding: 4px 10px;
}}
QPushButton:hover, QToolButton:hover {{
    border-color: {accent};
    background-color: {alt_base};
}}
QPushButton:pressed, QToolButton:pressed {{
    background-color: {accent_alt};
    border-color: {accent_alt};
    color: #ffffff;
}}
QToolButton:checked, QPushButton:checked {{
    background-color: {accent_alt};
    border-color: {accent_alt};
    color: #ffffff;
}}
QPushButton:disabled, QToolButton:disabled {{
    color: {dim_text};
    border-color: {border};
}}

/* Text entry & combos */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background-color: {base};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: {accent};
    selection-color: #ffffff;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {accent};
}}
QComboBox QAbstractItemView {{
    background-color: {window};
    border: 1px solid {border};
    selection-background-color: {accent};
    selection-color: #ffffff;
}}

/* Lists (layers, recent files) */
QListWidget, QTreeWidget, QTableWidget {{
    background-color: {base};
    border: 1px solid {border};
    border-radius: 4px;
    outline: none;
}}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {accent};
    color: #ffffff;
}}
QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {alt_base};
}}

/* Tabs (Preferences) */
QTabWidget::pane {{
    border: 1px solid {border};
    border-radius: 4px;
    top: -1px;
}}
QTabBar::tab {{
    background: {base};
    padding: 6px 14px;
    border: 1px solid {border};
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}}
QTabBar::tab:selected {{
    background: {alt_base};
    border-bottom: 3px solid {accent};
    color: #ffffff;
}}
QTabBar::tab:hover:!selected {{
    border-bottom: 3px solid {accent_alt};
}}

/* Sliders: blue groove fill, orange handle */
QSlider::groove:horizontal {{
    height: 5px;
    border-radius: 2px;
    background: {base};
}}
QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {accent_alt};
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: #ff7a4a;
}}
QSlider::groove:vertical {{
    width: 5px;
    border-radius: 2px;
    background: {base};
}}
QSlider::sub-page:vertical {{
    background: {accent};
    border-radius: 2px;
}}
QSlider::handle:vertical {{
    background: {accent_alt};
    height: 14px;
    margin: 0 -6px;
    border-radius: 7px;
}}

/* Scrollbars: slim, blue-gray, orange on hover */
QScrollBar:vertical {{
    background: {base};
    width: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {border};
    min-height: 24px;
    border-radius: 5px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{
    background: {accent_alt};
}}
QScrollBar:horizontal {{
    background: {base};
    height: 12px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {border};
    min-width: 24px;
    border-radius: 5px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {accent_alt};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0;
    height: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}

/* Progress (diffusion) */
QProgressBar {{
    border: 1px solid {border};
    border-radius: 4px;
    background: {base};
    text-align: center;
}}
QProgressBar::chunk {{
    border-radius: 3px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {accent}, stop:1 {accent_alt});
}}

/* Group boxes */
QGroupBox {{
    border: 1px solid {border};
    border-radius: 5px;
    margin-top: 8px;
    padding-top: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {accent};
}}

/* Status bar: thin orange top accent */
QStatusBar {{
    background: {base};
    border-top: 2px solid {accent_alt};
}}
QStatusBar QLabel {{
    background: transparent;
}}

/* Tooltips */
QToolTip {{
    background-color: {window};
    color: {text};
    border: 1px solid {accent};
    border-radius: 4px;
    padding: 3px 6px;
}}

/* Checkboxes / radios accent */
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {accent};
    border: 1px solid {accent};
}}
"""
