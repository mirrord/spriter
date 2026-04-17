# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Application settings / preferences (Phase 8).

All settings are persisted as a JSON file under the platform user-config
directory (``~/.config/spriter/settings.json`` on POSIX; handled via
:func:`pathlib.Path.home` for cross-platform compatibility).

Usage::

    from spriter.core.settings import Settings

    s = Settings.load()
    s.default_canvas_width = 64
    s.save()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _config_path() -> Path:
    """Return the path to the settings JSON file."""
    return Path.home() / ".config" / "spriter" / "settings.json"


class Settings:
    """Persistent application settings.

    All public attributes have sensible defaults.  Call :meth:`save` to
    persist changes and :meth:`load` (classmethod) to restore them.
    """

    # ── Canvas defaults ──────────────────────────────────────────────
    default_canvas_width: int = 32
    default_canvas_height: int = 32

    # ── Command stack ────────────────────────────────────────────────
    max_undo_depth: int = 100

    # ── Autosave ─────────────────────────────────────────────────────
    autosave_interval_ms: int = 60_000  # 0 = disabled

    # ── Visual ───────────────────────────────────────────────────────
    # Grid overlay pen colour (R, G, B, A)
    grid_color: Tuple[int, int, int, int] = (100, 100, 100, 140)
    # Transparency checker colours
    checker_light: Tuple[int, int, int] = (200, 200, 200)
    checker_dark: Tuple[int, int, int] = (150, 150, 150)
    # "dark" or "light"
    theme: str = "dark"

    # ── Recent files ─────────────────────────────────────────────────
    recent_files: List[str]
    max_recent_files: int = 10

    # ── Keybindings ──────────────────────────────────────────────────
    # Mapping from tool name → single letter shortcut
    keybindings: Dict[str, str]

    _DEFAULT_KEYBINDINGS: Dict[str, str] = {
        "pencil": "B",
        "eraser": "E",
        "line": "L",
        "rectangle": "R",
        "ellipse": "O",
        "fill": "G",
        "eyedropper": "I",
        "select": "S",
        "move": "M",
        "text": "T",
    }

    def __init__(self) -> None:
        self.default_canvas_width = 32
        self.default_canvas_height = 32
        self.max_undo_depth = 100
        self.autosave_interval_ms = 60_000
        self.grid_color = (100, 100, 100, 140)
        self.checker_light = (200, 200, 200)
        self.checker_dark = (150, 150, 150)
        self.theme = "dark"
        self.recent_files: List[str] = []
        self.max_recent_files = 10
        self.keybindings: Dict[str, str] = dict(self._DEFAULT_KEYBINDINGS)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize all settings to a JSON-serialisable dict."""
        return {
            "default_canvas_width": self.default_canvas_width,
            "default_canvas_height": self.default_canvas_height,
            "max_undo_depth": self.max_undo_depth,
            "autosave_interval_ms": self.autosave_interval_ms,
            "grid_color": list(self.grid_color),
            "checker_light": list(self.checker_light),
            "checker_dark": list(self.checker_dark),
            "theme": self.theme,
            "recent_files": list(self.recent_files),
            "max_recent_files": self.max_recent_files,
            "keybindings": dict(self.keybindings),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        """Deserialize a settings dict (unknown keys are ignored)."""
        s = cls()
        s.default_canvas_width = int(
            data.get("default_canvas_width", s.default_canvas_width)
        )
        s.default_canvas_height = int(
            data.get("default_canvas_height", s.default_canvas_height)
        )
        s.max_undo_depth = int(data.get("max_undo_depth", s.max_undo_depth))
        s.autosave_interval_ms = int(
            data.get("autosave_interval_ms", s.autosave_interval_ms)
        )
        gc = data.get("grid_color")
        if gc and len(gc) == 4:
            s.grid_color = tuple(int(v) for v in gc)  # type: ignore[assignment]
        cl = data.get("checker_light")
        if cl and len(cl) == 3:
            s.checker_light = tuple(int(v) for v in cl)  # type: ignore[assignment]
        cd = data.get("checker_dark")
        if cd and len(cd) == 3:
            s.checker_dark = tuple(int(v) for v in cd)  # type: ignore[assignment]
        s.theme = str(data.get("theme", s.theme))
        rf = data.get("recent_files")
        if isinstance(rf, list):
            s.recent_files = [str(p) for p in rf]
        s.max_recent_files = int(data.get("max_recent_files", s.max_recent_files))
        kb = data.get("keybindings")
        if isinstance(kb, dict):
            s.keybindings = {str(k): str(v) for k, v in kb.items()}
        return s

    def save(self, path: Optional[Path] = None) -> None:
        """Write settings to *path* (defaults to the standard config location).

        Args:
            path: Override the output file path.
        """
        target = path or _config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        """Load settings from *path* (defaults to the standard config location).

        Returns a default :class:`Settings` instance if the file does not exist
        or cannot be parsed.

        Args:
            path: Override the input file path.
        """
        target = path or _config_path()
        if not target.exists():
            return cls()
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except Exception:
            return cls()

    # ------------------------------------------------------------------
    # Recent-files helpers
    # ------------------------------------------------------------------

    def add_recent_file(self, path: str) -> None:
        """Prepend *path* to the recent-files list, trimming overflow.

        Args:
            path: Absolute or relative file path to record.
        """
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        self.recent_files = self.recent_files[: self.max_recent_files]
