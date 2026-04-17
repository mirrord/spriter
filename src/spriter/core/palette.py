# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Color palette management.

Supports JASC-PAL, GIMP GPL, and plain hex-list formats.
Up to 256 RGBA colors are stored.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple, Union

# A color is stored as a 4-tuple (R, G, B, A) with values 0–255.
Color = Tuple[int, int, int, int]

_MAX_COLORS = 256


def _clamp(value: int) -> int:
    return max(0, min(255, int(value)))


class Palette:
    """An ordered collection of up to 256 RGBA colors.

    Args:
        colors: Initial list of (R, G, B, A) tuples.
    """

    def __init__(self, colors: Optional[Sequence[Color]] = None) -> None:  # type: ignore[name-defined]
        self._colors: List[Color] = []
        if colors:
            for c in colors:
                self.add(c)

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._colors)

    def __getitem__(self, index: int) -> Color:
        return self._colors[index]

    def __setitem__(self, index: int, color: Color) -> None:
        self._colors[index] = _validate_color(color)

    def __iter__(self):
        return iter(self._colors)

    def __repr__(self) -> str:
        return f"Palette({len(self._colors)} colors)"

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, color: Color) -> int:
        """Append a color and return its index.

        Raises:
            ValueError: If the palette is already full (256 colors).
        """
        if len(self._colors) >= _MAX_COLORS:
            raise ValueError("Palette is full (256 colors maximum)")
        validated = _validate_color(color)
        self._colors.append(validated)
        return len(self._colors) - 1

    def remove(self, index: int) -> None:
        """Remove the color at *index*."""
        del self._colors[index]

    def move(self, from_index: int, to_index: int) -> None:
        """Reorder: move the color at *from_index* to *to_index*."""
        color = self._colors.pop(from_index)
        self._colors.insert(to_index, color)

    def sort_by_hue(self) -> None:
        """Sort colors by HSV hue (in-place)."""
        self._colors.sort(key=lambda c: _rgb_to_hsv(c[0], c[1], c[2])[0])

    # ------------------------------------------------------------------
    # JASC-PAL (.pal)
    # ------------------------------------------------------------------

    @classmethod
    def from_jasc(cls, path: Union[str, Path]) -> "Palette":
        """Load a JASC-PAL palette file.

        Args:
            path: Path to the .pal file.

        Returns:
            A new Palette instance.
        """
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if (
            len(lines) < 3
            or lines[0].strip() != "JASC-PAL"
            or lines[1].strip() != "0100"
        ):
            raise ValueError("Not a valid JASC-PAL file")
        count = int(lines[2].strip())
        colors: List[Color] = []
        for line in lines[3 : 3 + count]:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            colors.append((_clamp(r), _clamp(g), _clamp(b), 255))
        return cls(colors)

    def to_jasc(self, path: Union[str, Path]) -> None:
        """Save palette as JASC-PAL format.

        Args:
            path: Destination .pal file path.
        """
        lines = ["JASC-PAL", "0100", str(len(self._colors))]
        for r, g, b, _a in self._colors:
            lines.append(f"{r} {g} {b}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # GIMP GPL (.gpl)
    # ------------------------------------------------------------------

    @classmethod
    def from_gpl(cls, path: Union[str, Path]) -> "Palette":
        """Load a GIMP GPL palette file.

        Args:
            path: Path to the .gpl file.

        Returns:
            A new Palette instance.
        """
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines or not lines[0].startswith("GIMP Palette"):
            raise ValueError("Not a valid GIMP GPL file")
        colors: List[Color] = []
        for line in lines[1:]:
            line = line.strip()
            if (
                not line
                or line.startswith("#")
                or line.startswith("Name:")
                or line.startswith("Columns:")
            ):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                colors.append((_clamp(r), _clamp(g), _clamp(b), 255))
            except ValueError:
                continue
        return cls(colors)

    def to_gpl(self, path: Union[str, Path], name: str = "Spriter Palette") -> None:
        """Save palette as GIMP GPL format.

        Args:
            path: Destination .gpl file path.
            name: Palette name written into the file header.
        """
        lines = [
            "GIMP Palette",
            f"Name: {name}",
            f"Columns: 16",
            "#",
        ]
        for i, (r, g, b, _a) in enumerate(self._colors):
            label = f"Color {i}"
            lines.append(f"{r:3d} {g:3d} {b:3d}\t{label}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Hex list (.hex / .txt)
    # ------------------------------------------------------------------

    @classmethod
    def from_hex_list(cls, path: Union[str, Path]) -> "Palette":
        """Load a newline-separated hex color list (e.g. ``FF0000`` or ``#FF0000``).

        Args:
            path: Path to the hex list file.

        Returns:
            A new Palette instance.
        """
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        colors: List[Color] = []
        for line in lines:
            line = line.strip().lstrip("#")
            if not line or line.startswith(";") or line.startswith("//"):
                continue
            # Accept RRGGBB or RRGGBBAA
            match = re.fullmatch(r"([0-9a-fA-F]{6})([0-9a-fA-F]{2})?", line)
            if not match:
                continue
            r = int(line[0:2], 16)
            g = int(line[2:4], 16)
            b = int(line[4:6], 16)
            a = int(line[6:8], 16) if len(line) >= 8 else 255
            colors.append((_clamp(r), _clamp(g), _clamp(b), _clamp(a)))
        return cls(colors)

    def to_hex_list(self, path: Union[str, Path]) -> None:
        """Save palette as a newline-separated hex list (RRGGBBAA).

        Args:
            path: Destination file path.
        """
        lines = [f"{r:02X}{g:02X}{b:02X}{a:02X}" for r, g, b, a in self._colors]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _validate_color(color: Color) -> Color:
    if len(color) == 3:
        r, g, b = color  # type: ignore[misc]
        color = (_clamp(r), _clamp(g), _clamp(b), 255)
    elif len(color) == 4:
        r, g, b, a = color
        color = (_clamp(r), _clamp(g), _clamp(b), _clamp(a))
    else:
        raise ValueError(f"color must be (R,G,B) or (R,G,B,A), got {color!r}")
    return color  # type: ignore[return-value]


def _rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Return (hue 0–360, saturation 0–1, value 0–1)."""
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    cmax = max(rf, gf, bf)
    cmin = min(rf, gf, bf)
    delta = cmax - cmin
    if delta == 0:
        h = 0.0
    elif cmax == rf:
        h = 60.0 * (((gf - bf) / delta) % 6)
    elif cmax == gf:
        h = 60.0 * (((bf - rf) / delta) + 2)
    else:
        h = 60.0 * (((rf - gf) / delta) + 4)
    s = 0.0 if cmax == 0 else delta / cmax
    v = cmax
    return h, s, v


from typing import Optional  # noqa: E402 (needed for forward reference in __init__)
