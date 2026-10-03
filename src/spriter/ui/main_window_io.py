# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Export / import / palette / clipboard slots for :class:`MainWindow`.

Grouped into :class:`IOMixin` to keep the core window small.  Heavy codec
imports (Pillow, spritesheet helpers) stay lazy inside the slots.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

from ..core.palette import Palette
from ..core.sprite import Sprite
from ..io.gif_io import export_gif, import_gif
from ..io.png_io import export_all_frames, export_frame, import_png
from ..io.spritesheet import (
    export_atlas,
    export_sheet,
    import_sheet,
    import_sheet_auto,
)
from .main_window_base import _MainWindowProtocol


class IOMixin(_MainWindowProtocol):
    """Menu slots for exporting, importing, palettes and the clipboard."""

    # ------------------------------------------------------------------
    # Phase 7: Export actions
    # ------------------------------------------------------------------

    def _export_frame_png(self) -> None:
        if self._sprite is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Frame as PNG", self._dialog_dir("save"), "PNG Images (*.png)"
        )
        if not path:
            return
        self._remember_path(path, "save")
        fi = self._canvas.active_frame if self._canvas else 0
        try:
            export_frame(self._sprite, fi, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_all_frames_png(self) -> None:
        if self._sprite is None:
            return
        dir_path = QFileDialog.getExistingDirectory(
            self, "Export All Frames — Choose Folder", self._dialog_dir("save")
        )
        if not dir_path:
            return
        self._remember_directory(dir_path, "save")
        try:
            export_all_frames(self._sprite, dir_path)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_gif(self) -> None:
        if self._sprite is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Animated GIF", self._dialog_dir("save"), "GIF Images (*.gif)"
        )
        if not path:
            return
        self._remember_path(path, "save")
        try:
            export_gif(self._sprite, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_sheet(self) -> None:
        if self._sprite is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Sprite Sheet", self._dialog_dir("save"), "PNG Images (*.png)"
        )
        if not path:
            return
        self._remember_path(path, "save")
        try:
            export_sheet(self._sprite, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_atlas(self) -> None:
        if self._sprite is None:
            return
        sheet_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Sprite Sheet (image)",
            self._dialog_dir("save"),
            "PNG Images (*.png)",
        )
        if not sheet_path:
            return
        self._remember_path(sheet_path, "save")
        atlas_path, _ = QFileDialog.getSaveFileName(
            self, "Export Atlas (JSON)", self._dialog_dir("save"), "JSON files (*.json)"
        )
        if not atlas_path:
            return
        self._remember_path(atlas_path, "save")
        try:
            export_atlas(self._sprite, sheet_path, atlas_path)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_ico(self) -> None:
        if self._sprite is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export as ICO", self._dialog_dir("save"), "Icon files (*.ico)"
        )
        if not path:
            return
        self._remember_path(path, "save")
        fi = self._canvas.active_frame if self._canvas else 0
        try:
            from PIL import Image

            from ..core.compositor import composite_frame

            composite = composite_frame(self._sprite, fi)
            img = Image.fromarray(composite, mode="RGBA")
            img.save(path, format="ICO")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # Phase 7: Import actions
    # ------------------------------------------------------------------

    def _import_png(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import PNG as Sprite",
            self._dialog_dir("open"),
            "Images (*.png *.bmp *.jpg *.jpeg *.webp)",
        )
        if not path:
            return
        self._remember_path(path, "open")
        try:
            sprite = import_png(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
            return
        self._load_imported_sprite(sprite, Path(path).name)

    def _import_gif(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import GIF as Sprite",
            self._dialog_dir("open"),
            "Animated GIF (*.gif)",
        )
        if not path:
            return
        self._remember_path(path, "open")
        try:
            sprite = import_gif(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
            return
        self._load_imported_sprite(sprite, Path(path).name)

    def _import_sheet(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Sprite Sheet",
            self._dialog_dir("open"),
            "Images (*.png *.bmp *.jpg *.jpeg)",
        )
        if not path:
            return
        self._remember_path(path, "open")
        # Multi-row sheets can be split so each row becomes its own animation.
        split_rows = (
            QMessageBox.question(
                self,
                "Import Sheet",
                "Does each row contain a separate animation?\n\n"
                "Choose Yes to split each row into its own animation timeline; "
                "choose No to import all frames as a single animation.",
            )
            == QMessageBox.StandardButton.Yes
        )
        # Sheets with irregular frame spacing use contour-based auto-detection
        # instead of a fixed grid.
        inconsistent = (
            QMessageBox.question(
                self,
                "Import Sheet",
                "Does this sheet have inconsistent spacing between frames?\n\n"
                "Choose Yes to auto-detect each frame and centre it in a "
                "uniform cell; choose No to slice on a fixed grid.",
            )
            == QMessageBox.StandardButton.Yes
        )
        if inconsistent:
            try:
                sprite = import_sheet_auto(path, split_rows=split_rows)
            except Exception as exc:
                QMessageBox.critical(self, "Import Error", str(exc))
                return
            self._maybe_handle_background(sprite, path)
            self._load_imported_sprite(sprite, Path(path).name)
            return
        # Best-effort dimension estimation to pre-populate the dialogs.
        est_w, est_h, est_pad = 16, 16, 0
        try:
            from ..io.spritesheet import estimate_sheet_layout

            est = estimate_sheet_layout(path)
            est_w, est_h, est_pad = est.frame_width, est.frame_height, est.padding
        except Exception:
            pass
        fw, ok1 = QInputDialog.getInt(
            self, "Import Sheet", "Frame width (px):", est_w, 1, 4096
        )
        if not ok1:
            return
        fh, ok2 = QInputDialog.getInt(
            self, "Import Sheet", "Frame height (px):", est_h, 1, 4096
        )
        if not ok2:
            return
        pad, ok3 = QInputDialog.getInt(
            self, "Import Sheet", "Padding (px):", est_pad, 0, 64
        )
        if not ok3:
            return
        try:
            sprite = import_sheet(path, fw, fh, padding=pad, split_rows=split_rows)
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))
            return
        self._maybe_handle_background(sprite, path)
        self._load_imported_sprite(sprite, Path(path).name)

    def _maybe_handle_background(self, sprite: Sprite, source_path: str) -> None:
        """Detect a sheet background colour and prompt the user how to handle it."""
        from ..io.spritesheet import (
            detect_background_color,
            remove_background,
            split_background,
        )

        try:
            color = detect_background_color(source_path)
        except Exception:
            color = None
        if color is None:
            return
        r, g, b, _a = color
        options = [
            "Remove background (make transparent)",
            "Separate background & foreground layers",
            "Import as-is",
        ]
        choice, ok = QInputDialog.getItem(
            self,
            "Background Detected",
            f"A background colour (RGB {r}, {g}, {b}) was detected.\n"
            "How would you like to handle it?",
            options,
            0,
            False,
        )
        if not ok:
            return
        if choice == options[0]:
            remove_background(sprite, color)
        elif choice == options[1]:
            split_background(sprite, color)
        # options[2] ("Import as-is") leaves the sprite unchanged.

    # ------------------------------------------------------------------
    # Palette import / export
    # ------------------------------------------------------------------

    def _import_palette(self) -> None:
        """Import a palette file (.pal / .gpl / .hex / .txt) into the colour picker."""
        if self._color_picker is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Palette",
            self._dialog_dir("open"),
            "Palette files (*.pal *.gpl *.hex *.txt);;All files (*)",
        )
        if not path:
            return
        self._remember_path(path, "open")
        suffix = Path(path).suffix.lower()
        try:
            if suffix == ".gpl":
                palette = Palette.from_gpl(path)
            elif suffix in (".hex", ".txt"):
                palette = Palette.from_hex_list(path)
            else:
                palette = Palette.from_jasc(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Import Error", f"Could not load palette:\n{exc}"
            )
            return
        self._color_picker.load_palette(list(palette))

    def _export_palette(self) -> None:
        """Export the current palette grid to a file (.pal / .gpl / .hex)."""
        if self._color_picker is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Palette",
            self._dialog_dir("save"),
            "JASC-PAL (*.pal);;GIMP GPL (*.gpl);;Hex list (*.hex)",
        )
        if not path:
            return
        self._remember_path(path, "save")
        suffix = Path(path).suffix.lower()
        palette = Palette(self._color_picker._palette_colors)
        try:
            if suffix == ".gpl":
                palette.to_gpl(path)
            elif suffix == ".hex":
                palette.to_hex_list(path)
            else:
                palette.to_jasc(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Export Error", f"Could not save palette:\n{exc}"
            )

    # ------------------------------------------------------------------
    # Phase 7: Copy / Paste
    # ------------------------------------------------------------------

    def _copy_selection(self) -> None:
        """Copy the active cel (or selection) to the clipboard as a PNG image."""
        if self._sprite is None:
            return

        import numpy as np
        from PyQt6.QtGui import QImage
        from PyQt6.QtWidgets import QApplication

        from ..core.compositor import composite_frame

        fi = self._canvas.active_frame if self._canvas else 0
        composite = composite_frame(self._sprite, fi)

        if self._sprite.selection_mask is not None:
            # Zero out pixels outside the selection for copy.
            masked = composite.copy()
            masked[~self._sprite.selection_mask] = 0
            composite = masked

        h, w = composite.shape[:2]
        arr = np.ascontiguousarray(composite)
        qi = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        clipboard = QApplication.clipboard()
        assert clipboard is not None
        clipboard.setImage(qi)

    def _paste_clipboard(self) -> None:
        """Paste clipboard image onto the active layer / frame."""
        if self._sprite is None:
            return
        import numpy as np
        from PyQt6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        qi = clipboard.image() if clipboard else None
        if qi is None or qi.isNull():
            return
        # Convert QImage to numpy RGBA.
        qi = qi.convertToFormat(qi.Format.Format_RGBA8888)
        w, h = qi.width(), qi.height()
        ptr = qi.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4)).copy()  # type: ignore[call-overload]

        # Resize to canvas size if needed.
        if w != self._sprite.width or h != self._sprite.height:
            from PIL import Image

            img = Image.fromarray(arr, mode="RGBA")
            img = img.resize(
                (self._sprite.width, self._sprite.height), Image.Resampling.NEAREST
            )
            arr = np.array(img, dtype=np.uint8)

        li = self._layers_panel.active_layer if self._layers_panel else 0
        fi = self._canvas.active_frame if self._canvas else 0
        self._sprite.set_cel_pixels(li, fi, arr)
        if self._canvas:
            self._canvas.invalidate_cache()
        self._unsaved = True
