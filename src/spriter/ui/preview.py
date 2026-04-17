# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Real-time animation preview window.

:class:`PreviewWindow` shows a looping playback of the sprite's animation at
the correct per-frame speed.  It is opened as a floating tool window from the
main menu.

Controls
--------
* **Play / Pause** — toggle playback
* **◀ Step** / **Step ▶** — advance or rewind one frame
* **Zoom** buttons — 1×, 2×, 4×
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.sprite import Sprite


class _PreviewCanvas(QWidget):
    """Inner widget that renders the composited frame image."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._zoom: int = 2
        self.setMinimumSize(32, 32)

    def set_image(self, rgba: np.ndarray, zoom: int = 2) -> None:
        """Update the displayed image.

        Args:
            rgba: RGBA uint8 array of shape ``(H, W, 4)``.
            zoom: Integer zoom factor (1, 2, or 4).
        """
        self._zoom = zoom
        arr = np.ascontiguousarray(rgba)
        h, w = arr.shape[:2]
        self._image = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        self.setFixedSize(w * zoom, h * zoom)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        if self._image is not None:
            w = self._image.width() * self._zoom
            h = self._image.height() * self._zoom
            scaled = self._image.scaled(
                w,
                h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            painter.drawImage(0, 0, scaled)
        painter.end()


class PreviewWindow(QWidget):
    """Floating animation preview window.

    Args:
        sprite: The sprite to preview.
        parent: Optional Qt parent.
    """

    #: Emitted when the active frame changes during playback or stepping.
    frame_changed = pyqtSignal(int)

    def __init__(
        self,
        sprite: Sprite,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("Preview")
        self._sprite = sprite
        self._current_frame: int = 0
        self._playing: bool = False
        self._zoom: int = 2
        # Ping-pong direction: +1 = forward, -1 = backward.
        self._pp_direction: int = 1

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

        self._build_ui()
        self._render_frame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_sprite(self, sprite: Sprite) -> None:
        """Swap the displayed sprite (e.g. after new_project).

        Args:
            sprite: The new sprite to display.
        """
        self._sprite = sprite
        self._current_frame = 0
        self._pp_direction = 1
        self._render_frame()

    @property
    def current_frame(self) -> int:
        """Index of the frame currently shown."""
        return self._current_frame

    @property
    def is_playing(self) -> bool:
        """True while the animation timer is running."""
        return self._playing

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Canvas.
        self._canvas = _PreviewCanvas()
        layout.addWidget(self._canvas, alignment=Qt.AlignmentFlag.AlignCenter)

        # Frame label.
        self._frame_label = QLabel("Frame 1 / 1")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._frame_label)

        # Controls.
        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)

        self._btn_step_back = QPushButton("◀")
        self._btn_step_back.setFixedWidth(32)
        self._btn_step_back.clicked.connect(self.step_backward)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(48)
        self._btn_play.clicked.connect(self.toggle_play)

        self._btn_step_fwd = QPushButton("▶|")
        self._btn_step_fwd.setFixedWidth(32)
        self._btn_step_fwd.clicked.connect(self.step_forward)

        ctrl.addWidget(self._btn_step_back)
        ctrl.addWidget(self._btn_play)
        ctrl.addWidget(self._btn_step_fwd)
        ctrl.addStretch()

        # Zoom buttons.
        for factor in (1, 2, 4):
            btn = QPushButton(f"{factor}×")
            btn.setFixedWidth(30)
            btn.clicked.connect(lambda _, z=factor: self._set_zoom(z))
            ctrl.addWidget(btn)

        layout.addLayout(ctrl)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def toggle_play(self) -> None:
        """Start or pause playback."""
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        """Start playback."""
        if self._sprite.frame_count <= 1:
            return
        self._playing = True
        self._btn_play.setText("⏸")
        self._schedule_next()

    def pause(self) -> None:
        """Pause playback."""
        self._playing = False
        self._timer.stop()
        self._btn_play.setText("▶")

    def step_forward(self) -> None:
        """Advance one frame."""
        self._current_frame = (self._current_frame + 1) % max(
            1, self._sprite.frame_count
        )
        self._render_frame()
        self.frame_changed.emit(self._current_frame)

    def step_backward(self) -> None:
        """Rewind one frame."""
        total = max(1, self._sprite.frame_count)
        self._current_frame = (self._current_frame - 1) % total
        self._render_frame()
        self.frame_changed.emit(self._current_frame)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        if not self._playing:
            return
        dur = self._sprite.animation.get_frame_duration_ms(
            self._sprite, self._current_frame
        )
        self._timer.start(dur)

    def _advance_frame(self) -> None:
        if not self._playing:
            return
        anim = self._sprite.animation
        total = self._sprite.frame_count
        from ..core.animation import LoopMode

        if anim.loop_mode == LoopMode.PING_PONG and total > 1:
            next_f = self._current_frame + self._pp_direction
            if next_f >= total:
                self._pp_direction = -1
                next_f = total - 2
            elif next_f < 0:
                self._pp_direction = 1
                next_f = 1
            self._current_frame = max(0, min(next_f, total - 1))
        else:
            self._current_frame = anim.next_frame(self._current_frame, total)

        self._render_frame()
        self.frame_changed.emit(self._current_frame)
        self._schedule_next()

    def _render_frame(self) -> None:
        if self._sprite.frame_count == 0 or self._sprite.layer_count == 0:
            return
        fi = min(self._current_frame, self._sprite.frame_count - 1)
        from ..core.compositor import composite_frame

        composite = composite_frame(self._sprite, fi)
        self._canvas.set_image(composite, self._zoom)
        self._frame_label.setText(f"Frame {fi + 1} / {self._sprite.frame_count}")

    def _set_zoom(self, zoom: int) -> None:
        self._zoom = zoom
        self._render_frame()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.pause()
        super().closeEvent(event)
