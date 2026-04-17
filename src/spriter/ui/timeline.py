# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Animation timeline panel widget.

:class:`TimelinePanel` displays all frames as a horizontal strip of
clickable cells.  It sits in a dock at the bottom of the main window and
coordinates frame navigation with the canvas and preview widgets.

Signals
-------
frame_selected(int)
    Emitted when the user clicks a frame cell; carries the frame index.
frame_duration_changed(int, int)
    Emitted after the user edits a frame's duration; carries
    ``(frame_index, new_duration_ms)``.
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..commands.base import CommandStack
from ..commands.frame_ops import (
    AddFrameCommand,
    DuplicateFrameCommand,
    RemoveFrameCommand,
)
from ..core.sprite import Sprite


# ---------------------------------------------------------------------------
# Frame cell widget
# ---------------------------------------------------------------------------


class _FrameCell(QWidget):
    """A single clickable frame cell in the timeline strip.

    Args:
        frame_index: The frame this cell represents.
        duration_ms: Display duration of the frame in milliseconds.
        active: Whether this is the currently visible frame.
        parent: Optional Qt parent.
    """

    clicked = pyqtSignal(int)
    double_clicked = pyqtSignal(int)

    _CELL_W = 48
    _CELL_H = 40

    def __init__(
        self,
        frame_index: int,
        duration_ms: int,
        active: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.frame_index = frame_index
        self.duration_ms = duration_ms
        self.active = active
        self.setFixedSize(self._CELL_W, self._CELL_H)
        self.setToolTip(f"Frame {frame_index + 1}  ({duration_ms} ms)")

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        bg = QColor(80, 130, 200) if self.active else QColor(60, 60, 60)
        painter.fillRect(self.rect(), bg)
        # Border
        border_color = QColor(200, 200, 200) if self.active else QColor(40, 40, 40)
        painter.setPen(border_color)
        painter.drawRect(0, 0, self._CELL_W - 1, self._CELL_H - 1)
        # Frame number
        painter.setPen(QColor(240, 240, 240))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            str(self.frame_index + 1),
        )
        # Duration (ms)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            f"{self.duration_ms}ms",
        )
        painter.end()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.clicked.emit(self.frame_index)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        self.double_clicked.emit(self.frame_index)


# ---------------------------------------------------------------------------
# TimelinePanel
# ---------------------------------------------------------------------------


class TimelinePanel(QWidget):
    """Horizontal frame strip for navigation and frame management.

    Args:
        sprite: The sprite document whose frames are shown.
        stack: The undo/redo command stack used for add/delete/duplicate.
        parent: Optional Qt parent.
    """

    #: Emitted when the user selects a frame by clicking.
    frame_selected = pyqtSignal(int)
    #: Emitted when the user changes a frame's duration.
    frame_duration_changed = pyqtSignal(int, int)

    def __init__(
        self,
        sprite: Sprite,
        stack: CommandStack,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._sprite = sprite
        self._stack = stack
        self._active_frame: int = 0
        self._cells: List[_FrameCell] = []

        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active_frame(self) -> int:
        """Index of the currently highlighted frame."""
        return self._active_frame

    def set_active_frame(self, index: int) -> None:
        """Highlight *index* as the active frame and refresh the strip.

        Args:
            index: Frame index to activate.
        """
        if index != self._active_frame:
            self._active_frame = index
            self._update_active_cell()

    def refresh(self) -> None:
        """Rebuild the cell strip to match the current sprite frame list."""
        # Remove old cells.
        for cell in self._cells:
            self._strip_layout.removeWidget(cell)
            cell.deleteLater()
        self._cells.clear()

        for fi, frame in enumerate(self._sprite.frames):
            cell = _FrameCell(fi, frame.duration_ms, active=(fi == self._active_frame))
            cell.clicked.connect(self._on_cell_clicked)
            cell.double_clicked.connect(self._on_cell_double_clicked)
            self._strip_layout.addWidget(cell)
            self._cells.append(cell)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)

        # Button bar.
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(4)
        for label, slot in (
            ("+", self._add_frame),
            ("×", self._remove_frame),
            ("⧉", self._duplicate_frame),
        ):
            btn = QPushButton(label)
            btn.setFixedSize(28, 22)
            btn.clicked.connect(slot)
            btn_bar.addWidget(btn)
        btn_bar.addWidget(QLabel("Frames"))
        btn_bar.addStretch()
        root.addLayout(btn_bar)

        # Scrollable cell strip.
        self._strip_widget = QWidget()
        self._strip_layout = QHBoxLayout(self._strip_widget)
        self._strip_layout.setContentsMargins(4, 4, 4, 4)
        self._strip_layout.setSpacing(2)
        self._strip_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._strip_widget)
        scroll.setFixedHeight(_FrameCell._CELL_H + 20)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(scroll)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_active_cell(self) -> None:
        for cell in self._cells:
            cell.active = cell.frame_index == self._active_frame
            cell.update()

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_cell_clicked(self, frame_index: int) -> None:
        self._active_frame = frame_index
        self._update_active_cell()
        self.frame_selected.emit(frame_index)

    def _on_cell_double_clicked(self, frame_index: int) -> None:
        frame = self._sprite.frames[frame_index]
        ms, ok = QInputDialog.getInt(
            self,
            "Set Duration",
            f"Duration for frame {frame_index + 1} (ms):",
            frame.duration_ms,
            1,
            100_000,
        )
        if ok and ms != frame.duration_ms:
            frame.duration_ms = ms
            self.refresh()
            self.frame_duration_changed.emit(frame_index, ms)

    # ------------------------------------------------------------------
    # Frame management buttons
    # ------------------------------------------------------------------

    def _add_frame(self) -> None:
        insert_at = self._active_frame + 1
        cmd = AddFrameCommand(
            self._sprite,
            self._sprite.frames[self._active_frame].duration_ms,
            index=insert_at,
        )
        self._stack.push(cmd)
        self._active_frame = insert_at
        self.refresh()
        self.frame_selected.emit(self._active_frame)

    def _remove_frame(self) -> None:
        if self._sprite.frame_count <= 1:
            QMessageBox.warning(self, "Spriter", "Cannot delete the last frame.")
            return
        cmd = RemoveFrameCommand(self._sprite, self._active_frame)
        self._stack.push(cmd)
        self._active_frame = min(self._active_frame, self._sprite.frame_count - 1)
        self.refresh()
        self.frame_selected.emit(self._active_frame)

    def _duplicate_frame(self) -> None:
        cmd = DuplicateFrameCommand(self._sprite, self._active_frame)
        self._stack.push(cmd)
        self._active_frame = self._active_frame + 1
        self.refresh()
        self.frame_selected.emit(self._active_frame)
