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

import numpy as np
from typing import List, Optional

from PyQt6.QtCore import Qt, QPoint, QEvent, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
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
    MoveFrameCommand,
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
        thumbnail: Optional pre-rendered QPixmap of the frame composite.
        parent: Optional Qt parent.
    """

    clicked = pyqtSignal(int)
    double_clicked = pyqtSignal(int)
    right_clicked = pyqtSignal(int, object)  # (frame_index, QPoint global pos)

    _CELL_W = 56
    _CELL_H = 60
    _THUMB_SIZE = 40  # thumbnail display size in pixels

    def __init__(
        self,
        frame_index: int,
        duration_ms: int,
        active: bool = False,
        thumbnail: Optional[QPixmap] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.frame_index = frame_index
        self.duration_ms = duration_ms
        self.active = active
        self.thumbnail = thumbnail
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

        # Frame thumbnail (centred in the upper portion)
        thumb_y = 2
        if self.thumbnail is not None:
            scaled = self.thumbnail.scaled(
                self._THUMB_SIZE,
                self._THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            tx = (self._CELL_W - scaled.width()) // 2
            painter.drawPixmap(tx, thumb_y, scaled)
        else:
            # Placeholder grey square
            painter.fillRect(
                (self._CELL_W - self._THUMB_SIZE) // 2,
                thumb_y,
                self._THUMB_SIZE,
                self._THUMB_SIZE,
                QColor(90, 90, 90),
            )

        # Frame number (top-left, small)
        painter.setPen(QColor(240, 240, 240))
        from PyQt6.QtCore import QRect, QFont

        small_font = QFont()
        small_font.setPointSize(7)
        painter.setFont(small_font)
        painter.drawText(2, 2, self._CELL_W - 2, 12, 0, str(self.frame_index + 1))

        # Duration (ms) — bottom strip
        painter.setPen(QColor(180, 180, 180))
        from PyQt6.QtCore import QRect

        bot_rect = self.rect().adjusted(0, self._CELL_H - 14, 0, 0)
        painter.drawText(
            bot_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            f"{self.duration_ms}ms",
        )
        painter.end()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(self.frame_index, event.globalPosition().toPoint())
        else:
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

        # Drag-to-reorder state
        self._drag_source: Optional[int] = None  # frame index being dragged
        self._drag_start_pos: Optional[QPoint] = None
        self._dragging: bool = False
        self._drag_indicator: Optional[int] = None  # insert-before index

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
            thumbnail = self._make_thumbnail(fi)
            cell = _FrameCell(
                fi,
                frame.duration_ms,
                active=(fi == self._active_frame),
                thumbnail=thumbnail,
            )
            cell.clicked.connect(self._on_cell_clicked)
            cell.double_clicked.connect(self._on_cell_double_clicked)
            cell.right_clicked.connect(self._on_cell_context_menu)
            cell.installEventFilter(self)
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
            ("\u25c0", self._move_frame_left),
            ("\u25b6", self._move_frame_right),
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

    def _move_frame_left(self) -> None:
        if self._active_frame <= 0:
            return
        to = self._active_frame - 1
        cmd = MoveFrameCommand(self._sprite, self._active_frame, to)
        self._stack.push(cmd)
        self._active_frame = to
        self.refresh()
        self.frame_selected.emit(self._active_frame)

    def _move_frame_right(self) -> None:
        if self._active_frame >= self._sprite.frame_count - 1:
            return
        to = self._active_frame + 1
        cmd = MoveFrameCommand(self._sprite, self._active_frame, to)
        self._stack.push(cmd)
        self._active_frame = to
        self.refresh()
        self.frame_selected.emit(self._active_frame)

    def _on_cell_context_menu(self, frame_index: int, pos: object) -> None:
        """Show a right-click context menu for the given frame cell."""
        self._active_frame = frame_index
        self._update_active_cell()
        menu = QMenu(self)
        menu.addAction("Duplicate Frame", self._duplicate_frame)
        menu.addAction("Delete Frame", self._remove_frame)
        menu.addSeparator()
        menu.addAction(
            "Set Duration\u2026",
            lambda: self._on_cell_double_clicked(frame_index),
        )
        menu.addSeparator()
        menu.addAction("Move Left", self._move_frame_left)
        menu.addAction("Move Right", self._move_frame_right)
        menu.exec(pos if isinstance(pos, QPoint) else QPoint())

    # ------------------------------------------------------------------
    # Thumbnail helper
    # ------------------------------------------------------------------

    def _make_thumbnail(self, frame_index: int) -> Optional[QPixmap]:
        """Composite *frame_index* and return a small QPixmap thumbnail."""
        if (
            self._sprite.frame_count == 0
            or self._sprite.layer_count == 0
            or self._sprite.width == 0
            or self._sprite.height == 0
        ):
            return None
        try:
            from ..core.compositor import composite_frame

            fi = min(frame_index, self._sprite.frame_count - 1)
            arr = composite_frame(self._sprite, fi)
            arr = np.ascontiguousarray(arr)
            h, w = arr.shape[:2]
            img = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
            sz = _FrameCell._THUMB_SIZE
            scaled = img.scaled(
                sz,
                sz,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
            return QPixmap.fromImage(scaled)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Drag-to-reorder event filter
    # ------------------------------------------------------------------

    _DRAG_THRESHOLD = 8  # Manhattan distance before drag activates

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        """Intercept mouse events on _FrameCell widgets for drag reorder."""
        if not isinstance(obj, _FrameCell):
            return False

        et = event.type()

        if et == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self._drag_source = obj.frame_index
                self._drag_start_pos = event.globalPosition().toPoint()
                self._dragging = False
            return False  # let normal click/press propagate

        if et == QEvent.Type.MouseMove:
            if (
                self._drag_source is not None
                and self._drag_start_pos is not None
                and (event.buttons() & Qt.MouseButton.LeftButton)
            ):
                delta = event.globalPosition().toPoint() - self._drag_start_pos
                if (
                    not self._dragging
                    and delta.manhattanLength() >= self._DRAG_THRESHOLD
                ):
                    self._dragging = True
                if self._dragging:
                    # Find the cell under the current global position.
                    local = self._strip_widget.mapFromGlobal(
                        event.globalPosition().toPoint()
                    )
                    target = self._frame_index_at(local)
                    if target != self._drag_indicator:
                        self._drag_indicator = target
                        self._update_drag_highlights()
                    return True  # consume while dragging
            return False

        if et == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self._dragging:
                src = self._drag_source
                local = self._strip_widget.mapFromGlobal(
                    event.globalPosition().toPoint()
                )
                dst = self._frame_index_at(local)
                self._drag_source = None
                self._drag_start_pos = None
                self._dragging = False
                self._drag_indicator = None
                self._clear_drag_highlights()
                if dst is not None and src is not None and dst != src:
                    cmd = MoveFrameCommand(self._sprite, src, dst)
                    self._stack.push(cmd)
                    self._active_frame = dst
                    self.refresh()
                    self.frame_selected.emit(self._active_frame)
                return True  # consume the release that ended the drag
            # Clean up state on any release
            self._drag_source = None
            self._drag_start_pos = None
            self._dragging = False
            self._drag_indicator = None
            return False

        return False

    def _frame_index_at(self, strip_local: QPoint) -> Optional[int]:
        """Return the frame index of the cell under *strip_local* (strip widget coords)."""
        child = self._strip_widget.childAt(strip_local)
        if isinstance(child, _FrameCell):
            return child.frame_index
        return None

    def _update_drag_highlights(self) -> None:
        """Visually highlight the drag target cell."""
        for cell in self._cells:
            target = self._drag_indicator
            cell.setStyleSheet(
                "background-color: rgb(180, 100, 30);"
                if cell.frame_index == target and target is not None
                else ""
            )

    def _clear_drag_highlights(self) -> None:
        """Remove drag highlighting from all cells."""
        for cell in self._cells:
            cell.setStyleSheet("")
