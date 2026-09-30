# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Animation timeline panel widget.

:class:`TimelinePanel` displays all frames of the active animation timeline
as a horizontal strip of clickable cells, plus an animation navigation bar
(up/down + add/remove/rename) for switching between a sprite's timelines.
It sits in a dock at the bottom of the main window and coordinates frame
and animation navigation with the canvas and preview widgets.

Signals
-------
frame_selected(int)
    Emitted when the user clicks a frame cell; carries the frame index.
frame_duration_changed(int, int)
    Emitted after the user edits a frame's duration; carries
    ``(frame_index, new_duration_ms)``.
animation_changed(int)
    Emitted after the active animation timeline changes; carries the new
    timeline index.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPropertyAnimation,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..commands.animation_ops import (
    AddTimelineCommand,
    RemoveTimelineCommand,
    RenameTimelineCommand,
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
        thumbnail: QPixmap | None = None,
        parent: QWidget | None = None,
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
        from PyQt6.QtGui import QFont

        small_font = QFont()
        small_font.setPointSize(7)
        painter.setFont(small_font)
        painter.drawText(2, 2, self._CELL_W - 2, 12, 0, str(self.frame_index + 1))

        # Duration (ms) — bottom strip
        painter.setPen(QColor(180, 180, 180))

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
    #: Emitted when the active animation timeline changes; carries its index.
    animation_changed = pyqtSignal(int)

    def __init__(
        self,
        sprite: Sprite,
        stack: CommandStack,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sprite = sprite
        self._stack = stack
        self._active_frame: int = 0
        self._cells: list[_FrameCell] = []

        # Drag-to-reorder state
        self._drag_source: int | None = None  # frame index being dragged
        self._drag_start_pos: QPoint | None = None
        self._dragging: bool = False
        self._drag_indicator: int | None = None  # insert-before index
        self._home_positions: list[QPoint] = []  # cell resting positions
        self._grab_offset_x: int = 0  # cursor-to-cell-left offset at grab
        self._cell_anims: dict[int, QPropertyAnimation] = {}

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
        self._update_animation_label()

    # ------------------------------------------------------------------
    # Animation (timeline) navigation
    # ------------------------------------------------------------------

    def _update_animation_label(self) -> None:
        """Refresh the ``Anim i/N: name`` label from the sprite state."""
        idx = self._sprite.active_timeline_index
        total = self._sprite.timeline_count
        name = self._sprite.timelines[idx].name
        self._anim_label.setText(f"Anim {idx + 1}/{total}: {name}")

    def _go_to_animation(self, index: int) -> None:
        index = max(0, min(self._sprite.timeline_count - 1, index))
        if index == self._sprite.active_timeline_index:
            return
        self._sprite.set_active_timeline(index)
        self._active_frame = 0
        self.refresh()
        self.animation_changed.emit(index)

    def _prev_animation(self) -> None:
        self._go_to_animation(self._sprite.active_timeline_index - 1)

    def _next_animation(self) -> None:
        self._go_to_animation(self._sprite.active_timeline_index + 1)

    def _add_animation(self) -> None:
        self._stack.push(AddTimelineCommand(self._sprite))
        self._active_frame = 0
        self.refresh()
        self.animation_changed.emit(self._sprite.active_timeline_index)

    def _remove_animation(self) -> None:
        if self._sprite.timeline_count <= 1:
            QMessageBox.warning(self, "Spriter", "Cannot delete the last animation.")
            return
        self._stack.push(
            RemoveTimelineCommand(self._sprite, self._sprite.active_timeline_index)
        )
        self._active_frame = 0
        self.refresh()
        self.animation_changed.emit(self._sprite.active_timeline_index)

    def _rename_animation(self) -> None:
        idx = self._sprite.active_timeline_index
        current = self._sprite.timelines[idx].name
        name, ok = QInputDialog.getText(
            self, "Rename Animation", "Animation name:", text=current
        )
        if ok and name and name != current:
            self._stack.push(RenameTimelineCommand(self._sprite, idx, name))
            self._update_animation_label()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)

        # Animation (timeline) navigation bar.
        anim_bar = QHBoxLayout()
        anim_bar.setSpacing(4)
        up_btn = QPushButton("\u25b2")
        up_btn.setFixedSize(28, 22)
        up_btn.setToolTip("Previous animation")
        up_btn.clicked.connect(self._prev_animation)
        down_btn = QPushButton("\u25bc")
        down_btn.setFixedSize(28, 22)
        down_btn.setToolTip("Next animation")
        down_btn.clicked.connect(self._next_animation)
        anim_bar.addWidget(up_btn)
        anim_bar.addWidget(down_btn)
        self._anim_label = QLabel()
        anim_bar.addWidget(self._anim_label)
        anim_bar.addStretch()
        for label, slot, tip in (
            ("+", self._add_animation, "Add animation"),
            ("\u00d7", self._remove_animation, "Delete animation"),
            ("\u270e", self._rename_animation, "Rename animation"),
        ):
            btn = QPushButton(label)
            btn.setFixedSize(28, 22)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            anim_bar.addWidget(btn)
        root.addLayout(anim_bar)

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

    def _make_thumbnail(self, frame_index: int) -> QPixmap | None:
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
    _ANIM_DURATION_MS = 120  # duration of the displaced-cell slide

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
                    self._begin_drag(event)
                if self._dragging:
                    local = self._strip_widget.mapFromGlobal(
                        event.globalPosition().toPoint()
                    )
                    self._move_dragged_cell(local.x())
                    hover = self._insertion_index_for_x(local.x())
                    if hover is not None and hover != self._drag_indicator:
                        self._drag_indicator = hover
                        self._animate_displaced(hover)
                    return True  # consume while dragging
            return False

        if et == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self._dragging:
                src = self._drag_source
                dst = self._drag_indicator
                if dst is None:
                    local = self._strip_widget.mapFromGlobal(
                        event.globalPosition().toPoint()
                    )
                    dst = self._frame_index_at(local)
                self._reset_drag_state()
                if dst is not None and src is not None and dst != src:
                    cmd = MoveFrameCommand(self._sprite, src, dst)
                    self._stack.push(cmd)
                    self._active_frame = dst
                    self.refresh()
                    self.frame_selected.emit(self._active_frame)
                else:
                    # No reorder: snap the lifted cell back into place.
                    self.refresh()
                return True  # consume the release that ended the drag
            self._reset_drag_state()
            return False

        return False

    # ------------------------------------------------------------------
    # Drag geometry / animation helpers
    # ------------------------------------------------------------------

    def _begin_drag(self, event) -> None:  # type: ignore[no-untyped-def]
        """Activate dragging: snapshot resting positions and lift the cell."""
        self._dragging = True
        self._home_positions = [cell.pos() for cell in self._cells]
        cell = self._cell_for_index(self._drag_source)
        if cell is not None:
            local = self._strip_widget.mapFromGlobal(event.globalPosition().toPoint())
            self._grab_offset_x = local.x() - cell.x()
            cell.raise_()
        self._drag_indicator = self._drag_source

    def _move_dragged_cell(self, cursor_x: int) -> None:
        """Move the lifted cell so it follows the cursor horizontally."""
        cell = self._cell_for_index(self._drag_source)
        if cell is None or not self._home_positions:
            return
        y = self._home_positions[0].y()
        span = self._cell_span()
        base_x = self._home_positions[0].x()
        max_x = base_x + (len(self._cells) - 1) * span
        new_x = max(base_x, min(max_x, cursor_x - self._grab_offset_x))
        cell.move(new_x, y)

    def _animate_displaced(self, hover: int) -> None:
        """Slide every non-dragged cell to the slot layout for *hover*."""
        if not self._home_positions or self._drag_source is None:
            return
        y = self._home_positions[0].y()
        targets = self._compute_target_positions(self._drag_source, hover)
        for fi, tx in targets.items():
            cell = self._cell_for_index(fi)
            if cell is None:
                continue
            existing = self._cell_anims.get(fi)
            if existing is not None:
                existing.stop()
            anim = QPropertyAnimation(cell, b"pos", self)
            anim.setDuration(self._ANIM_DURATION_MS)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.setStartValue(cell.pos())
            anim.setEndValue(QPoint(tx, y))
            anim.start()
            self._cell_anims[fi] = anim

    def _compute_target_positions(self, source: int, hover: int) -> dict[int, int]:
        """Return ``frame_index -> target_x`` for the non-dragged cells.

        The dragged frame *source* is reserved slot *hover*; the remaining
        frames fill the other slots in order, opening a gap at the hover slot
        and closing the gap left by the source.
        """
        if not self._home_positions:
            return {}
        base_x = self._home_positions[0].x()
        span = self._cell_span()
        n = len(self._cells)
        hover = max(0, min(n - 1, hover))
        targets: dict[int, int] = {}
        slot = 0
        for fi in range(n):
            if fi == source:
                continue
            if slot == hover:
                slot += 1
            targets[fi] = base_x + slot * span
            slot += 1
        return targets

    def _insertion_index_for_x(self, x: int) -> int | None:
        """Map a cursor x (strip coords) to a target slot ``0..n-1``."""
        if not self._home_positions:
            return None
        span = self._cell_span()
        if span <= 0:
            return None
        base_x = self._home_positions[0].x()
        idx = round((x - base_x) / span)
        return max(0, min(len(self._cells) - 1, int(idx)))

    def _cell_span(self) -> int:
        """Horizontal distance between adjacent cell left edges."""
        return _FrameCell._CELL_W + self._strip_layout.spacing()

    def _cell_for_index(self, frame_index: int | None) -> _FrameCell | None:
        """Return the cell whose ``frame_index`` matches, else ``None``."""
        if frame_index is None:
            return None
        for cell in self._cells:
            if cell.frame_index == frame_index:
                return cell
        return None

    def _reset_drag_state(self) -> None:
        """Stop animations and clear all transient drag state."""
        for anim in self._cell_anims.values():
            anim.stop()
        self._cell_anims.clear()
        self._drag_source = None
        self._drag_start_pos = None
        self._dragging = False
        self._drag_indicator = None
        self._home_positions = []
        self._grab_offset_x = 0

    def _frame_index_at(self, strip_local: QPoint) -> int | None:
        """Return the frame index of the cell under *strip_local* (strip widget coords)."""
        child = self._strip_widget.childAt(strip_local)
        if isinstance(child, _FrameCell):
            return child.frame_index
        return None
