# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Diffusion / AI frame-prediction slots for :class:`MainWindow`.

These optional features (restyle, next-frame prediction, video animation,
prompt-to-sheet) live in :class:`DiffusionMixin` so the core window stays
focused.  All heavy ML imports happen lazily inside the slots, and each slot
runs on a :class:`_DiffusionWorker` behind an indeterminate progress dialog.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
)

from ..commands.ai_ops import (
    GenerateAnimationCommand,
    GenerateFrameCommand,
    GenerateFramesCommand,
)
from .main_window_base import _MainWindowProtocol


class _DiffusionWorker(QThread):
    """Runs a blocking callable off the UI thread.

    Emits :attr:`finished_ok` with the callable's result on success or
    :attr:`failed` with the exception message on error.
    """

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # surface any backend/model error to the UI
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(result)


class DiffusionMixin(_MainWindowProtocol):
    """Optional AI frame-prediction menu slots."""

    # ------------------------------------------------------------------
    # Diffusion (optional AI frame prediction)
    # ------------------------------------------------------------------

    def _diffusion_models_dir(self) -> Path:
        """Default directory where downloaded models are stored."""
        return Path.home() / ".config" / "spriter" / "models"

    def _diffusion_select_model(self) -> None:
        start = self._settings.diffusion_model_path or str(self._diffusion_models_dir())
        choice, ok = QInputDialog.getItem(
            self,
            "Select Model",
            "Model type:",
            ["Model folder", "Single .safetensors file"],
            0,
            False,
        )
        if not ok:
            return
        if choice == "Single .safetensors file":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Diffusion Model File",
                start,
                "Safetensors Model (*.safetensors)",
            )
        else:
            path = QFileDialog.getExistingDirectory(
                self, "Select Diffusion Model Folder", start
            )
        if not path:
            return
        self._settings.diffusion_model_path = path
        self._settings.save()
        QMessageBox.information(self, "Diffusion", f"Model set to:\n{path}")

    def _diffusion_select_controlnet(self) -> None:
        start = self._settings.diffusion_controlnet_path or str(
            self._diffusion_models_dir()
        )
        path = QFileDialog.getExistingDirectory(
            self, "Select ControlNet Model Folder", start
        )
        if not path:
            return
        self._settings.diffusion_controlnet_path = path
        self._settings.save()
        QMessageBox.information(self, "Diffusion", f"ControlNet set to:\n{path}")

    def _diffusion_select_ip_adapter(self) -> None:
        start = self._settings.diffusion_ip_adapter_path or str(
            self._diffusion_models_dir()
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select IP-Adapter Weight File",
            start,
            "IP-Adapter Weights (*.safetensors *.bin)",
        )
        if not path:
            return
        self._settings.diffusion_ip_adapter_path = path
        self._settings.save()
        QMessageBox.information(self, "Diffusion", f"IP-Adapter set to:\n{path}")

    def _diffusion_select_video_model(self) -> None:
        start = self._settings.diffusion_video_model_path or str(
            self._diffusion_models_dir()
        )
        path = QFileDialog.getExistingDirectory(
            self, "Select Wan Video Model Folder", start
        )
        if not path:
            return
        self._settings.diffusion_video_model_path = path
        self._settings.save()
        QMessageBox.information(self, "Diffusion", f"Video model set to:\n{path}")

    def _diffusion_select_video_lora(self) -> None:
        start = self._settings.diffusion_video_lora_path or str(
            self._diffusion_models_dir()
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video LoRA Weight File",
            start,
            "LoRA Weights (*.safetensors *.bin *.pt);;All Files (*)",
        )
        if not path:
            return
        self._settings.diffusion_video_lora_path = path
        self._settings.save()
        QMessageBox.information(self, "Diffusion", f"Video LoRA set to:\n{path}")

    def _diffusion_model_info(self) -> None:
        from ..ai import diffusion

        model_path = self._settings.diffusion_model_path
        if not model_path:
            QMessageBox.information(
                self,
                "Model Info",
                "No diffusion model selected. Use Diffusion \u2192 Select Model "
                "or Download Model first.",
            )
            return
        info = diffusion.model_info(model_path)
        available = (
            "Yes" if diffusion.is_available() else "No (install spriter[diffusion])"
        )
        device_label = {"cuda": "GPU (CUDA)", "mps": "GPU (Apple MPS)", "cpu": "CPU"}
        device = diffusion.active_device()
        lines = [
            f"Path: {info['path']}",
            f"Type: {info['type']}",
            f"Exists: {'Yes' if info['exists'] else 'No'}",
            f"Size: {self._format_size(info['size_bytes'])}",
            f"Dependencies installed: {available}",
            f"Compute device: {device_label.get(device, device)}",
        ]
        QMessageBox.information(self, "Model Info", "\n".join(lines))

    @staticmethod
    def _format_size(num_bytes: int) -> str:
        size = float(num_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
            size /= 1024.0
        return f"{size:.1f} TB"

    def _diffusion_download_model(self) -> None:
        from ..ai import diffusion

        if not diffusion.is_available():
            self._diffusion_unavailable_message()
            return
        repo_id, ok = QInputDialog.getText(
            self,
            "Download Model",
            "Hugging Face repo ID (e.g. runwayml/stable-diffusion-v1-5):",
        )
        if not ok or not repo_id.strip():
            return
        repo_id = repo_id.strip()
        dest = self._diffusion_models_dir() / repo_id.replace("/", "__")

        def task() -> str:
            path = diffusion.download_model(repo_id, dest)
            return str(path)

        def on_done(result) -> None:
            self._settings.diffusion_model_path = str(result)
            self._settings.save()
            QMessageBox.information(
                self, "Download Model", f"Model downloaded to:\n{result}"
            )

        self._run_diffusion_task(task, on_done, f"Downloading {repo_id}\u2026")

    def _diffusion_generate_frame(self) -> None:
        """Backwards-compatible alias for :meth:`_diffusion_restyle_frame`."""
        self._diffusion_restyle_frame()

    def _diffusion_restyle_frame(self) -> None:
        if self._sprite is None:
            return
        from ..ai import diffusion
        from ..ai.postprocess import prepare_generated_frame

        if not diffusion.is_available():
            self._diffusion_unavailable_message()
            return
        model_path = self._settings.diffusion_model_path
        if not model_path or not Path(model_path).exists():
            QMessageBox.information(
                self,
                "Restyle Frame",
                "No diffusion model selected. Use Diffusion \u2192 Select Model "
                "or Download Model first.",
            )
            return

        prompt, ok = QInputDialog.getText(
            self,
            "Restyle Frame",
            "Optional prompt to guide the restyle:",
        )
        if not ok:
            return

        from ..core.compositor import composite_frame

        li, fi = self._active_layer_frame()
        init_rgba = composite_frame(self._sprite, fi)
        canvas_w, canvas_h = self._sprite.width, self._sprite.height

        def task():
            pipeline = diffusion.load_pipeline(model_path)
            generated = diffusion.generate(pipeline, init_rgba, prompt.strip())
            return prepare_generated_frame(generated, canvas_w, canvas_h)

        self._run_diffusion_task(
            task, self._insert_generated_frame(fi, li), "Restyling frame\u2026"
        )

    def _diffusion_predict_next_frame(self) -> None:
        if self._sprite is None:
            return
        from ..ai import diffusion
        from ..ai.postprocess import prepare_generated_frame

        if not diffusion.is_available():
            self._diffusion_unavailable_message()
            return
        model_path = self._settings.diffusion_model_path
        if not model_path or not Path(model_path).exists():
            QMessageBox.information(
                self,
                "Predict Next Frame",
                "No diffusion model selected. Use Diffusion \u2192 Select Model "
                "or Download Model first.",
            )
            return
        cn_path = self._settings.diffusion_controlnet_path
        if not cn_path or not Path(cn_path).exists():
            QMessageBox.information(
                self,
                "Predict Next Frame",
                "Next-frame prediction needs a ControlNet model. Use Diffusion "
                "\u2192 Select ControlNet first (or use Restyle Frame instead).",
            )
            return

        from ..core.compositor import composite_frame

        li, fi = self._active_layer_frame()
        if fi < 1:
            QMessageBox.information(
                self,
                "Predict Next Frame",
                "Next-frame prediction needs at least two preceding frames. "
                "Draw an earlier frame first, or use Restyle Frame.",
            )
            return

        prompt, ok = QInputDialog.getText(
            self,
            "Predict Next Frame",
            "Optional prompt to guide generation:",
        )
        if not ok:
            return

        context = [composite_frame(self._sprite, i) for i in (fi - 1, fi)]
        canvas_w, canvas_h = self._sprite.width, self._sprite.height

        # Automatic, per-project fine-tuning: derive the adapter location, train
        # once on demand, and thereafter apply it without any user management.
        adapter_dir, train_frames = self._diffusion_prepare_adapter()
        label = (
            "Learning animation & predicting\u2026"
            if train_frames is not None
            else "Predicting next frame\u2026"
        )

        # Optional IP-Adapter identity lock (holds the sprite's look).
        ip_path = self._settings.diffusion_ip_adapter_path
        ip_adapter = ip_path if ip_path and Path(ip_path).exists() else None

        def task():
            if train_frames is not None:
                from ..ai import train

                train.train_next_frame_predictor(model_path, train_frames, adapter_dir)
            use_adapter = self._diffusion_ready_adapter(adapter_dir)
            pipeline = diffusion.load_controlnet_pipeline(
                model_path,
                cn_path,
                adapter_path=use_adapter,
                ip_adapter_path=ip_adapter,
            )
            generated = diffusion.generate_next_frame(
                pipeline,
                context,
                prompt.strip(),
                use_ip_adapter=ip_adapter is not None,
            )
            return prepare_generated_frame(generated, canvas_w, canvas_h)

        self._run_diffusion_task(task, self._insert_generated_frame(fi, li), label)

    def _diffusion_prepare_adapter(self):
        """Return ``(adapter_dir, frames_to_train)`` for automatic fine-tuning.

        *adapter_dir* is the per-project adapter location (or ``None`` when the
        training dependencies are unavailable).  *frames_to_train* is a harvested
        frame list when a one-time training run should happen first, else
        ``None`` (an adapter already exists, there is too little data, or the
        user declined).
        """
        try:
            from ..ai import train
            from ..ai.dataset import harvest_frames
        except ImportError:
            return None, None
        if not train.is_train_available():
            return None, None
        assert self._sprite is not None
        key = str(self._current_path) if self._current_path else "untitled"
        adapter_dir = train.adapter_dir_for(key)
        if train.has_trained_adapter(adapter_dir):
            return adapter_dir, None
        frames = harvest_frames(self._sprite)
        if len(frames) < train.MIN_TRAIN_FRAMES:
            return adapter_dir, None
        answer = QMessageBox.question(
            self,
            "Predict Next Frame",
            "Spriter can learn this animation once to make predictions match "
            "your sprite. This runs in the background now; later predictions "
            "reuse it automatically.\n\nLearn from this animation?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            return adapter_dir, frames
        return adapter_dir, None

    @staticmethod
    def _diffusion_ready_adapter(adapter_dir):
        """Return *adapter_dir* only when it holds trained weights, else ``None``."""
        if adapter_dir is None:
            return None
        from ..ai import train

        return adapter_dir if train.has_trained_adapter(adapter_dir) else None

    def _insert_generated_frame(self, fi: int, li: int):
        """Return an on-done callback that inserts generated *pixels* after *fi*."""

        def on_done(pixels) -> None:
            assert self._sprite is not None
            cmd = GenerateFrameCommand(self._sprite, fi, li, pixels)
            self._stack.push(cmd)
            new_fi = fi + 1
            if self._canvas:
                self._canvas.active_frame = new_fi
                self._canvas.invalidate_cache()
            if self._timeline:
                self._timeline.set_active_frame(new_fi)
                self._timeline.refresh()
            self._unsaved = True
            self._refresh_undo_redo_labels()

        return on_done

    def _diffusion_animate_from_frame(self) -> None:
        if self._sprite is None:
            return
        from ..ai import diffusion
        from ..ai.postprocess import prepare_video_frames

        if not diffusion.video_is_available():
            self._diffusion_unavailable_message()
            return
        model_path = self._settings.diffusion_video_model_path
        if not model_path or not Path(model_path).exists():
            QMessageBox.information(
                self,
                "Animate From Frame",
                "No video model selected. Use Diffusion \u2192 Select Video Model "
                "first.",
            )
            return

        prompt, ok = QInputDialog.getText(
            self,
            "Animate From Frame",
            "Prompt describing the motion:",
        )
        if not ok:
            return
        count, ok = QInputDialog.getInt(
            self,
            "Animate From Frame",
            "Number of frames to generate:",
            diffusion.VIDEO_DEFAULT_NUM_FRAMES,
            2,
            257,
        )
        if not ok:
            return

        from ..core.compositor import composite_frame

        li, fi = self._active_layer_frame()
        init_rgba = composite_frame(self._sprite, fi)
        canvas_w, canvas_h = self._sprite.width, self._sprite.height
        lora = self._settings.diffusion_video_lora_path or None

        def task():
            pipeline = diffusion.load_video_pipeline(model_path, lora_path=lora)
            frames = diffusion.generate_video_frames(
                pipeline, init_rgba, prompt.strip(), num_frames=count
            )
            prepared = prepare_video_frames(frames, canvas_w, canvas_h)
            # Drop the first frame: it mirrors the current one.
            return prepared[1:] if len(prepared) > 1 else prepared

        def on_done(frames) -> None:
            assert self._sprite is not None
            if not frames:
                QMessageBox.information(
                    self, "Animate From Frame", "No frames were generated."
                )
                return
            cmd = GenerateFramesCommand(self._sprite, fi, li, frames)
            self._stack.push(cmd)
            new_fi = fi + 1
            if self._canvas:
                self._canvas.active_frame = new_fi
                self._canvas.invalidate_cache()
            if self._timeline:
                self._timeline.set_active_frame(new_fi)
                self._timeline.refresh()
            self._unsaved = True
            self._refresh_undo_redo_labels()

        self._run_diffusion_task(task, on_done, "Generating animation\u2026")

    def _diffusion_generate_sheet(self) -> None:
        if self._sprite is None:
            return
        from ..ai import diffusion
        from ..ai.postprocess import prepare_video_frames

        if not diffusion.text_video_is_available():
            self._diffusion_unavailable_message()
            return
        model_path = self._settings.diffusion_video_model_path
        if not model_path or not Path(model_path).exists():
            QMessageBox.information(
                self,
                "Generate Sprite Sheet",
                "No video model selected. Use Diffusion \u2192 Select Video Model "
                "first (a Wan text-to-video model).",
            )
            return

        prompt, ok = QInputDialog.getText(
            self, "Generate Sprite Sheet", "Describe the animation:"
        )
        if not ok or not prompt.strip():
            return
        width, ok = QInputDialog.getInt(
            self, "Generate Sprite Sheet", "Frame width:", self._sprite.width, 1, 1024
        )
        if not ok:
            return
        height, ok = QInputDialog.getInt(
            self,
            "Generate Sprite Sheet",
            "Frame height:",
            self._sprite.height,
            1,
            1024,
        )
        if not ok:
            return
        count, ok = QInputDialog.getInt(
            self,
            "Generate Sprite Sheet",
            "Number of frames:",
            diffusion.VIDEO_DEFAULT_NUM_FRAMES,
            2,
            257,
        )
        if not ok:
            return

        lora = self._settings.diffusion_video_lora_path or None
        name = prompt.strip()[:40] or "Generated"
        li = self._layers_panel.active_layer if self._layers_panel else 0

        def task():
            pipeline = diffusion.load_text_video_pipeline(model_path, lora_path=lora)
            frames = diffusion.generate_text_video_frames(
                pipeline, prompt.strip(), width=width, height=height, num_frames=count
            )
            return prepare_video_frames(frames, width, height)

        def on_done(frames) -> None:
            assert self._sprite is not None
            if not frames:
                QMessageBox.information(
                    self, "Generate Sprite Sheet", "No frames were generated."
                )
                return
            cmd = GenerateAnimationCommand(
                self._sprite, name, frames, width, height, li
            )
            self._stack.push(cmd)
            self._rebuild_ui()
            self._unsaved = True
            self._refresh_undo_redo_labels()

        self._run_diffusion_task(task, on_done, "Generating sprite sheet\u2026")

    def _diffusion_unavailable_message(self) -> None:
        QMessageBox.information(
            self,
            "Diffusion",
            "Diffusion features require optional dependencies.\n\n"
            "Install them with:\n    pip install spriter[diffusion]",
        )

    def _run_diffusion_task(self, task, on_done, label: str) -> None:
        """Run *task* in a worker thread behind an indeterminate progress dialog."""
        progress = QProgressDialog(label, "", 0, 0, self)
        progress.setWindowTitle("Diffusion")
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        worker = _DiffusionWorker(task, self)
        # Keep a reference so the thread is not garbage-collected mid-run.
        self._diffusion_worker = worker

        def cleanup() -> None:
            progress.close()
            self._diffusion_worker = None

        def handle_ok(result) -> None:
            cleanup()
            on_done(result)

        def handle_err(message: str) -> None:
            cleanup()
            QMessageBox.critical(self, "Diffusion Error", message)

        worker.finished_ok.connect(handle_ok)
        worker.failed.connect(handle_err)
        worker.start()
        progress.show()
