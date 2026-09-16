# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the optional diffusion frame-prediction feature."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _write_safetensors_stub(path, keys):
    """Write a minimal valid ``.safetensors`` file whose header lists *keys*."""
    import json
    import struct

    header = {}
    offset = 0
    for key in keys:
        header[key] = {
            "dtype": "F16",
            "shape": [1],
            "data_offsets": [offset, offset + 2],
        }
        offset += 2
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * offset)


# ---------------------------------------------------------------------------
# Post-processing (pure, no ML dependencies)
# ---------------------------------------------------------------------------


class TestPostprocess:
    def _image_on_white(self, size=64, block=20):
        """White background with a solid red block in the centre."""
        arr = np.zeros((size, size, 4), dtype=np.uint8)
        arr[..., :3] = 255  # white
        arr[..., 3] = 255
        lo = (size - block) // 2
        hi = lo + block
        arr[lo:hi, lo:hi] = (255, 0, 0, 255)  # red block
        return arr, lo, hi

    def test_remove_flat_background_keys_out_border_color(self):
        from spriter.ai.postprocess import remove_flat_background

        arr, lo, hi = self._image_on_white()
        out = remove_flat_background(arr)
        # White border pixels become fully transparent.
        assert out[0, 0, 3] == 0
        # Red block survives.
        assert tuple(out[lo + 1, lo + 1]) == (255, 0, 0, 255)

    def test_remove_flat_background_returns_copy(self):
        from spriter.ai.postprocess import remove_flat_background

        arr, _, _ = self._image_on_white()
        out = remove_flat_background(arr)
        assert out is not arr

    def test_autocrop_reduces_to_content_bbox(self):
        from spriter.ai.postprocess import autocrop_rgba, remove_flat_background

        arr, lo, hi = self._image_on_white(size=64, block=20)
        keyed = remove_flat_background(arr)
        cropped = autocrop_rgba(keyed)
        assert cropped.shape[:2] == (hi - lo, hi - lo)

    def test_autocrop_fully_transparent_is_noop(self):
        from spriter.ai.postprocess import autocrop_rgba

        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        out = autocrop_rgba(arr)
        assert out.shape[:2] == (10, 10)

    def test_fit_into_canvas_downscales_large_image(self):
        from spriter.ai.postprocess import fit_into_canvas

        big = np.zeros((100, 80, 4), dtype=np.uint8)
        big[..., 3] = 255
        out = fit_into_canvas(big, 32, 32)
        assert out.shape == (32, 32, 4)
        # Content is present somewhere.
        assert bool((out[..., 3] > 0).any())

    def test_fit_into_canvas_never_upscales(self):
        from spriter.ai.postprocess import fit_into_canvas

        small = np.zeros((8, 8, 4), dtype=np.uint8)
        small[..., 3] = 255
        out = fit_into_canvas(small, 32, 32)
        assert out.shape == (32, 32, 4)
        # 8x8 opaque block stays 8x8 (64 opaque pixels), not scaled up.
        assert int((out[..., 3] > 0).sum()) == 64

    def test_fit_into_canvas_centers_content(self):
        from spriter.ai.postprocess import fit_into_canvas

        small = np.zeros((4, 4, 4), dtype=np.uint8)
        small[..., 3] = 255
        out = fit_into_canvas(small, 10, 10)
        # Centred: offset (10-4)//2 = 3.
        assert bool((out[3:7, 3:7, 3] > 0).all())
        assert out[0, 0, 3] == 0

    def test_prepare_generated_frame_returns_canvas_sized(self):
        from spriter.ai.postprocess import prepare_generated_frame

        arr, _, _ = self._image_on_white(size=64, block=20)
        out = prepare_generated_frame(arr, 32, 24)
        assert out.shape == (24, 32, 4)

    def test_validate_rejects_non_rgba(self):
        from spriter.ai.postprocess import remove_flat_background

        with pytest.raises(ValueError):
            remove_flat_background(np.zeros((4, 4, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# GenerateFrameCommand
# ---------------------------------------------------------------------------


class TestGenerateFrameCommand:
    def _sprite(self):
        from spriter.core.sprite import Sprite

        sprite = Sprite(16, 16)
        sprite.add_layer("Layer 1")
        sprite.add_frame()
        sprite.add_frame()
        return sprite

    def _pixels(self, sprite, value=(1, 2, 3, 255)):
        arr = np.zeros((sprite.height, sprite.width, 4), dtype=np.uint8)
        arr[:] = value
        return arr

    def test_execute_inserts_frame_after_current(self):
        from spriter.commands.ai_ops import GenerateFrameCommand

        sprite = self._sprite()
        before = sprite.frame_count
        cmd = GenerateFrameCommand(sprite, 0, 0, self._pixels(sprite))
        cmd.execute()
        assert sprite.frame_count == before + 1

    def test_execute_writes_pixels_to_active_layer(self):
        from spriter.commands.ai_ops import GenerateFrameCommand

        sprite = self._sprite()
        pix = self._pixels(sprite, (10, 20, 30, 255))
        cmd = GenerateFrameCommand(sprite, 0, 0, pix)
        cmd.execute()
        cel = sprite.get_cel(0, 1)
        assert tuple(cel.pixels[0, 0]) == (10, 20, 30, 255)

    def test_undo_removes_inserted_frame(self):
        from spriter.commands.ai_ops import GenerateFrameCommand

        sprite = self._sprite()
        before = sprite.frame_count
        cmd = GenerateFrameCommand(sprite, 0, 0, self._pixels(sprite))
        cmd.execute()
        cmd.undo()
        assert sprite.frame_count == before

    def test_redo_reapplies(self):
        from spriter.commands.ai_ops import GenerateFrameCommand

        sprite = self._sprite()
        pix = self._pixels(sprite, (5, 6, 7, 255))
        cmd = GenerateFrameCommand(sprite, 0, 0, pix)
        cmd.execute()
        cmd.undo()
        cmd.execute()
        assert tuple(sprite.get_cel(0, 1).pixels[0, 0]) == (5, 6, 7, 255)

    def test_rejects_mismatched_pixels(self):
        from spriter.commands.ai_ops import GenerateFrameCommand

        sprite = self._sprite()
        bad = np.zeros((8, 8, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            GenerateFrameCommand(sprite, 0, 0, bad)


# ---------------------------------------------------------------------------
# Diffusion backend (dependencies mocked)
# ---------------------------------------------------------------------------


class TestDiffusionBackend:
    def test_is_available_returns_bool(self):
        from spriter.ai import diffusion

        assert isinstance(diffusion.is_available(), bool)

    def test_download_model_requires_dependencies(self):
        from spriter.ai import diffusion

        with patch.object(diffusion, "is_available", return_value=False):
            with pytest.raises(RuntimeError):
                diffusion.download_model("some/repo", "/tmp/x")

    def test_generate_wires_pipeline_call(self):
        # Fake pipeline returns an object with an `images` list of PIL images.
        from PIL import Image

        from spriter.ai import diffusion

        fake_img = Image.new("RGB", (32, 32), (0, 128, 255))
        result = MagicMock()
        result.images = [fake_img]
        pipeline = MagicMock(return_value=result)

        init = np.zeros((16, 16, 4), dtype=np.uint8)
        init[..., 3] = 255
        out = diffusion.generate(pipeline, init, "a knight", strength=0.4)

        pipeline.assert_called_once()
        _, kwargs = pipeline.call_args
        assert kwargs["prompt"] == "a knight"
        assert kwargs["strength"] == 0.4
        assert out.shape == (32, 32, 4)

    def test_generate_upscales_init_image_to_native(self):
        from PIL import Image

        from spriter.ai import diffusion

        fake_img = Image.new("RGB", (32, 32), (0, 128, 255))
        result = MagicMock()
        result.images = [fake_img]
        pipeline = MagicMock(return_value=result)  # class name has no "XL" -> SD

        init = np.zeros((16, 16, 4), dtype=np.uint8)
        init[..., 3] = 255
        diffusion.generate(pipeline, init, "hi")

        _, kwargs = pipeline.call_args
        # 16x16 init upscaled so its longer side is the SD native size (512).
        assert kwargs["image"].size == (512, 512)

    def test_native_size_for_sd_and_sdxl(self):
        from spriter.ai import diffusion

        class StableDiffusionImg2ImgPipeline:
            pass

        class StableDiffusionXLImg2ImgPipeline:
            pass

        assert (
            diffusion._native_size_for(StableDiffusionImg2ImgPipeline())
            == diffusion.SD_NATIVE_SIZE
        )
        assert (
            diffusion._native_size_for(StableDiffusionXLImg2ImgPipeline())
            == diffusion.SDXL_NATIVE_SIZE
        )

    def test_target_generation_size_upscales_square(self):
        from spriter.ai import diffusion

        assert diffusion._target_generation_size((16, 16), 512) == (512, 512)

    def test_target_generation_size_preserves_aspect(self):
        from spriter.ai import diffusion

        assert diffusion._target_generation_size((64, 32), 512) == (512, 256)

    def test_target_generation_size_snaps_to_multiple_of_8(self):
        from spriter.ai import diffusion

        w, h = diffusion._target_generation_size((100, 30), 512)
        assert w == 512
        assert h % 8 == 0

    def test_load_pipeline_uses_single_file_for_safetensors(
        self, tmp_path, monkeypatch
    ):
        import sys

        from spriter.ai import diffusion

        model_file = tmp_path / "model.safetensors"
        _write_safetensors_stub(
            model_file, ["cond_stage_model.transformer.text_model.weight"]
        )

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_cls = MagicMock()
        fake_diffusers = MagicMock()
        fake_diffusers.StableDiffusionImg2ImgPipeline = fake_cls

        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
        monkeypatch.setattr(diffusion, "is_available", lambda: True)

        diffusion.load_pipeline(model_file)

        fake_cls.from_single_file.assert_called_once()
        fake_cls.from_pretrained.assert_not_called()
        # Safety checker disabled by default to avoid false positives / crashes.
        _, kwargs = fake_cls.from_single_file.call_args
        assert kwargs["safety_checker"] is None
        assert kwargs["requires_safety_checker"] is False

    def test_load_pipeline_uses_from_pretrained_for_directory(
        self, tmp_path, monkeypatch
    ):
        import sys

        from spriter.ai import diffusion

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_cls = MagicMock()
        fake_diffusers = MagicMock()
        fake_diffusers.AutoPipelineForImage2Image = fake_cls

        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
        monkeypatch.setattr(diffusion, "is_available", lambda: True)

        diffusion.load_pipeline(model_dir)

        fake_cls.from_pretrained.assert_called_once()
        fake_cls.from_single_file.assert_not_called()

    def test_load_pipeline_can_keep_safety_checker(self, tmp_path, monkeypatch):
        import sys

        from spriter.ai import diffusion

        model_file = tmp_path / "model.safetensors"
        model_file.write_bytes(b"x")

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_cls = MagicMock()
        fake_diffusers = MagicMock()
        fake_diffusers.StableDiffusionImg2ImgPipeline = fake_cls

        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
        monkeypatch.setattr(diffusion, "is_available", lambda: True)

        diffusion.load_pipeline(model_file, disable_safety_checker=False)

        _, kwargs = fake_cls.from_single_file.call_args
        assert "safety_checker" not in kwargs
        assert "requires_safety_checker" not in kwargs

    def test_read_safetensors_header_keys(self, tmp_path):
        from spriter.ai import diffusion

        model_file = tmp_path / "m.safetensors"
        _write_safetensors_stub(model_file, ["a.weight", "b.bias"])
        keys = diffusion._read_safetensors_header_keys(model_file)
        assert set(keys) == {"a.weight", "b.bias"}

    def test_is_sdxl_checkpoint_detects_sdxl(self, tmp_path):
        from spriter.ai import diffusion

        model_file = tmp_path / "sdxl.safetensors"
        _write_safetensors_stub(
            model_file,
            [
                "conditioner.embedders.1.model.token_embedding.weight",
                "model.diffusion_model.input_blocks.0.0.weight",
            ],
        )
        assert diffusion._is_sdxl_checkpoint(model_file) is True

    def test_is_sdxl_checkpoint_false_for_sd(self, tmp_path):
        from spriter.ai import diffusion

        model_file = tmp_path / "sd.safetensors"
        _write_safetensors_stub(
            model_file,
            [
                "cond_stage_model.transformer.text_model.embeddings.weight",
                "model.diffusion_model.input_blocks.0.0.weight",
            ],
        )
        assert diffusion._is_sdxl_checkpoint(model_file) is False

    def test_is_sdxl_checkpoint_false_for_garbage(self, tmp_path):
        from spriter.ai import diffusion

        model_file = tmp_path / "bad.safetensors"
        model_file.write_bytes(b"x")
        assert diffusion._is_sdxl_checkpoint(model_file) is False

    def test_load_pipeline_routes_sdxl_checkpoint(self, tmp_path, monkeypatch):
        import sys

        from spriter.ai import diffusion

        model_file = tmp_path / "sdxl.safetensors"
        _write_safetensors_stub(
            model_file, ["conditioner.embedders.1.model.token_embedding.weight"]
        )

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        xl_cls = MagicMock()
        sd_cls = MagicMock()
        fake_diffusers = MagicMock()
        fake_diffusers.StableDiffusionXLImg2ImgPipeline = xl_cls
        fake_diffusers.StableDiffusionImg2ImgPipeline = sd_cls

        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
        monkeypatch.setattr(diffusion, "is_available", lambda: True)

        diffusion.load_pipeline(model_file)

        xl_cls.from_single_file.assert_called_once()
        sd_cls.from_single_file.assert_not_called()
        # SDXL has no safety checker component, so those kwargs are not passed.
        _, kwargs = xl_cls.from_single_file.call_args
        assert "safety_checker" not in kwargs

    def test_load_inpaint_pipeline_routes_sd(self, tmp_path, monkeypatch):
        import sys

        from spriter.ai import diffusion

        model_file = tmp_path / "sd.safetensors"
        _write_safetensors_stub(model_file, ["cond_stage_model.transformer.weight"])

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        sd_cls = MagicMock()
        xl_cls = MagicMock()
        fake_diffusers = MagicMock()
        fake_diffusers.StableDiffusionInpaintPipeline = sd_cls
        fake_diffusers.StableDiffusionXLInpaintPipeline = xl_cls

        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
        monkeypatch.setattr(diffusion, "is_available", lambda: True)

        diffusion.load_inpaint_pipeline(model_file)

        sd_cls.from_single_file.assert_called_once()
        xl_cls.from_single_file.assert_not_called()

    def test_load_inpaint_pipeline_routes_sdxl(self, tmp_path, monkeypatch):
        import sys

        from spriter.ai import diffusion

        model_file = tmp_path / "sdxl.safetensors"
        _write_safetensors_stub(
            model_file, ["conditioner.embedders.1.model.token_embedding.weight"]
        )

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        sd_cls = MagicMock()
        xl_cls = MagicMock()
        fake_diffusers = MagicMock()
        fake_diffusers.StableDiffusionInpaintPipeline = sd_cls
        fake_diffusers.StableDiffusionXLInpaintPipeline = xl_cls

        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
        monkeypatch.setattr(diffusion, "is_available", lambda: True)

        diffusion.load_inpaint_pipeline(model_file)

        xl_cls.from_single_file.assert_called_once()
        sd_cls.from_single_file.assert_not_called()

    def test_build_inpaint_row_layout_and_mask(self):
        from spriter.ai import diffusion

        f = np.zeros((8, 8, 4), dtype=np.uint8)
        f[..., 3] = 255
        image, mask, bbox = diffusion._build_inpaint_row([f, f, f], 8, 8, 512, seed=0)
        assert image.size == (512, 512)
        assert mask.size == (512, 512)
        # Mask marks exactly the final (NEW) cell of a 1x4 row padded to square.
        m = np.asarray(mask)
        x0, y0, w, h = bbox
        assert m[y0 : y0 + h, x0 : x0 + w].min() == 255
        # Region outside the NEW cell is unmasked.
        assert m[:, :x0].max() == 0

    def test_build_inpaint_row_pads_missing_frames(self):
        from spriter.ai import diffusion

        f = np.zeros((8, 8, 4), dtype=np.uint8)
        f[..., 3] = 255
        # Only one context frame supplied; should still build a valid square.
        image, mask, bbox = diffusion._build_inpaint_row([f], 8, 8, 512, seed=0)
        assert image.size == (512, 512)
        assert mask.size == (512, 512)

    def test_generate_next_frame_inpaints_and_crops(self):
        from PIL import Image

        from spriter.ai import diffusion

        native = 512
        out_img = Image.new("RGB", (native, native), (10, 20, 30))
        result = MagicMock()
        result.images = [out_img]
        pipeline = MagicMock(return_value=result)  # no "XL" -> SD, native 512

        f = np.zeros((16, 16, 4), dtype=np.uint8)
        f[..., 3] = 255
        out = diffusion.generate_next_frame(pipeline, [f, f, f], 16, 16, "walk")

        _, kwargs = pipeline.call_args
        assert "mask_image" in kwargs
        assert kwargs["image"].size == (native, native)
        # Output is the cropped NEW cell (RGBA).
        assert out.ndim == 3 and out.shape[2] == 4

    def test_model_info_for_safetensors_file(self, tmp_path):
        from spriter.ai import diffusion

        model_file = tmp_path / "model.safetensors"
        model_file.write_bytes(b"abcd")
        info = diffusion.model_info(model_file)
        assert info["exists"] is True
        assert info["kind"] == "file"
        assert ".safetensors" in info["type"]
        assert info["size_bytes"] == 4

    def test_model_info_for_directory(self, tmp_path):
        from spriter.ai import diffusion

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "a.bin").write_bytes(b"xy")
        (model_dir / "b.bin").write_bytes(b"z")
        info = diffusion.model_info(model_dir)
        assert info["kind"] == "directory"
        assert info["type"] == "Diffusers model folder"
        assert info["size_bytes"] == 3

    def test_model_info_for_missing_path(self, tmp_path):
        from spriter.ai import diffusion

        info = diffusion.model_info(tmp_path / "nope")
        assert info["exists"] is False
        assert info["kind"] == "missing"
        assert info["size_bytes"] == 0


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------


class TestSettingsPersistence:
    def test_diffusion_model_path_roundtrip(self, tmp_path):
        from spriter.core.settings import Settings

        s = Settings()
        s.diffusion_model_path = str(tmp_path / "model")
        path = tmp_path / "settings.json"
        s.save(path)
        loaded = Settings.load(path)
        assert loaded.diffusion_model_path == str(tmp_path / "model")

    def test_diffusion_model_path_default_empty(self):
        from spriter.core.settings import Settings

        assert Settings().diffusion_model_path == ""


# ---------------------------------------------------------------------------
# UI wiring
# ---------------------------------------------------------------------------


def _diffusion_menu(win):
    for action in win.menuBar().actions():
        if action.text().replace("&", "") == "Diffusion":
            return action.menu()
    return None


class TestDiffusionMenu:
    def test_menu_present_with_three_actions(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        menu = _diffusion_menu(win)
        assert menu is not None
        labels = [a.text().replace("&", "") for a in menu.actions() if a.text()]
        assert "Select Model\u2026" in labels
        assert "Download Model\u2026" in labels
        assert "Model Info\u2026" in labels
        assert "Generate Next Frame\u2026" in labels
        win.close()

    def test_model_info_shows_message_when_no_model(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        win._settings.diffusion_model_path = ""
        with patch("spriter.ui.main_window.QMessageBox.information") as info:
            win._diffusion_model_info()
        info.assert_called_once()
        win.close()

    def test_model_info_shows_details_for_selected_model(self, qapp, tmp_path):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        model_file = tmp_path / "model.safetensors"
        model_file.write_bytes(b"abcd")
        win._settings.diffusion_model_path = str(model_file)
        with patch("spriter.ui.main_window.QMessageBox.information") as info:
            win._diffusion_model_info()
        info.assert_called_once()
        shown_text = info.call_args[0][2]
        assert str(model_file) in shown_text
        assert ".safetensors" in shown_text
        win.close()

    def test_generate_shows_message_when_unavailable(self, qapp):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        with patch("spriter.ai.diffusion.is_available", return_value=False), patch(
            "spriter.ui.main_window.QMessageBox.information"
        ) as info:
            win._diffusion_generate_frame()
        info.assert_called_once()
        win.close()

    def test_select_model_saves_to_settings(self, qapp, tmp_path):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        model_dir = str(tmp_path / "mymodel")
        with patch(
            "spriter.ui.main_window.QInputDialog.getItem",
            return_value=("Model folder", True),
        ), patch(
            "spriter.ui.main_window.QFileDialog.getExistingDirectory",
            return_value=model_dir,
        ), patch("spriter.ui.main_window.QMessageBox.information"), patch(
            "spriter.core.settings.Settings.save"
        ):
            win._diffusion_select_model()
        assert win._settings.diffusion_model_path == model_dir
        win.close()

    def test_select_model_accepts_safetensors_file(self, qapp, tmp_path):
        from spriter.ui.main_window import MainWindow

        win = MainWindow()
        win._unsaved = False
        model_file = str(tmp_path / "model.safetensors")
        with patch(
            "spriter.ui.main_window.QInputDialog.getItem",
            return_value=("Single .safetensors file", True),
        ), patch(
            "spriter.ui.main_window.QFileDialog.getOpenFileName",
            return_value=(model_file, ""),
        ), patch("spriter.ui.main_window.QMessageBox.information"), patch(
            "spriter.core.settings.Settings.save"
        ):
            win._diffusion_select_model()
        assert win._settings.diffusion_model_path == model_file
        win.close()

    def test_generate_full_flow_inserts_frame(self, qapp, tmp_path):
        from spriter.ai import diffusion
        from spriter.ui.main_window import MainWindow, _DiffusionWorker

        win = MainWindow()
        win._unsaved = False
        assert win._sprite is not None
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        win._settings.diffusion_model_path = str(model_dir)
        before = win._sprite.frame_count

        generated = np.zeros((48, 48, 4), dtype=np.uint8)
        generated[..., :3] = 255  # white bg
        generated[10:30, 10:30] = (255, 0, 0, 255)  # red block

        # Run the worker synchronously so signals fire on the calling thread.
        with patch.object(
            _DiffusionWorker, "start", _DiffusionWorker.run
        ), patch.object(diffusion, "is_available", return_value=True), patch.object(
            diffusion, "load_inpaint_pipeline", return_value=MagicMock()
        ), patch.object(
            diffusion, "generate_next_frame", return_value=generated
        ), patch(
            "spriter.ui.main_window.QInputDialog.getText",
            return_value=("a hero", True),
        ):
            win._diffusion_generate_frame()

        assert win._sprite.frame_count == before + 1
        win._unsaved = False
        win.close()
