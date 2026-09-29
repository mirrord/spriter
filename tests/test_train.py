# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Tests for automatic fine-tuning orchestration (heavy training is mocked)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from spriter.core.sprite import Sprite


def _frames(n, size=8):
    out = []
    for i in range(n):
        arr = np.zeros((size, size, 4), dtype=np.uint8)
        arr[:, :, i % 3] = 255
        arr[:, :, 3] = 255
        out.append(arr)
    return out


class TestAvailabilityAndPaths:
    def test_is_train_available_returns_bool(self):
        from spriter.ai import train

        assert isinstance(train.is_train_available(), bool)

    def test_adapter_dir_for_is_stable_and_scoped(self):
        from spriter.ai import train

        a = train.adapter_dir_for("/projects/hero.spr")
        b = train.adapter_dir_for("/projects/hero.spr")
        c = train.adapter_dir_for("/projects/villain.spr")
        assert a == b
        assert a != c
        assert train.default_adapter_root() in a.parents

    def test_has_trained_adapter_detects_weight_file(self, tmp_path):
        from spriter.ai import train

        assert train.has_trained_adapter(tmp_path) is False
        (tmp_path / train.ADAPTER_WEIGHT_NAME).write_bytes(b"x")
        assert train.has_trained_adapter(tmp_path) is True


class TestTrainOrchestration:
    def test_requires_dependencies(self, tmp_path):
        from spriter.ai import train

        with patch.object(train, "is_train_available", return_value=False):
            with pytest.raises(RuntimeError):
                train.train_next_frame_predictor(
                    tmp_path / "model", _frames(8), tmp_path / "adapter"
                )

    def test_rejects_too_few_frames(self, tmp_path):
        from spriter.ai import train

        with patch.object(train, "is_train_available", return_value=True):
            with pytest.raises(ValueError):
                train.train_next_frame_predictor(
                    tmp_path / "model", _frames(2), tmp_path / "adapter"
                )

    def test_creates_output_and_invokes_training(self, tmp_path):
        from spriter.ai import train

        adapter = tmp_path / "adapter"
        with patch.object(train, "is_train_available", return_value=True), patch.object(
            train, "_run_lora_training"
        ) as run:
            result = train.train_next_frame_predictor(
                tmp_path / "model", _frames(8), adapter, steps=10
            )
        assert result == adapter
        assert adapter.is_dir()
        run.assert_called_once()
        # Auto-defaults are threaded through without user configuration.
        _, kwargs = run.call_args
        assert kwargs["steps"] == 10
        assert kwargs["rank"] == train.DEFAULT_LORA_RANK

    def test_auto_train_harvests_then_trains(self, tmp_path):
        from spriter.ai import train

        sprite = Sprite(8, 8)
        sprite.add_layer("Layer")
        for i, pixels in enumerate(_frames(8)):
            sprite.add_frame()
            sprite.set_cel_pixels(0, i, pixels)

        with patch.object(train, "is_train_available", return_value=True), patch.object(
            train, "_run_lora_training"
        ) as run:
            train.auto_train_from_sprite(sprite, tmp_path / "model", tmp_path / "ad")
        run.assert_called_once()
        # The harvested frames (not a raw sprite) are what gets trained on.
        args, _ = run.call_args
        assert isinstance(args[1], list)
        assert len(args[1]) == 8

    def test_progress_callback_threaded_through(self, tmp_path):
        from spriter.ai import train

        seen = []

        def cb(step, total):
            seen.append((step, total))

        with patch.object(train, "is_train_available", return_value=True), patch.object(
            train, "_run_lora_training"
        ) as run:
            train.train_next_frame_predictor(
                tmp_path / "model", _frames(8), tmp_path / "ad", progress_cb=cb
            )
        _, kwargs = run.call_args
        assert kwargs["progress_cb"] is cb
