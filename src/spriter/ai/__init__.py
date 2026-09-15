# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Optional AI features — diffusion-based frame prediction.

The heavy machine-learning dependencies (``torch``, ``diffusers``) live behind
the ``spriter[diffusion]`` optional extra.  Modules in this package import them
lazily so the base application remains usable without them installed.
"""
