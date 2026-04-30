# SPDX-FileCopyrightText: 2026-present Dane Howard <mirrord@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Embedded application resources (icons, etc.) as base64 bytes.

Keeping assets embedded in Python avoids filesystem path issues in
both development mode and PyInstaller frozen bundles.
"""

import base64

# assets/sprite.ico — 18×18 RGBA PNG-in-ICO
_SPRITE_ICO_B64 = (
    "AAABAAEAEhIQAAEABAATAQAAFgAAAIlQTkcNChoKAAAADUlIRFIAAAASAAAAEggGAAAAVs6O"
    "VwAAANpJREFUOI2tk9ENwyAQQ43UYcxGHYdxslHYxv0od7kAIW1VIhSJ4HfGORIg/GM8"
    "br4LQCKzV6t1T7YeN6YPHEk6NDkzAvui80lSJAVIEk6zrYf9I0AGiKL+PeomkL66iwsU"
    "XF6CpAIpbO4hEBzWg4a/lrcW5l492Di4cRb04cgtt6pWWQUqWB1J8qM5oD0mKnhD7LgY"
    "g9aQkYmjiwiNOVlLXP61U9AteHdjzs8w1w6dTWbVZwW3I+gYrl2XZWfHwGPfrLp/mpED"
    "Fv1yC5o6+hgyv2v4HjIJ+9fxAsRNVnfwCn6jAAAAAElFTkSuQmCC"
)

SPRITE_ICO_BYTES: bytes = base64.b64decode(_SPRITE_ICO_B64)
