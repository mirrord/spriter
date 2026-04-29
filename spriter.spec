# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Spriter
# Build with:  hatch run dist:build
# Or directly: pyinstaller spriter.spec --noconfirm

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['launcher.py'],
    pathex=[str(Path('src').resolve())],
    binaries=[],
    datas=[],
    hiddenimports=[
        # PyQt6 widgets / modules that PyInstaller may miss
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.sip',
        # PIL/Pillow image plugins needed for PNG, GIF, ICO I/O
        'PIL.Image',
        'PIL.PngImagePlugin',
        'PIL.GifImagePlugin',
        'PIL.IcoImagePlugin',
        'PIL.BmpImagePlugin',
        # NumPy internals that are often missed
        'numpy.lib.format',
        # Spriter internal packages (src layout requires explicit listing)
        'spriter',
        'spriter.app',
        'spriter.core',
        'spriter.core.animation',
        'spriter.core.compositor',
        'spriter.core.frame',
        'spriter.core.layer',
        'spriter.core.palette',
        'spriter.core.settings',
        'spriter.core.sprite',
        'spriter.commands',
        'spriter.commands.base',
        'spriter.commands.draw',
        'spriter.commands.frame_ops',
        'spriter.commands.layer_ops',
        'spriter.commands.transform',
        'spriter.io',
        'spriter.io.gif_io',
        'spriter.io.png_io',
        'spriter.io.project_io',
        'spriter.io.spritesheet',
        'spriter.tools',
        'spriter.tools.base',
        'spriter.tools.ellipse',
        'spriter.tools.eraser',
        'spriter.tools.eyedropper',
        'spriter.tools.fill',
        'spriter.tools.line',
        'spriter.tools.move',
        'spriter.tools.pencil',
        'spriter.tools.rectangle',
        'spriter.tools.select',
        'spriter.tools.text',
        'spriter.ui',
        'spriter.ui.canvas',
        'spriter.ui.color_picker',
        'spriter.ui.layers_panel',
        'spriter.ui.main_window',
        'spriter.ui.preferences',
        'spriter.ui.preview',
        'spriter.ui.timeline',
        'spriter.ui.toolbar',
        'spriter.utils',
        'spriter.utils.geometry',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='spriter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # GUI app — no console window on Windows
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icon.ico',  # Uncomment and supply an icon file to embed one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='spriter',
)
