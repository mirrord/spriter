# Building a standalone .exe

Spriter uses [PyInstaller](https://pyinstaller.org/) (driven by a Hatch environment) to package the app as a standalone Windows executable.

## Prerequisites

- Hatch installed and the project's dependencies resolvable (`hatch shell` works)
- Windows (the spec targets a Windows build; `console=False` hides the terminal window)

## Build

From the repo root:

```console
# Folder build (dist/spriter/spriter.exe + dependent files)
hatch run dist:build

# Single-file build (dist/spriter.exe, slower to start, nothing else to ship)
hatch run dist:build-onefile
```

Both commands install `pyinstaller` into the `dist` Hatch environment on first run. `dist:build` invokes `pyinstaller spriter.spec --noconfirm` directly; `dist:build-onefile` sets the `SPRITER_ONEFILE=1` environment variable before running PyInstaller with the same spec (PyInstaller rejects the `--onefile`/`--onedir` CLI flags once a `.spec` file is given, so [spriter.spec](spriter.spec) branches on that env var instead to switch between a one-dir `COLLECT` build and a single-file `EXE`).

The build entry point is [launcher.py](launcher.py), which imports `spriter.app.main` directly (avoids relying on the installed console-script entry point). Packaging config — hidden imports, icon, excludes — lives in [spriter.spec](spriter.spec).

## Output

- Folder build: `dist/spriter/spriter.exe` (plus supporting DLLs/data alongside it — ship the whole `dist/spriter/` folder)
- One-file build: `dist/spriter.exe` (single portable file)

## Troubleshooting

- **Missing module errors at runtime**: add the module to `hiddenimports` in [spriter.spec](spriter.spec) and rebuild.
- **Stale code in the build**: PyInstaller bundles whatever is importable in the active environment — run `hatch run pip install -e .` first if you've made source changes and the build doesn't reflect them.
- **Icon**: the spec embeds `assets/sprite.ico`; replace that file to change the app icon.
