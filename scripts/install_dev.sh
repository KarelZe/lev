#!/usr/bin/env bash
# Build the release wheel and copy the .so directly into the venv,
# bypassing uv's version-based wheel cache (which skips reinstall when
# the version string hasn't changed).
set -euo pipefail

uv run maturin build --release

# Single-quoted EOF: no shell expansion inside the Python script.
# All path handling stays in Python; nothing is interpolated into source code.
uv run python - <<'EOF'
import glob, os, pathlib, site, zipfile

wheels = glob.glob('target/wheels/lev-*.whl')
if not wheels:
    raise RuntimeError("No wheel found in target/wheels/ — run maturin build first.")
wheel = max(wheels, key=os.path.getmtime)

with zipfile.ZipFile(wheel) as z:
    so_entries = [n for n in z.namelist() if n.endswith('.so')]
    if not so_entries:
        raise RuntimeError(f"No .so file found in {wheel}")
    so_name = pathlib.Path(so_entries[0]).name   # e.g. lev.cpython-313-darwin.so
    data = z.read(so_entries[0])

# Locate the .so on disk by scanning site-packages — never touches __init__.py.
candidates = [
    p for sp in site.getsitepackages()
    for p in pathlib.Path(sp).glob(f"lev/{so_name}")
]
if not candidates:
    raise RuntimeError(
        f"{so_name} not found in site-packages — run 'uv sync' first."
    )
dest = candidates[0]
dest.write_bytes(data)
print(f'Installed {len(data):,} bytes → {dest}')
EOF
