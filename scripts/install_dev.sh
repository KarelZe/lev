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

# The package was renamed (lev -> lev_rs), so never glob a name prefix: a
# stale wheel under the old name would shadow every fresh build. Take the
# newest wheel whose .so matches one already present in site-packages, so
# wheels built for a different interpreter are skipped as well.
wheels = sorted(glob.glob('target/wheels/*.whl'), key=os.path.getmtime, reverse=True)
if not wheels:
    raise RuntimeError("No wheel found in target/wheels/ — run maturin build first.")

for wheel in wheels:
    with zipfile.ZipFile(wheel) as z:
        so_entries = [n for n in z.namelist() if n.endswith('.so')]
        if not so_entries:
            continue
        so_name = pathlib.Path(so_entries[0]).name   # e.g. lev.cpython-313-darwin.so
        # Locate the .so on disk by scanning site-packages — never touches
        # __init__.py.
        candidates = [
            p for sp in site.getsitepackages()
            for p in pathlib.Path(sp).glob(f"lev/{so_name}")
        ]
        if not candidates:
            continue
        data = z.read(so_entries[0])
        dest = candidates[0]
        dest.write_bytes(data)
        print(f'Installed {len(data):,} bytes → {dest} (from {os.path.basename(wheel)})')
        break
else:
    raise RuntimeError(
        "No wheel matching the installed extension found — run 'uv sync' first."
    )
EOF
