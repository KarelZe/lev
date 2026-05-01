#!/usr/bin/env bash
# Build the release wheel and copy the .so directly into the venv,
# bypassing uv's version-based wheel cache (which skips reinstall when
# the version string hasn't changed).
set -euo pipefail

uv run maturin build --release

SO_SRC=$(uv run python -c "
import zipfile, glob
wheels = glob.glob('target/wheels/lev-*.whl')
w = max(wheels, key=__import__('os').path.getmtime)
with zipfile.ZipFile(w) as z:
    names = [n for n in z.namelist() if n.endswith('.so')]
    print(names[0] + '|' + w)
")
NAME="${SO_SRC%%|*}"
WHEEL="${SO_SRC##*|}"

DEST=$(uv run python -c "import lev.lev; print(lev.lev.__file__)")
uv run python -c "
import zipfile
with zipfile.ZipFile('$WHEEL') as z:
    data = z.read('$NAME')
    with open('$DEST', 'wb') as f:
        f.write(data)
print('Installed', len(data), 'bytes →', '$DEST')
"
