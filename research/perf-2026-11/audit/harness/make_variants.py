# ruff: file-ignore[undocumented-public-function]
"""Create one source copy per audit variant: base with exactly one layer disabled or re-gated."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
BASE = AUDIT / "base"

# name -> list of (relative file, old, new); each old must occur exactly once.
VARIANTS: dict[str, list[tuple]] = {
    # V1: wide (UCS-2/4) patterns of 9-16 units leave the tiny path (u8 patterns keep TINY_M = 16).
    "tinywide8": [
        (
            "src/lib.rs",
            "fn compute_sorted<K: Kernel, T: CodeUnit>(a: &[T], b: &[T]) -> usize {\n    let (a, b) = strip_affix(a, b);\n    if a.is_empty() {\n        return b.len();\n    }\n    if a.len() <= TINY_M {",
            "fn compute_sorted<K: Kernel, T: CodeUnit>(a: &[T], b: &[T]) -> usize {\n    let (a, b) = strip_affix(a, b);\n    if a.is_empty() {\n        return b.len();\n    }\n    if a.len() <= 8 {",
        ),
        (
            "src/lib.rs",
            "    let (a, b) = strip_affix_mixed(a, b);\n    if a.is_empty() {\n        return b.len();\n    }\n    if a.len() <= TINY_M {",
            "    let (a, b) = strip_affix_mixed(a, b);\n    if a.is_empty() {\n        return b.len();\n    }\n    if a.len() <= if std::mem::size_of::<T1>() == 1 { TINY_M } else { 8 } {",
        ),
    ],
    # V2: no small_ub gate and no mbleven for distance.
    "nombleven": [
        (
            "src/lib.rs",
            "    type Isa = X;\n\n    const MBLEVEN: bool = true;",
            "    type Isa = X;\n\n    const MBLEVEN: bool = false;",
        ),
    ],
    # V3: no speculative small bands (#260) for distance.
    "noband": [
        (
            "src/lib.rs",
            "    const MBLEVEN: bool = true;\n\n    const SMALL_BAND: bool = true;",
            "    const MBLEVEN: bool = true;\n\n    const SMALL_BAND: bool = false;",
        ),
    ],
    # V4: no table-free NEON match masks (#266) on Apple.
    "notablefree": [
        (
            "src/kernels/mod.rs",
            '    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]\n    if short_text {',
            '    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]\n    if short_text && false {',
        ),
    ],
    # V5: single-word hash always 128 slots (undo adaptive 64/128 sizing).
    "hash128": [
        (
            "src/lib.rs",
            "    if m <= 32 {\n        hyrro_64_hash_slots::<K, 64, T1, T2>(pattern, text)",
            "    if false {\n        hyrro_64_hash_slots::<K, 64, T1, T2>(pattern, text)",
        ),
    ],
    # V6: u8 tiny patterns of 1-8 bytes use the NEON compare instead of SWAR on AArch64.
    "noswar": [
        (
            "src/lib.rs",
            "        if pattern.len() <= 8 {\n            return tiny_swar::<K, _, _>(pattern, text);\n        }",
            '        #[cfg(not(target_arch = "aarch64"))]\n        if pattern.len() <= 8 {\n            return tiny_swar::<K, _, _>(pattern, text);\n        }',
        ),
    ],
    # V7: LargeCtx without band passes (Levenshtein and LCS go straight to the full matrix).
    "nolargeband": [
        (
            "src/lib.rs",
            "        let mut t = (self.n - self.m + 32).max(64);\n        if band_blocks(t) * 3 > self.w {",
            "        let mut t = (self.n - self.m + 32).max(64);\n        if true || band_blocks(t) * 3 > self.w {",
        ),
        (
            "src/lib.rs",
            "        let mut t = 64;\n        if lcs_band_words(slack, t) * 3 > self.w {",
            "        let mut t = 64;\n        if true || lcs_band_words(slack, t) * 3 > self.w {",
        ),
    ],
    # V8: Apple single-word step in plain Rust (Neoverse form) instead of the pinned asm! step.
    "noasmstep": [
        (
            "src/kernels/mod.rs",
            '    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]\n    return aarch64::apple::myers_step(eq, vp, vn);',
            '    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]\n    return aarch64::neoverse::myers_step(eq, vp, vn);',
        ),
    ],
    # V2b: mbleven only for ub <= 2, or ub == 3 on patterns longer than one word.
    "mb3gate": [
        (
            "src/lib.rs",
            "if let Some(ub) = small_ub::<K::Isa, _>(a, b) {",
            "if let Some(ub) = small_ub::<K::Isa, _>(a, b).filter(|&ub| ub < MBLEVEN_MAX || a.len() > 64) {",
            2,
        ),
        (
            "src/lib.rs",
            "if let Some(ub) = small_ub_mixed(a, b) {",
            "if let Some(ub) = small_ub_mixed(a, b).filter(|&ub| ub < MBLEVEN_MAX || a.len() > 64) {",
        ),
    ],
    # V3b: run a small band only when it can certify >= 32 edits beyond the length difference.
    "bands32": [
        (
            "src/lib.rs",
            "    if s < 64 {\n        // SAFETY: forwarded caller contract.",
            "    if s < 32 {\n        // SAFETY: forwarded caller contract.",
        ),
        ("src/lib.rs", "if w < SMALL_BAND2_MIN_W || s >= 128 {", "if w < SMALL_BAND2_MIN_W || s >= 96 {"),
    ],
    # V7b: after a failed first LargeCtx band pass, allow a next pass covering up to 2/3 of the words.
    "large2pass": [
        (
            "src/lib.rs",
            "        while band_blocks(t) * 3 <= self.w {",
            "        let mut cap = self.w;\n        while band_blocks(t) * 3 <= cap {",
        ),
        (
            "src/lib.rs",
            "        while lcs_band_words(slack, t) * 3 <= self.w {",
            "        let mut cap = self.w;\n        while lcs_band_words(slack, t) * 3 <= cap {",
        ),
        (
            "src/lib.rs",
            "                Err(j) => t = next_band(t, j, self.n),",
            "                Err(j) => {\n                    t = next_band(t, j, self.n);\n                    cap = 2 * self.w;\n                }",
            2,
        ),
    ],
    # V2c: skip mbleven only for its two largest model sets (ub = 3, diff <= 1) on patterns of <= 32 units.
    "mb3gate32": [
        (
            "src/lib.rs",
            "if let Some(ub) = small_ub::<K::Isa, _>(a, b) {",
            "if let Some(ub) = small_ub::<K::Isa, _>(a, b).filter(|&ub| ub < MBLEVEN_MAX || a.len() > 32 || b.len() - a.len() >= 2) {",
            2,
        ),
        (
            "src/lib.rs",
            "if let Some(ub) = small_ub_mixed(a, b) {",
            "if let Some(ub) = small_ub_mixed(a, b).filter(|&ub| ub < MBLEVEN_MAX || a.len() > 32 || b.len() - a.len() >= 2) {",
        ),
    ],
    # V7c: certified LCS band only when its first pass covers at most a fifth of the words (m >~ 1216).
    "lcsgate5": [
        (
            "src/lib.rs",
            "        if lcs_band_words(slack, t) * 3 > self.w {",
            "        if lcs_band_words(slack, t) * 5 > self.w {",
        ),
        (
            "src/lib.rs",
            "        while lcs_band_words(slack, t) * 3 <= self.w {",
            "        while lcs_band_words(slack, t) * 5 <= self.w {",
        ),
    ],
    # V7d: resolve the text's row offsets once (as the band passes do) also when LargeCtx goes straight to the full matrix.
    "basesalways": [
        (
            "src/lib.rs",
            "        if band_blocks(t) * 3 > self.w {\n            return kernels::full(self);\n        }\n        let bases: Vec<usize> = (0..self.n).map(&self.base_of).collect();\n        let ctx = self.with_bases(&bases);",
            "        let bases: Vec<usize> = (0..self.n).map(&self.base_of).collect();\n        let ctx = self.with_bases(&bases);\n        if band_blocks(t) * 3 > self.w {\n            return kernels::full(&ctx);\n        }",
        ),
        (
            "src/lib.rs",
            "        if lcs_band_words(slack, t) * 3 > self.w {\n            return kernels::lcs_full::<X, _>(self);\n        }\n        let bases: Vec<usize> = (0..self.n).map(&self.base_of).collect();\n        let ctx = self.with_bases(&bases);",
            "        let bases: Vec<usize> = (0..self.n).map(&self.base_of).collect();\n        let ctx = self.with_bases(&bases);\n        if lcs_band_words(slack, t) * 3 > self.w {\n            return kernels::lcs_full::<X, _>(&ctx);\n        }",
        ),
    ],
    # R1 probe (not a candidate): the PyO3 entry points return right after argument extraction.
    "r1probe": [
        (
            "src/python.rs",
            "    if s1.is(s2) {\n        return 0;\n    }\n    compute::<Lev<X>>(&view(&s1), &view(&s2))",
            "    if s1.is(s2) {\n        return 0;\n    }\n    let _ = (view, compute::<Lev<X>>);\n    0",
        ),
        (
            "src/python.rs",
            "    if s1.is(s2) {\n        return 1.0;\n    }\n    let (v1, v2) = (view(&s1), view(&s2));",
            "    if s1.is(s2) || true {\n        return 1.0;\n    }\n    let (v1, v2) = (view(&s1), view(&s2));",
        ),
    ],
}


def make(name: str, edits: list[tuple]) -> Path:
    dst = AUDIT / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(BASE, dst, ignore=shutil.ignore_patterns("wheels", ".git", "target"))
    for edit in edits:
        rel, old, new = edit[:3]
        want = edit[3] if len(edit) > 3 else 1
        p = dst / rel
        text = p.read_text()
        count = text.count(old)
        if count != want:
            raise SystemExit(f"{name}: {rel}: expected 1 match, found {count}:\n{old}")
        p.write_text(text.replace(old, new))
    diff = subprocess.run(["diff", "-ru", str(BASE / "src"), str(dst / "src")], capture_output=True, text=True).stdout
    (AUDIT / "patches").mkdir(exist_ok=True)
    (AUDIT / "patches" / f"{name}.diff").write_text(diff)
    return dst


if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    for n in names:
        print("made", make(n, VARIANTS[n]))
