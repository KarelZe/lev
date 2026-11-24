import Std.Tactic.BVDecide

/-!
# Equivalence gate for `myers_step` forms

Every form of the single-word Levenshtein step must compute the same
`(vp, vn)` as `portable::myers_step` on every reachable state
(`vp &&& vn = 0`). `bv_decide` decides this for all 2^192 inputs with a SAT
solver and checks the solver's LRAT certificate with Lean's verified checker.
The checker runs as compiled code, so each proof also trusts the Lean
compiler (`#print axioms` lists a `bv_decide` native axiom).

Each `def` below is a hand transcription of one Rust or asm form. The
Rust test `myers_step_matches_portable` (src/kernels/tests.rs) checks the
compiled forms against the reference on sampled states, which guards the
transcription.

To gate a new candidate: add a `def`, add a theorem in the same shape, run
`lean MyersStep.lean` from this directory. No output means every theorem
holds. A refuted theorem prints a counterexample.

AArch64 asm in Lean terms: `orn d, a, b` is `a ||| ~~~b`, `bic d, a, b` is
`a &&& ~~~b`, `eor d, a, b` is `a ^^^ b`, `eon d, a, b` is `a ^^^ ~~~b`,
`lsl d, a, #1` is `a <<< 1`.
-/

abbrev W := BitVec 64

/-- `portable::myers_step` (src/kernels/portable.rs), the reference. -/
def ref (eq vp vn : W) : W × W :=
  let sum := (eq &&& vp) + vp
  let ph := vn ||| ~~~(sum ||| (vp ||| eq))
  let mh := vp &&& (eq ||| ~~~sum)
  let xv := eq ||| vn
  let phS := (ph <<< 1) ||| 1
  ((mh <<< 1) ||| ~~~(xv ||| phS), phS &&& xv)

/-- `aarch64::neoverse::myers_step_chain6`. -/
def neoverse (eq vp vn : W) : W × W :=
  let sum := (eq &&& vp) + vp
  let x := sum ||| vp ||| eq
  let mh := vp &&& (eq ||| ~~~sum)
  let xv := eq ||| vn
  let m := xv ||| (vn <<< 1)
  let xs := x <<< 1
  let s := xs &&& ~~~(vn <<< 1)
  let a := xs &&& ~~~m
  ((mh <<< 1) ||| a, xv &&& ~~~s)

/-- `x86_64::bmi1::myers_step_andn_chain6`. -/
def andn (eq vp vn : W) : W × W :=
  let sum := (eq &&& vp) + vp
  let x := sum ||| (vp ||| eq)
  let mh := vp &&& ~~~(sum &&& ~~~eq)
  let xv := eq ||| vn
  let vns := vn <<< 1
  let xs := x <<< 1
  let m := xv ||| vns
  let s := ~~~vns &&& xs
  let a := ~~~m &&& xs
  ((mh <<< 1) ||| a, ~~~s &&& xv)

/-- `aarch64::apple::myers_step_chain6`, transcribed from its asm block. -/
def apple (eq vp vn : W) : W × W :=
  let sum := eq &&& vp
  let u := eq ||| vp
  let xv := eq ||| vn
  let vns := vn <<< 1
  let sum := sum + vp
  let m := xv ||| vns
  let x := u ||| sum
  let t := eq ||| ~~~sum
  let xs := x <<< 1
  let mh := t &&& vp
  let mhs := mh <<< 1
  let s := xs &&& ~~~vns
  let a := xs &&& ~~~m
  (mhs ||| a, xv &&& ~~~s)

/-- `aarch64::apple::myers_step`, transcribed from its asm block, one `let`
per instruction (the asm reuses registers; the names here are the values). -/
def appleChain5 (eq vp vn : W) : W × W :=
  let e := eq <<< 1
  let p := vp <<< 1
  let xv := eq ||| vn
  let n := vn <<< 1
  let t := e &&& p
  let xe := xv ^^^ ~~~p
  let en := xv <<< 1
  let xp := xv &&& ~~~p
  let s := t + p
  let d := xe &&& ~~~en
  let g := xp &&& ~~~en
  let m := xv ||| n
  let y := d &&& ~~~s
  let gs := g &&& ~~~s
  let a := t ||| ~~~m
  let xn := xv &&& n
  (a ^^^ y, xn ||| gs)

/-- `aarch64::neoverse::myers_step`: chain 5, shifts in shifted operands. -/
def neoverseChain5 (eq vp vn : W) : W × W :=
  let xv := eq ||| vn
  let u := eq &&& vp
  let sum := u + vp
  let m := xv ||| (vn <<< 1)
  let xe := xv ^^^ ~~~(vp <<< 1)
  let g := xv &&& ~~~((eq ||| vp) <<< 1)
  let xn := xv &&& (vn <<< 1)
  let na := m &&& ~~~(u <<< 1)
  let d := xe &&& ~~~(xv <<< 1)
  let gs := g &&& ~~~(sum <<< 1)
  let y := d &&& ~~~(sum <<< 1)
  (y ^^^ ~~~na, xn ||| gs)

/-- `x86_64::bmi1::myers_step_andn`: chain 5, negations as `!a & b`. -/
def andnChain5 (eq vp vn : W) : W × W :=
  let xv := eq ||| vn
  let e := eq <<< 1
  let p := vp <<< 1
  let n := vn <<< 1
  let t := e &&& p
  let s := t + p
  let na := ~~~t &&& (xv ||| n)
  let z := (xv ^^^ p) ||| (xv <<< 1) ||| s
  let gs := ~~~s &&& (~~~(e ||| p) &&& xv)
  (na ^^^ z, (xv &&& n) ||| gs)

/-- One word of `portable::multiword` (Hyyrö's form) with no carry in and
the row-0 edge shifted in (`prev_hp = 1`, `prev_hn = 0`). -/
def hyyro (eq vp vn : W) : W × W :=
  let x := eq ||| vn
  let sum := (x &&& vp) + vp
  let d0 := (sum ^^^ vp) ||| x
  let hp := vn ||| ~~~(d0 ||| vp)
  let hn := vp &&& d0
  let hpS := (hp <<< 1) ||| 1
  let hnS := hn <<< 1
  (hnS ||| ~~~(d0 ||| hpS), hpS &&& d0)

theorem neoverse_eq (eq vp vn : W) (h : vp &&& vn = 0) :
    neoverse eq vp vn = ref eq vp vn := by
  simp only [neoverse, ref, Prod.mk.injEq]
  bv_decide

theorem andn_eq (eq vp vn : W) (h : vp &&& vn = 0) :
    andn eq vp vn = ref eq vp vn := by
  simp only [andn, ref, Prod.mk.injEq]
  bv_decide

theorem apple_eq (eq vp vn : W) (h : vp &&& vn = 0) :
    apple eq vp vn = ref eq vp vn := by
  simp only [apple, ref, Prod.mk.injEq]
  bv_decide

theorem appleChain5_eq (eq vp vn : W) (h : vp &&& vn = 0) :
    appleChain5 eq vp vn = ref eq vp vn := by
  simp only [appleChain5, ref, Prod.mk.injEq]
  bv_decide

theorem neoverseChain5_eq (eq vp vn : W) (h : vp &&& vn = 0) :
    neoverseChain5 eq vp vn = ref eq vp vn := by
  simp only [neoverseChain5, ref, Prod.mk.injEq]
  bv_decide

theorem andnChain5_eq (eq vp vn : W) (h : vp &&& vn = 0) :
    andnChain5 eq vp vn = ref eq vp vn := by
  simp only [andnChain5, ref, Prod.mk.injEq]
  bv_decide

theorem hyyro_eq (eq vp vn : W) (h : vp &&& vn = 0) :
    hyyro eq vp vn = ref eq vp vn := by
  simp only [hyyro, ref, Prod.mk.injEq]
  bv_decide
