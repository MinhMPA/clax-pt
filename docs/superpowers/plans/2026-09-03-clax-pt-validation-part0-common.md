# clax-pt Validation — Part 0: Shared Constraints, Environments, Run Recipes, Task DAG

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Every task in Parts 1a, 1b and 2 inherits this file.** Read it first, then the one task you were assigned, then the reference sections that task cites.

**Goal:** Validate clax-pt (`clax/ept.py`) against CLASS-PT for P_ℓ(k), ℓ=0,2,4, on 15 cosmologies (5 ΛCDM + 5 νΛCDM + 5 w0waCDM) at z ∈ {0, 0.38, 0.8}, with CLASS-PT's in-loop Alcock–Paczynski remap mirrored inside clax-pt and every discrepancy traced to a term, never fudged.

**Architecture:** Track A builds the CLASS-PT oracle (dedicated `classpt` env, patched accessors, reference generator, provenance gate). Track B refactors `clax/ept.py` at α=1 into a single GL μ-loop that consumes a remappable channel stack, fixes three known defects, then adds the in-loop AP remap. Track P adds `PerturbationResult.delta_cb`. Track C (Part 2) wires `compute_ept_from_clax(..., omfid, field)`, the multi-cosmology stage and end-to-end tests, the V100 campaign job, thresholds, and the report.

**Tech Stack:** JAX 0.9.2 (`clax` env), CLASS-PT `09d5531a` built into env `classpt` (Python 3.10, NumPy < 2, Cython < 3, OpenBLAS from conda-forge), pytest, SLURM on the igpu cluster.

**Spec:** `docs/superpowers/specs/2026-09-03-clax-pt-validation-design.md` (authoritative; §-numbers below refer to it unless prefixed "ref").

**Companion reference:** `docs/superpowers/plans/2026-09-03-clax-pt-validation-classpt-inloop-reference.md` — CLASS-PT internals transcribed with line numbers ("ref §N"). Tasks cite the sections they need; read those, not the whole file.

**Plan files:**

| File | Track | Tasks |
|---|---|---|
| this file | — | constraints, environments, recipes, DAG |
| `2026-09-03-clax-pt-validation-part1a-oracle.md` | A | A1 cosmology table · A2 env+build · A3 generator+assembly · A4 provenance gate · A5 refgen job+MANIFEST |
| `2026-09-03-clax-pt-validation-part1b-ept-ap.md` | B, P | B1 baseline+guards · B2 `ap_ratios` · B3 α=1 loop refactor · B4 bug fixes · B5 in-loop AP · B6 AP gradients · B7 assembly tests · P1 `delta_cb` |
| `2026-09-03-clax-pt-validation-part2-campaign.md` | C | C0 `compute_ept_from_clax` seam · C1 stage tests · C2 e2e tests · C3 campaign job+report · C4 ratchet+docs+PR |

---

## Global Constraints

Copied from the spec and the standing user/cluster rules. Every task's requirements include this section.

**Worktree and git**
- Work in `/home/n2minh/clax-ptval` on branch `campaign/clax-pt-validation` (cut from `origin/main` @ `9198580`; spec at `bf8ac18`). Each Bash call starts in `/home/n2minh/clax`; prefix commands with `cd /home/n2minh/clax-ptval &&`.
- Commit messages via `git commit -F <file>` (no heredocs). Every message ends with the two trailers:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
  ```
- One logical change per commit. Draft PR only at the end (Part 2, C4); no merges, no force-push, no branch deletion.
- CHANGELOG date headings use `Mon D, YYYY` (e.g. `### Sep 3, 2026: ...`).
- `git fetch/pull/push/checkout` run on the login node only — compute nodes have no SSH key.

**Environments (spec §4.8)**
- `clax` env: `/home/n2minh/micromamba/envs/clax/bin/python` (Python 3.14, JAX 0.9.2, NumPy 2). Importing clax from the worktree needs `PYTHONPATH=/home/n2minh/clax-ptval`.
- `classpt` env: created by Task A2 (recipe there). CLASS-PT lives at `/home/n2minh/CLASS-PT` (commit `09d5531a`, currently unbuilt).
- Never touch envs `base`, `carpile`, `cosmodesi`, `cosmopower`, `fli-mf-nuts`, `lmstudio`, and never install CLASS-PT into them.
- Never touch SLURM jobs you did not submit; in particular never touch jobs named `fnl_closure_resumable`, `p2_d*_real`, `p2_d*_fourier`.

**Compute placement**
- Login node: bounded CPU probes only (≤ 2 minutes, ≤ 2 threads):
  `JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval /home/n2minh/micromamba/envs/clax/bin/python ...`
- Everything heavier (any `perturbations_solve`, the full `pytest tests/ --fast`, CLASS-PT reference generation, the campaign) goes through sbatch (templates below).

**Physics rules (CLAUDE.md)**
- No fudge factors. A failing threshold means a term is wrong: bisect (spec §8) and fix the term.
- Every equation carries a `nonlinear_pt.c:<line>` / `classy.pyx:<line>` comment.
- Physics-facing tests run at ≥ 3 cosmologies in full mode (`lcdm_cosmology` / `nulcdm_cosmology` / new `ept_case` fixtures); `--fast` prunes to the spec's three (`lcdm_fiducial`, `massive_nu_015`, `w0wa_m07_m10` at z=0.38).
- Thresholds (spec §4.7): `|Δ|/max_k|P_ref|` ≤ 1% for ℓ=0,2 and real-space, ≤ 2% for ℓ=4, over `k_h[10] ≤ k ≤ 0.3 h/Mpc` (Nside window, ref §2); seams: `f` 1e-3 rel, `P_cb,lin` 0.1% rel for k ≤ 0.3, `hratio`/`Dratio` 1e-4 rel. After the campaign, thresholds ratchet to ≥ 2× the measured worst case in a separate commit (C4).
- Both sides use `Omfid = 0.31`, the cb spectrum (`cb: Yes` / `field="cb"`), and the background growth rate f (spec §4.4–4.5).

**Commit gate (two tiers — reconciles "run `--fast` before commit" with "no heavy JAX on the login node")**
- *Local gate, every commit:* run the task's own test file(s) on the login node with the CPU flags above. Tests in Parts 1a/1b are designed to finish in < 2 min on CPU (`compute_ept` on the 256-point grid is ~20 s; no perturbation solves).
- *Cluster gate, at the checkpoints marked in tasks (end of A5, P1, B2, B3, B5, B7, every C task):* submit `slurm/ptval-fast-suite.sbatch` (created in B1) and wait for `PASS` in its log before the next commit on that track. Tests marked `slow` (every test with a `perturbations_solve`) are deselected by `--fast`; each task that adds one also runs it in full mode on the node before its commit (P1: `ptval-p1-deltacb.sbatch`; Track B: `ptval-track-b-full.sbatch`; Track C: `ptval-track-c.sbatch` with `PTVAL_PYTEST_ARGS`, or the campaign job).

---

## Oracle findings every task must respect

One line each; the cited ref § has the code and line numbers.

1. AP ratios: `hratio = E(z)/E_fid(z)`, `Dratio = D_A(z)·H0·(1+z) / Dfid(z)` in H0-units, with `Dfid` a 2000-point trapezoid whose radiation term is frozen at the target z — mirror the quirk, do not "fix" it (ref §1, §14.1).
2. Under AP, CLASS-PT compares only inside the Nside=10 window: `k_h[10] ≤ k ≤ k_h[-11]`; the campaign window is `k_h[10] ≤ k ≤ 0.3` (ref §2).
3. Remap is a natural cubic spline in **linear** k over the 256-point grid (`AP_INTERP_FAST`, ref §3); outside the grid CLASS-PT continues the end interval's cubic (`AP_BSEARCH_SETUP` clamps the interval, not the weights, `2383-2394`). clax's `CubicSpline.evaluate` **clamps** to constant extrapolation, so B5 evaluates its own `_channels_at` on `_compute_natural_spline_coeffs`; at a knot both give `a=1, b=0` (`searchsorted(side="right")-1`), so α=1 is an exact no-op.
4. Legendre projection weights use the **fiducial** μ; every kernel μ-power uses `mutrue` (ref §4, §5).
5. `V = hratio/Dratio²` multiplies every in-loop term and nothing outside the loop (ref §4).
6. Tree power is folded into the `2_dd`, `4_vd`, `4_dd` rows (`_ap_ir`); `0_*`, `2_vv`, `2_vd`, `4_vv` carry loop only (ref §4).
7. Counterterms `Pctr0/2/4` and `IFG2_{0b1,0,2}` are in-loop at `ktrue` with the anisotropic damping `Exp(μ)`; clax's analytic forms match only at α=1 without IR differences (ref §4, §5, §14.7).
8. In-loop bias block re-accumulates `Id2d2`, `Id2G2`, `IG2G2` monopoles and *generates* `P_4_b1b2`, `P_4_b1bG2` from the ℓ=0,2 inputs; `p_lo` IR ratio applies to `IFG2` only (ref §5).
9. `pm[1] = −P_Id2d2 ≤ 0`; `P_Id2d2 = |P(k) − P(k0) + 1e-6|` with `P = 2k³f22_real` — the same array as clax's `Pk_Id2d2`; classy `pk_gg_l0` adds `0.25 b2² (Pd2d2_0 − P_Id2d2)` with `Pd2d2_0 = simpson(P_lin² kh³, ln kh)/π²`, `pk_gg_real` adds `−0.25 b2² P_Id2d2` (ref §9, §12). **clax Bug #2**: both accessors add `+0.25 b2² Pk_Id2d2` (wrong sign) and `pk_gg_l0` lacks `Pd2d2_0`.
10. classy `pk_gg_l2` = `pm18+pm24 + b1(pm19+pm25) + b1²·pm26 + ...`; `pk_gg_l4` = `pm20+pm27 + b1·pm28 + b1²·pm29 + ...` (ref §11). **clax Bug #3**: `pk_gg_l2` omits `b1²·Pk_2_dd`, `pk_gg_l4` omits the b1 weights.
11. **clax Bug #1**: `ept.py:1088-1092` rebinds `nu1, nu2` to the matter basis; the RSD bias kernels from line 1506 must use the bias basis `nu1 = -0.5*eta_i`, `nu2 = -0.5*eta_l` (ref §15).
12. `Pk_4_vd1` disagrees with `pm[28]·h³ − Pk_4_vd` by a median ratio ≈ 0.83 at the legacy fiducial — unexplained, investigated in B3 before the loop changes (ref §15).
13. The legacy generator (`scripts/generate_classpt_reference.py:149-152`) calls `pk_gg_l0` with 7 positional args; the current 9-arg signature (`b1,b2,bG2,bGamma3,cs0,Pshot_nbar,a0_nbar,a2_nbar,b4`) would raise `TypeError`, so the legacy npz came from an older classy — A3's binding test decides what its `pk_gg_*` contain (ref §11 signature note).
14. `initialize_output(k, z, k_size)` stores `self.kh = k` — the *1/Mpc* array the caller passed — and uses it in `Pd2d2_0` and the `b4·kh²` terms (ref §12); `classy_kh_units.patch` sets `self.kh = k/h`. The generator stores which convention was active (`kh_convention` key).
15. `T_ncdm` in CLASS is in units of T_cmb: pass `T_ncdm = 0.71611` (spec §5.3). `cb: Yes` with `N_ncdm = 0` is undefined behaviour in CLASS-PT — the generator refuses it.
16. clax loads `/home/n2minh/CLASS-PT/pt_matrices/gauss_tab.dat` for its 40 GL nodes and silently falls back to 10-point `leggauss` if the file is missing (`ept.py:72-77`). Every sbatch script and `tests/test_ept_ap.py` assert `len(clax.ept._GAUSS_NODES) == 40` (ref §7).
17. CLASS-PT `Makefile:50` hard-codes `OPENBLAS = /share/software/user/open/openblas/0.3.28/lib/libopenblas.a` (absent here) and `python/setup.py` reads `OPENBLAS_PATH` from the environment; both must be overridden (A2).
18. **clax Bug #4 — real-space tree is not IR-resummed.** CLASS-PT `pm[14]` is `Ptree = Pnw + Pw·exp(−Σ_BAO k²)·(1 + Σ_BAO k²)` (`nonlinear_pt.c:2999`, splined to the output grid at `3580/3599`; the no-IR path `3294` uses `Ptree = Pbin`). clax sets `Pk_tree = pk_lin_h` (raw linear) on all three IR branches (`ept.py:1774, 1787, 1790`). Affects `pk_mm_real / pk_gg_real / pk_gm_real` and the `Pd2d2_0` integrand (`Plin_hMpc3 = pm[14]·h³`, ref §12). B4 fixes it; `Pk_ctr = −k² pk_resummed` (`ept.py:1846`) already matches `P_CTR = k²·Pbin` (`nonlinear_pt.c:3540`) and needs no change.
19. The local `/home/n2minh/CLASS-PT/source/nonlinear_pt.c` (commit `09d5531a`, 6069 lines) is a **refactored** CLASS-PT: the line numbers cited in clax's `ept.py` comments (`12871`, `12927`, …, `13339`) refer to the original 13k-line file and do not exist here. Verified local anchors: AP ratios `1245-1296`; `FILL_M22_BIAS` macro `2623-2640` (uses `nu1 = -0.5*etam2[_ib_]`, `nu2 = -0.5*etam2[_lb_]` — the bias basis, cf. Bug #1); `TIDAL_P22` `2582-2591`; `Ptree` `2999`; GL RSD loop `4386-4562`; RSD bias kernels `M22_0_b1b2 … M22_4_bG2` at `5052-5092`; bias loop `5225-5366`. B4 rewrites clax's citations to these anchors.
20. **clax Bug #5 — `pk_gm_real` counterterm.** classy.pyx:4821 is `(2·cs·b1 + cs0)·pm[10]/h²`; clax `pk_gm_real` has `(cs·b1 + cs0)·Pk_ctr` — the factor 2 on `cs·b1` is missing (invisible at legacy `cs = 0`). B4 Step 6b fixes it.
21. classy `pk_gg_l4` (classy.pyx:4902-4915) never reads `pm[40]`, `pm[41]` (`P_4_b1b2`, `P_4_b1bG2`) although CLASS-PT generates them in-loop; clax mirrors the accessor (they stay as leaves, unused by `pk_gg_l4`). Not a bug to fix — a quirk to state in the validation report.
22. CLASS-PT's AP μ-loop runs only over the interior `index_j = Nside … Nmax − Nside` (`4386`), so `ktrue` beyond `k[-1]` never occurs in CLASS-PT's own comparison window; clax evaluates all 256 points and reaches the end-cubic extrapolation at the last ~10 points for `hratio > 1` — irrelevant on the campaign window (finding 2), but it is why B5's transcription test covers extrapolated nodes explicitly.

---

## File structure

```
scripts/validation_cosmologies.py        A1  pure-Python case table, CLASS-PT param mapping, paths (both envs)
scripts/classpt_assembly.py              A3  NumPy twin of classy accessors (pd2d2_0, assemble_from_pm)
scripts/generate_classpt_reference.py    A3  rewritten CLI generator (classpt env)
scripts/write_classpt_manifest.py        A5  renders reference_data/classpt/MANIFEST.md from the npz files
scripts/setup_classpt_env.sh             A2  idempotent env + build recipe
scripts/classpt_patches/classy_ap_ratios.patch, classy_kh_units.patch   A2
scripts/freeze_ept_alpha1_baseline.py    B1  writes reference_data/ept_alpha1_baseline.npz
scripts/summarize_ept_validation.py      C3  campaign log → markdown tables
docs/classpt-build-notes.md              A2
docs/validation/2026-09-clax-pt-multipoles.md   C3
slurm/classpt-refgen.sbatch              A5  CPU job, env classpt
slurm/ptval-fast-suite.sbatch            B1  V100 job: pytest tests/ --fast -x -q
slurm/ptval-track-b-full.sbatch          B2  V100 job: Track B's own test files in full mode
slurm/ptval-p1-deltacb.sbatch            P1  V100 job: TestDeltaCb in full mode
slurm/ptval-track-c.sbatch               C0  V100 job running $PTVAL_PYTEST_ARGS (Track C smokes)
slurm/ept-multicosmo-e2e.sbatch          C3  V100 campaign job
clax/ap.py                               B2  ap_ratios(bg, z, omfid) -> (hratio, Dratio)
clax/ept.py                              B3–B6, C0  loop refactor, bug fixes, AP remap, accessors; C0 ept_inputs_from_clax + compute_ept_from_clax(omfid, field)
clax/lensing.py                          C0  compute_ept_from_clax(..., field="m")
clax/perturbations.py                    P1  PerturbationResult.delta_cb
tests/conftest.py                        A1  re-export grids from validation_cosmologies; ept_case fixture
tests/test_validation_cosmologies.py     A1
tests/test_classpt_assembly.py           A3, A4
tests/test_ap.py                         B2
tests/test_ept_ap.py                     B1, B3, B4, B5, B6  (Bug #1–#5 tests live here too)
tests/test_ept_assembly.py               B7 (extend, never clobber); C1 imports its helpers from ept_campaign_utils
tests/test_perturbations.py              P1 (append class TestDeltaCb)
tests/test_ept_from_clax.py              C0
tests/ept_campaign_utils.py              C1  SPECTRA, window, THRESHOLDS, metric, JSONL log, B7 helpers
tests/test_ept_multicosmo.py             C1
tests/test_ept_e2e_multicosmo.py         C2
tests/test_summarize_ept_validation.py   C3
reference_data/classpt/<case>/z{z:.3f}_{ap_omfid{X}|noap}_{cb|m}[_biasnz][_<tag>].npz   A3/A5
reference_data/classpt/MANIFEST.md       A5
reference_data/ept_alpha1_baseline.npz   B1 (refrozen by B3, B4 with a recorded reason)
```

---

## Task DAG

```
Track A (env classpt, CPU)        Track B (env clax)                    Track P
A1 cosmology table ──┐            B1 baseline + guards                  P1 delta_cb
A2 env + build       │            B2 ap_ratios (needs nothing)
A3 generator+twin ◄──┘ (A1,A2)    B3 α=1 loop refactor (B1)
A4 provenance gate (A3)           B4 bug fixes #1–#5 (B3)
A5 refgen job + MANIFEST (A4)     B5 in-loop AP remap (B4, B2)
                                  B6 AP gradients + smoke (B5)
                                  B7 assembly tests vs legacy (B4, A3)
Part 2 — Track C: C0 (B5, B6, P1) → C1 (A5, C0) → C2 (C1) → C3 (C2) → C4 (C3)
```

A1, A2, B1, B2, P1 start immediately and in parallel. A3 waits for A1+A2. B3 waits for B1. B7 waits for B4 and A3 (needs `scripts/classpt_assembly.py`). Track C starts when A5, B6 and P1 are on the branch.

Parallel implementers touch disjoint files; the only shared files are `clax/ept.py` (Track B only, strictly sequential B3→B4→B5→B6) and `tests/conftest.py` (A1 only).

---

## Run recipes

### Login-node probe (bounded)

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 \
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" \
  PYTHONPATH=/home/n2minh/clax-ptval \
  /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_ap.py -x -q
```

### Before the first sbatch of the campaign

```bash
mkdir -p /lustre/work/n2minh/std/clax/ptval
```

### sbatch template — `classpt` env, CPU only (Track A jobs)

```bash
#!/bin/bash -l
#SBATCH --job-name=classpt-refgen
#SBATCH --output=/lustre/work/n2minh/std/clax/ptval/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ptval/%x.err.%j
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
set -euo pipefail
eval "$(micromamba shell hook --shell bash)"
micromamba activate classpt
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
REPO=${CLAX_REPO:-/home/n2minh/clax-ptval}
cd "$REPO"
python -c "import classy, numpy; print('classy', classy.__file__, 'numpy', numpy.__version__)"
# ... job body ...
```

### sbatch template — `clax` env, V100 (Track B/C jobs)

Header copied from `slurm/bench-v100-igpu.sbatch`; only job-name, output dir, time and body change.

```bash
#!/bin/bash -l
#SBATCH --job-name=ptval-fast-suite
#SBATCH --output=/lustre/work/n2minh/std/clax/ptval/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ptval/%x.err.%j
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
set -euo pipefail
eval "$(micromamba shell hook --shell bash)"
micromamba activate clax
for d in "${CONDA_PREFIX}"/lib/python3.*/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
REPO=${CLAX_REPO:-/home/n2minh/clax-ptval}
cd "$REPO"
export PYTHONPATH="$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('devices', jax.devices())"
python -c "import clax.ept as e; assert len(e._GAUSS_NODES) == 40, 'gauss_tab.dat missing: 10-pt fallback active'; print('GL nodes', len(e._GAUSS_NODES))"
# ... job body ...
```

(The `LD_LIBRARY_PATH` loop must match the one in `slurm/bench-v100-igpu.sbatch` verbatim — copy it from there if the two differ.)

### Submitting and waiting

```bash
cd /home/n2minh/clax-ptval && sbatch slurm/<job>.sbatch     # prints "Submitted batch job <id>"
squeue -u n2minh --name=<job-name> -h                           # empty when finished
tail -n 30 /lustre/work/n2minh/std/clax/ptval/<job-name>.out.<id>
```

Poll with `squeue` every few minutes; never spin on `tail -f`.

### Commit recipe

Write the message to `/tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-<task>.txt` with the Write tool:

```
<type>(<scope>): <subject>

<body: what changed, what was measured, which spec § it satisfies>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

then `cd /home/n2minh/clax-ptval && git add <files> && git commit -F <that path>`.

---

## Reviewer briefs (used by the dispatcher after every task)

Two reviewers per task, each dispatched with the task text, the diff, and this brief.

**Spec-compliance reviewer** — verify, do not trust: (1) every formula in the diff against the cited `nonlinear_pt.c`/`classy.pyx` line in `/home/n2minh/CLASS-PT` (open the C/pyx file; a mismatch in a sign, an index, a μ-power, or which-μ (fiducial vs true) is a blocking finding); (2) every test the implementer claims to have run — re-run it yourself with the login-node recipe and paste the last 5 lines; (3) every threshold — is it the spec's number, and is the measured value recorded in the test log or CHANGELOG; (4) every "unchanged" claim — diff the baseline npz yourself. Report `BLOCK` / `PASS` with the evidence lines.

**Code-quality reviewer** — verify the diff reads like the surrounding code (naming, comment density, `nonlinear_pt.c:` citations), that no Python `if` depends on a traced value, that pure functions stay pure, that no test was weakened or skipped, and that the CHANGELOG entry (when the task has one) uses the `Mon D, YYYY` heading. Report `BLOCK` / `PASS` with file:line pointers.

Both reviewers are told: "The implementer's claims are hypotheses. Your job is to try to falsify them."
