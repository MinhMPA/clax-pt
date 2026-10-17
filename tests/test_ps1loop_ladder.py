# tests/test_ps1loop_ladder.py
"""Smoke tests of the convergence ladder (spec §7). Exempt from the multi-cosmology rule:
a diagnostic that gates nothing; these check its plumbing at one point each.

  * The wiggle-cut patch takes effect AND is fully undone (jit caches; Review Focus 1). ps_1loop
    0.5.0 already ships CLASS-PT's DST cuts 120-240, so the patched rung is `dewiggle_legacy`
    (0.4.0's 140-210; controller amendment A, Oct 12, 2026).
  * The remainder fields at (lcdm_fiducial, z = 0.38) are the derivative gate's own numbers: the ladder's
    nfft512 rung is the gate setting (ps nfft = test_ps1loop_gradients.PS_NFFT_GATE, Simpson num 256), and
    the primal remainder ps - clax - D, the stretch-tangent direct / closed / ratio and the primal remainder
    over the derivative's scale equal what tests/test_ps1loop_gradients.py computes at the same point, to
    rel 1e-10 (same code path: br.tangent_setup, br.primal_closure, br.tangent_closure, the same lru-cached
    ps model, the same D). So the ladder measures the remainder the gate asserts, not a look-alike, with no
    golden numbers to go stale (Task 11; they replace the values pinned from jobs 24864 / 24845). The AP fields
    (amendment D) are finite.
  * The PFS-slice AP floor at the default setting reproduces Task 8's s = 0.001 measurement (job 24871:
    Delta chi^2 of ps - clax over the 7 slices 0.1283), so the floor at other settings is the same
    construction. That one value stays pinned; regenerate it when clax or ps numerics change on purpose."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")

from tests import ept_campaign_utils as cu
from tests import ps1loop_bridge as br

TASK8_FLOOR_DCHI2 = 0.1283   # Task 8 scratch job 24871: lcdm_fiducial, offsets x 0.001, 7 slices, cfog 500
SAME_PATH_REL = 1e-10        # ladder rung vs gradient gate: one code path, so only round-off may differ


def test_ladder_patch_takes_effect_and_is_undone(tmp_path, monkeypatch):
    if cu.load_reference("lcdm_fiducial", 2.2) is None:
        pytest.skip("reference missing: lcdm_fiducial z=2.2 -- run slurm/classpt-refgen-pfs.sbatch")
    monkeypatch.setattr(br, "LOG_DIR", tmp_path)
    monkeypatch.setattr(br, "ERROR_LOG", tmp_path / "errors.jsonl")
    from scripts import ps1loop_convergence_ladder as lad
    recs = lad.run(points=(("lcdm_fiducial", 2.2),), rungs=("default", "dewiggle_legacy", "default"))
    assert [r["rung"] for r in recs] == ["default", "dewiggle_legacy", "default"]
    # the cuts in effect, read from ps_1loop's remove_wiggle signature inside each rung
    assert [r["settings"]["dst_cuts"] for r in recs] == [[120, 240], [140, 210], [120, 240]]
    assert recs[2]["delta_chi2"] == pytest.approx(recs[0]["delta_chi2"], rel=1e-12)
    assert abs(recs[1]["delta_chi2"] - recs[0]["delta_chi2"]) > 1e-9 * max(recs[0]["delta_chi2"], 1e-30)
    assert len((tmp_path / "errors.jsonl").read_text().splitlines()) == 3


def test_ladder_remainders_match_the_gradient_gate(tmp_path, monkeypatch):
    """The ladder's nfft512 rung at (lcdm_fiducial, z = 0.38) reports the derivative gate's numbers (module
    docstring): primal remainder, stretch direct / closed / ratio / amplification and primal remainder over
    the derivative's scale, each to SAME_PATH_REL of br.primal_closure / br.tangent_closure called through
    tests/test_ps1loop_gradients.py's own setup."""
    if cu.load_reference("lcdm_fiducial", 0.38) is None:
        pytest.skip("reference missing: lcdm_fiducial z=0.38 -- run slurm/classpt-refgen.sbatch")
    monkeypatch.setattr(br, "LOG_DIR", tmp_path)
    monkeypatch.setattr(br, "ERROR_LOG", tmp_path / "errors.jsonl")
    from scripts import ps1loop_convergence_ladder as lad
    from tests import test_ps1loop_gradients as g
    assert g.Z == 0.38
    (r,) = lad.run(points=(("lcdm_fiducial", g.Z),), rungs=("nfft512",))
    assert r["rung"] == "nfft512" and not np.isfinite(r["delta_chi2"])   # z = 0.38 is not a PFS slice
    assert r["settings"]["nfft"] == g.PS_NFFT_GATE and r["settings"]["num"] == 256   # the gate's ps setting

    inp, mask, theta = g._inputs("lcdm_fiducial")
    kw = inp["k_h"][mask]
    model, num = lad._model_and_num("nfft512", inp)
    assert model is g._model(inp) and num == 256   # the same lru-cached ps instance, the gate's Simpson num
    ps_fun, clax_fun, x0, tangents = g._setup("lcdm_fiducial")
    primal = br.primal_closure(ps_fun, clax_fun, inp, theta, kw, x0, model=g._model(inp))
    st = br.tangent_closure(ps_fun, clax_fun, inp, theta, kw, x0, tangents["stretch"], fd_step=g.FD_STEP,
                            model=g._model(inp))
    lines = []
    for l in ("l0", "l2", "l4"):
        over = primal[l]["resid_max"] / st[l]["scale"]
        pairs = {"primal_remainder": (r["primal_remainder"][l]["err"], primal[l]["closed"]),
                 "primal_remainder_over_dscale": (r["primal_remainder_over_dscale"][l], over),
                 "stretch_direct": (r["stretch_direct"][l], st[l]["direct"]),
                 "stretch_closed": (r["stretch_closed"][l], st[l]["closed"]),
                 "stretch_ratio": (r["stretch_ratio"][l], st[l]["ratio"]),
                 "stretch_amplification": (r["stretch_amplification"][l], st[l]["closed"] / over)}
        for name, (got, want) in pairs.items():
            if not got == pytest.approx(want, rel=SAME_PATH_REL):
                lines.append(f"ERROR ladder nfft512 {name} {l}: {got!r} vs gradient gate {want!r} "
                             f"(rel > {SAME_PATH_REL:.0e}; the two no longer share a code path)")
        for field in ("ap_zero_offset", "ap_unit_alpha", "ap_zero_offset_clax"):
            if not (np.isfinite(r[field][l]["err"]) and r[field][l]["err"] > 0.0):
                lines.append(f"ERROR ladder nfft512 {field} {l}: {r[field][l]['err']!r} not finite and positive")
    print("ladder nfft512 = gradient gate (lcdm_fiducial z=0.38): stretch direct/closed % "
          + " | ".join(f"{l} {100 * st[l]['direct']:.3f}/{100 * st[l]['closed']:.3f} r={st[l]['ratio']:.3f}"
                       for l in ("l0", "l2", "l4")))
    assert not lines, "\n".join(lines)
    assert len((tmp_path / "errors.jsonl").read_text().splitlines()) == 1


def test_ladder_pfs_floor_reproduces_task8(tmp_path, monkeypatch):
    if any(cu.load_reference("lcdm_fiducial", sl_z) is None for sl_z in (0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2)):
        pytest.skip("reference missing: lcdm_fiducial PFS slices -- run slurm/classpt-refgen-pfs.sbatch")
    monkeypatch.setattr(br, "LOG_DIR", tmp_path)
    monkeypatch.setattr(br, "ERROR_LOG", tmp_path / "errors.jsonl")
    from scripts import ps1loop_convergence_ladder as lad
    (r,) = lad.pfs_ap_floor(rungs=("default",))
    assert r["vs_clax"] == pytest.approx(TASK8_FLOOR_DCHI2, abs=5e-5), (
        f"ERROR PFS-slice AP floor {r['vs_clax']:.5f} != {TASK8_FLOOR_DCHI2} (Task 8 scratch job 24871, "
        "reproduced by the ladder in job 24919); regenerate the pin when clax or ps numerics change on purpose")
    assert len(r["slices"]) == 7
    assert r["ps_jump_max_sigma"] > r["clax_jump_max_sigma"] > 0.0   # Task 8: 0.150 sigma vs 0.002 sigma
    assert len((tmp_path / "errors.jsonl").read_text().splitlines()) == 1
