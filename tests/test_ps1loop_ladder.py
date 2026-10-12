# tests/test_ps1loop_ladder.py
"""Smoke tests of the convergence ladder (spec §7). Exempt from the multi-cosmology rule:
a diagnostic that gates nothing; these check its plumbing at one point each.

  * The wiggle-cut patch takes effect AND is fully undone (jit caches; Review Focus 1). ps_1loop
    0.5.0 already ships CLASS-PT's DST cuts 120-240, so the patched rung is `dewiggle_legacy`
    (0.4.0's 140-210; controller amendment A, Oct 12, 2026).
  * The remainder fields at (lcdm_fiducial, z = 0.38) are Task 7's own numbers at the default rung
    (controller amendment B): the primal ps - clax - D and the stretch-tangent direct / closed
    differences equal what test_cross_code_tangents printed for lcdm_fiducial in job 24864 (to the
    printed 0.001%), so the ladder measures the remainder the red test reports and not a look-alike.
    The primal remainder over the derivative's scale equals Task 7's R0_over_scale (scratch job 24845),
    so the amplification is a like-for-like ratio (fix round 1). The AP fields (amendment D) are finite.
  * The PFS-slice AP floor at the default setting reproduces Task 8's s = 0.001 measurement (job 24871:
    Delta chi^2 of ps - clax over the 7 slices 0.1283), so the floor at other settings is the same
    construction."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")

from tests import ept_campaign_utils as cu
from tests import ps1loop_bridge as br

# Task 7, job 24864, lcdm_fiducial z = 0.38 (% of max|clax|; l = 0, 2, 4)
TASK7_PRIMAL_CLOSED = (0.014, 0.024, 0.048)
# Task 7 scratch job 24845 (closure_diag.json, stretch row R0_over_scale): max|ps - clax - D| / max|jvp_clax|
TASK7_R0_OVER_DSCALE = (2.506255e-4, 4.301191e-4, 1.059289e-3)
TASK8_FLOOR_DCHI2 = 0.1283   # Task 8 scratch job 24871: lcdm_fiducial, offsets x 0.001, 7 slices, cfog 500
TASK7_STRETCH_DIRECT = (0.304, 0.534, 1.979)
TASK7_STRETCH_CLOSED = (0.283, 0.458, 1.493)
PRINTED = 5e-4   # half of the printed 0.001% digit


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


def test_ladder_remainders_reproduce_task7(tmp_path, monkeypatch):
    if cu.load_reference("lcdm_fiducial", 0.38) is None:
        pytest.skip("reference missing: lcdm_fiducial z=0.38 -- run slurm/classpt-refgen.sbatch")
    monkeypatch.setattr(br, "LOG_DIR", tmp_path)
    monkeypatch.setattr(br, "ERROR_LOG", tmp_path / "errors.jsonl")
    from scripts import ps1loop_convergence_ladder as lad
    (r,) = lad.run(points=(("lcdm_fiducial", 0.38),), rungs=("default",))
    assert r["rung"] == "default" and not np.isfinite(r["delta_chi2"])   # z = 0.38 is not a PFS slice
    for i, l in enumerate(("l0", "l2", "l4")):
        assert 100 * r["primal_remainder"][l]["err"] == pytest.approx(TASK7_PRIMAL_CLOSED[i], abs=PRINTED)
        assert 100 * r["stretch_direct"][l] == pytest.approx(TASK7_STRETCH_DIRECT[i], abs=PRINTED)
        assert 100 * r["stretch_closed"][l] == pytest.approx(TASK7_STRETCH_CLOSED[i], abs=PRINTED)
        assert r["stretch_ratio"][l] == pytest.approx(r["stretch_closed"][l] / r["stretch_direct"][l], rel=1e-12)
        assert r["primal_remainder_over_dscale"][l] == pytest.approx(TASK7_R0_OVER_DSCALE[i], rel=1e-5)
        assert r["stretch_amplification"][l] == pytest.approx(
            r["stretch_closed"][l] / r["primal_remainder_over_dscale"][l], rel=1e-12)
        for field in ("ap_zero_offset", "ap_unit_alpha", "ap_zero_offset_clax"):
            assert np.isfinite(r[field][l]["err"]) and r[field][l]["err"] > 0.0, field
    assert len((tmp_path / "errors.jsonl").read_text().splitlines()) == 1


def test_ladder_pfs_floor_reproduces_task8(tmp_path, monkeypatch):
    if any(cu.load_reference("lcdm_fiducial", sl_z) is None for sl_z in (0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2)):
        pytest.skip("reference missing: lcdm_fiducial PFS slices -- run slurm/classpt-refgen-pfs.sbatch")
    monkeypatch.setattr(br, "LOG_DIR", tmp_path)
    monkeypatch.setattr(br, "ERROR_LOG", tmp_path / "errors.jsonl")
    from scripts import ps1loop_convergence_ladder as lad
    (r,) = lad.pfs_ap_floor(rungs=("default",))
    assert r["vs_clax"] == pytest.approx(TASK8_FLOOR_DCHI2, abs=5e-5)
    assert len(r["slices"]) == 7
    assert r["ps_jump_max_sigma"] > r["clax_jump_max_sigma"] > 0.0   # Task 8: 0.150 sigma vs 0.002 sigma
    assert len((tmp_path / "errors.jsonl").read_text().splitlines()) == 1
