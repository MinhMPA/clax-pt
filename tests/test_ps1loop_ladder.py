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
    The AP zero-offset jump (amendment D) must be finite."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")

from tests import ept_campaign_utils as cu
from tests import ps1loop_bridge as br

# Task 7, job 24864, lcdm_fiducial z = 0.38 (% of max|clax|; l = 0, 2, 4)
TASK7_PRIMAL_CLOSED = (0.014, 0.024, 0.048)
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
        assert np.isfinite(r["ap_zero_offset"][l]["err"]) and r["ap_zero_offset"][l]["err"] > 0.0
        assert np.isfinite(r["ap_zero_offset_clax"][l]["err"])
    assert len((tmp_path / "errors.jsonl").read_text().splitlines()) == 1
