# tests/test_ps1loop_real_space.py
"""Real-space P_mm, P_gg, P_gm with IR resummation OFF in both codes (spec §5.4): ps_1loop's
real-space path has no IR resummation, so only IR-off is a numerics comparison. 42 files in
full mode, vc.FAST_CASES x vc.FAST_Z under --fast; tie-breaker from the IR-off CLASS-PT files
where they exist. Plus: clax-pt's IR-off path on its own vs CLASS-PT IR-off (3 cosmologies).
Multi-cosmology rule: by construction.

AP (controller ruling, Oct 7, 2026): real space has no AP and ps_1loop applies none there, so
clax-pt's real-space spectra are evaluated with hratio = Dratio = 1 (br.real_coefficient_triplet).
CLASS-PT, and clax-pt mirroring it, store the isotropic loops Id2d2, Id2G2, IG2G2 (pk_mult rows
1, 4, 5) as the AP-folded l = 0 moment of the mu-loop and reuse them in pk_gg_real, so an AP file's
real-space b2*b2, b2*bG2, bG2*bG2 rows carry AP; that is a CLASS-PT convention, recorded as a
finding and never gated here. The CLASS-PT tie-breaker uses only the IR-off files, which are no-AP
(asserted in the bridge).

Not gated against ps_1loop here: the P_gg and P_gm counterterm maps (both codes run with their P_gg / P_gm
counterterms off, br._ps_real_jit and br.clax_real_batch, because the two parameterise them differently) and
ps's scale-dependent stochastic term a0 (k/k_nl)^2 (a0 = 0 in REAL_MONOMIALS). Only the P_mm counterterm
shape (c0) is compared."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")

from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu
from tests import ps1loop_bridge as br


def pytest_generate_tests(metafunc):
    if "case_z" in metafunc.fixturenames:
        if metafunc.config.getoption("--fast", default=False):
            grid = [(c, vc.FAST_Z) for c in vc.FAST_CASES]
        else:
            grid = [(c, z) for c in vc.distinct_cases() for z in vc.Z_LIST]
        metafunc.parametrize("case_z", grid, ids=[f"{c}-z{z:.2f}" for c, z in grid])


def test_real_space_coefficients_ir_off(case_z):
    case, z = case_z
    inp = br.shared_inputs(cu.require_reference(case, z))
    irno = cu.load_reference(case, z, ap=False, tag=vc.IRNO_TAG)
    res, kw = br.real_coefficient_triplet(inp, irno_inp=br.shared_inputs(irno) if irno is not None else None)
    errs, lines = br.gate(res, kw, channels=br.REAL_CHANNELS, thresholds=br.REAL_THRESHOLDS)
    br.log_record(layer="real", case=case, z=z, preset="ir-off", errors=errs)
    print(f"{case} z={z:.2f} [real, IR off] {br.worst(errs)}")
    assert not lines, f"{case} z={z:.2f}:\n" + "\n".join(lines)


@pytest.mark.parametrize("case", vc.IRNO_CASES)
def test_clax_ir_off_matches_classpt_ir_off(case):
    ref = cu.require_reference(case, 0.38, ap=False, tag=vc.IRNO_TAG)
    e = br.clax_ept(br.shared_inputs(ref), ir=False)
    nine = {n: np.asarray(a) for n, a in cu.clax_nine(e, json.loads(str(ref["bias_json"]))).items()}
    errs = cu.compare_spectra({n: nine[n] for n in br.REAL_CHANNELS}, ref, ref["k_h"])
    br.log_record(layer="real", case=case, z=0.38, preset="clax-vs-classpt-irno", errors=errs)
    bad = cu.failures(errs, cu.THRESHOLDS)
    print(f"{case} [clax IR off vs CLASS-PT IR off] {br.worst(errs)}")
    assert not bad, f"ERROR {case}: " + "; ".join(bad)
