# tests/test_ps1loop_coefficients.py
"""Code-accuracy gate on coefficient spectra (spec §5.1): ps_1loop vs clax-pt on CLASS-PT's
own inputs, for every monomial whose translation is exact (spec §4.1). Exact translation does not mean an
identical model: the rows of b2, b1*b2, b2^2, bG2^2, b2*bG2, the six F_G2 kept rows and 1, b1, b1^2 also carry
the spec §4.2 items 5-6 model difference (ps's IR scheme on the kept bias rows; the matter loops' O(P_w^2)
term), which this gate does not subtract. So these passes are partly physics-limited: b2:l2 uses 0.878% of its
1% gate (omega_cdm_low z = 0.38), b1*b2:l2 0.854% (omega_cdm_low z = 0), b2:l4 1.316% of 2%
(massive_nu_015_omega_cdm_low z = 0; job 25493), and at lcdm_fiducial z = 0.38 the scheme term explains b2:l2
almost entirely (0.612% -> 0.026%, scratch job 24970; not measured at omega_cdm_low). 14 cosmologies x
z in (0, 0.38, 0.8) in full mode; vc.FAST_CASES x vc.FAST_Z under --fast. Multi-cosmology
rule: satisfied by construction. The thirteen br.CLOSURE_LINES (six F_G2 C1 lines, the six
isotropic-loop l2/l4 lines of b2*b2, bG2*bG2, b2*bG2, and b1*b2:l4 = CLASS-PT's computed-but-unused
pm[40]) are logged but not gated here; they are asserted by the
closed-form closure in tests/test_ps1loop_model_difference.py. Failures carry the CLASS-PT tie-breaker verdict (spec §6)."""
from __future__ import annotations

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


def test_exact_coefficients(case_z):
    case, z = case_z
    inp = br.shared_inputs(cu.require_reference(case, z))
    res, kw = br.coefficient_triplet(inp, br.EXACT_MONOMIALS)
    errs, lines = br.gate(res, kw, exclude=br.CLOSURE_LINES)
    br.log_record(layer="coef", case=case, z=z, preset="exact", errors=errs)
    print(f"{case} z={z:.2f} [coef] {br.worst(errs)}")
    assert not lines, f"{case} z={z:.2f}:\n" + "\n".join(lines)
