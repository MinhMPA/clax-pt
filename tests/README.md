 ## Summary

  The `tests/` suite was reorganized to give each file one clear ownership boundary.

  Main goals of the reorganization:
  - remove duplicated `P(k)` and gradient checks across unrelated files
  - keep public API smoke tests lightweight
  - separate forward-value contracts from backward-gradient contracts where cost differs materially
  - standardize test docstrings and naming so failures map cleanly to the owning layer
  - make full-suite execution safe on high-memory local Macs by running heavy files serially

  ## Current Ownership

  - `test_end_to_end.py`
    - Public API smoke only
    - Owns no physics-accuracy or gradient contracts

  - `test_background.py`
    - Background value and gradient contracts

  - `test_thermodynamics.py`
    - Thermodynamics forward-value contracts

  - `test_perturbations.py`
    - Perturbation-layer invariants and cheap forward checks only

  - `test_pk_accuracy.py`
    - Forward `P(k)` accuracy contract

  - `test_pk_gradients.py`
    - Regular-suite `P(k)` gradient contract

  - `test_harmonic.py`
    - Scalar unlensed `C_l` forward/API checks

  - `test_high_l.py`
    - High-`l` helper and consistency checks

  - `test_lensing.py`
    - Lensing forward behavior and lensed-spectrum checks

  - `test_tensor.py`
    - Tensor-mode forward checks with coarse reduced-precision tolerances

  - `test_nonlinear.py`
    - Nonlinear `P(k)` behavior and local differentiability checks

  - `test_multipoint.py`
    - Non-fiducial regression points

  ## What Changed

  ### Ownership cleanup
  - Removed `P(k)` value and gradient checks from files that did not own those contracts.
  - Reduced `test_end_to_end.py` to smoke-level coverage.
  - Reduced `test_perturbations.py` to perturbation-local forward checks.

  ### Docstring cleanup
  - Standardized module docstrings to:
    - Contract
    - Scope
    - Notes
  - Standardized class docstrings to short one-line ownership statements.
  - Standardized test docstrings to:
    - behavior under test
    - expected outcome or tolerance

  ### Diagnostics cleanup
  - Removed print-only diagnostics and pseudo-tests from the regular suite where they did not enforce a real contract.

  ## Current Accuracy Status

  ### Forward `P(k)`
  - Regular suite currently gates forward `P(k)` against CLASS in `test_pk_accuracy.py`
  - Current tested range: `0.001 <= k <= 0.3 Mpc^-1`
  - Current tolerance: `<1.5%` max relative error
  - Growth-ratio check `P(k,z=0.5)/P(k,z=0)` is gated at `<1%`

  ### `P(k)` gradients
  - Regular suite currently gates stable scalar gradients in `test_pk_gradients.py`
  - Current gated parameters:
    - `ln10A_s`
    - `n_s`
  - Current tolerance: `<5%` AD-vs-FD relative error
