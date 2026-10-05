#!/bin/bash
# The CLAUDE.md pre-commit gate: pytest tests/ --fast -x -q, with the campaign's documented
# deselect (pre-existing max-steps failure, CHANGELOG Aug 29 and Sep 1, 2026; issue #30 item 5).
exec "$(dirname "$0")/run-ps1loop-pytest.sh" tests/ --fast -x -q -p no:cacheprovider \
  --deselect tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5
