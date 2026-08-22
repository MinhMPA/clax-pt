# tests/test_analyze_benchmark.py
import json
import sys
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "benchmark"
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_benchmark import find_result_files, MetricStatus

def test_find_result_files_all_present(tmp_path):
    for name in ["solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy", "planck_cl"]:
        (tmp_path / f"ravg1002-a100-{name}.txt").write_text("dummy")
    found, missing = find_result_files(tmp_path)
    assert set(found.keys()) == {"solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy", "planck_cl"}
    assert missing == []

def test_find_result_files_partial(tmp_path):
    for name in ["solvers", "gradients", "fit_cl"]:
        (tmp_path / f"ravg1002-a100-{name}.txt").write_text("dummy")
    found, missing = find_result_files(tmp_path)
    assert set(found.keys()) == {"solvers", "gradients", "fit_cl"}
    assert set(missing) == {"ept", "clpp", "accuracy", "planck_cl"}

def test_metric_status_values():
    assert MetricStatus.PASS == "PASS"
    assert MetricStatus.FAIL == "FAIL"
    assert MetricStatus.WARN == "WARN"
    assert MetricStatus.MISSING == "MISSING"


from scripts.analyze_benchmark import (
    parse_solvers, parse_gradients, parse_speed,
    parse_ept, parse_clpp, parse_accuracy,
)

def _fixture(name):
    return (FIXTURES / name).read_text()

def test_parse_solvers():
    m = parse_solvers(_fixture("solvers.txt"))
    assert abs(m["kvaerno5_median_s"] - 13.09) < 0.01
    assert abs(m["rodas5_median_s"] - 6.21) < 0.01
    assert abs(m["rodas5_speedup"] - 2.11) < 0.01
    assert abs(m["rosenbrock_batched_median_s"] - 6.19) < 0.01
    assert abs(m["rodas5_vs_kv_max_pct"] - 0.0872) < 0.001
    assert m["platform"] == "gpu"

def test_parse_gradients():
    m = parse_gradients(_fixture("gradients.txt"))
    assert abs(m["fwd_cached_s"] - 13.09) < 0.01
    assert abs(m["ad_median_s"] - 28.34) < 0.01
    assert abs(m["bwd_fwd_ratio"] - 2.2) < 0.01
    assert abs(m["ad_fd_speedup"] - 4.7) < 0.01
    assert abs(m["ad_fd_agreement"]["h"] - 0.0405) < 0.001
    assert abs(m["ad_fd_agreement"]["omega_b"] - 0.0087) < 0.001

def test_parse_speed_fit_cl():
    m = parse_speed(_fixture("fit_cl.txt"))
    assert m["preset"] == "fit_cl"
    assert abs(m["total_cached_s"] - 33.0) < 0.1
    assert abs(m["pt_cached_s"] - 30.2) < 0.1
    assert abs(m["cl_errs"][20]["tt"] - (-0.123)) < 0.001
    assert abs(m["cl_errs"][1000]["ee"] - 0.001) < 0.001

def test_parse_ept():
    m = parse_ept(_fixture("ept.txt"))
    assert abs(m["ept_fwd_cached_s"] - 0.412) < 0.001
    assert abs(m["bwd_fwd_ratio"] - 3.0) < 0.01   # fixture was fixed: 3.0x not 0.37x
    assert abs(m["pmm_max_pct"] - 0.312) < 0.001
    assert abs(m["pgg_max_pct"] - 0.312) < 0.001
    assert abs(m["grad_cached_s"] - 1.234) < 0.001

def test_parse_clpp():
    m = parse_clpp(_fixture("clpp.txt"))
    assert abs(m["clpp_none_s"] - 0.087) < 0.001
    assert abs(m["clpp_halofit_s"] - 0.234) < 0.001
    assert abs(m["clpp_ept_s"] - 0.891) < 0.001
    assert m["clpp_linear_max_abspct"] < 0.2
    assert abs(m["bb_ratios"][2] - 0.9982) < 0.001
    assert abs(m["bb_ratios"][100] - 0.9996) < 0.001
    assert abs(m["bb_ratios"][200] - 0.9990) < 0.001

def test_parse_accuracy_pass():
    m = parse_accuracy(_fixture("accuracy.txt"))
    assert m["all_pass"] is True
    assert abs(m["spectra"]["pk_mm_real"]["max_pct"] - 0.31) < 0.01
    assert m["spectra"]["pk_mm_real"]["pass"] is True
    assert len(m["spectra"]) == 9

def test_parse_accuracy_fail():
    m = parse_accuracy(_fixture("accuracy_fail.txt"))
    assert m["all_pass"] is False
    assert m["spectra"]["pk_gg_l4"]["pass"] is False
    assert abs(m["spectra"]["pk_gg_l4"]["max_pct"] - 2.15) < 0.01


from scripts.analyze_benchmark import check_thresholds

def test_check_thresholds_all_pass():
    metrics = {
        "solvers": parse_solvers(_fixture("solvers.txt")),
        "gradients": parse_gradients(_fixture("gradients.txt")),
        "fit_cl": parse_speed(_fixture("fit_cl.txt")),
        "ept": parse_ept(_fixture("ept.txt")),
        "clpp": parse_clpp(_fixture("clpp.txt")),
        "accuracy": parse_accuracy(_fixture("accuracy.txt")),
        "planck_cl": parse_speed(_fixture("planck_cl.txt")),
    }
    checked = check_thresholds(metrics)
    fails = [k for k, v in checked.items() if v["status"] == "FAIL"]
    assert fails == [], f"Unexpected FAILs: {fails}"

def test_check_thresholds_fit_cl_total_fail():
    metrics = {
        "fit_cl": {"total_cached_s": 55.0, "pt_cached_s": 28.0,
                   "cl_errs": {}, "preset": "fit_cl"},
    }
    checked = check_thresholds(metrics)
    assert checked["fit_cl.total_cached_s"]["status"] == "FAIL"

def test_check_thresholds_fit_cl_total_warn():
    metrics = {
        "fit_cl": {"total_cached_s": 48.0, "pt_cached_s": 28.0,
                   "cl_errs": {}, "preset": "fit_cl"},
    }
    checked = check_thresholds(metrics)
    assert checked["fit_cl.total_cached_s"]["status"] == "WARN"

def test_check_thresholds_accuracy_fail():
    metrics = {"accuracy": parse_accuracy(_fixture("accuracy_fail.txt"))}
    checked = check_thresholds(metrics)
    assert checked["accuracy.all_pass"]["status"] == "FAIL"

def test_check_thresholds_missing_file():
    checked = check_thresholds({})
    assert checked["planck_cl.total_cached_s"]["status"] == "MISSING"


from scripts.analyze_benchmark import write_report, build_results_dict

def _parse_all(found):
    """Helper: parse all found files using the right parser."""
    parsed = {}
    for name, path in found.items():
        text = path.read_text()
        if name == "solvers":
            parsed[name] = parse_solvers(text)
        elif name == "gradients":
            parsed[name] = parse_gradients(text)
        elif name in ("fit_cl", "planck_cl"):
            parsed[name] = parse_speed(text)
        elif name == "ept":
            parsed[name] = parse_ept(text)
        elif name == "clpp":
            parsed[name] = parse_clpp(text)
        elif name == "accuracy":
            parsed[name] = parse_accuracy(text)
    return parsed

def test_write_report_creates_files(tmp_path):
    for name in ["solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy"]:
        (tmp_path / f"ravg1002-a100-{name}.txt").write_text(
            _fixture(f"{name}.txt"))
    found, missing = find_result_files(tmp_path)
    parsed = _parse_all(found)
    checked = check_thresholds(parsed)
    results = build_results_dict(tmp_path, found, missing, parsed, checked)
    write_report(results, tmp_path)
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "REPORT.md").exists()

def test_results_json_structure(tmp_path):
    (tmp_path / "ravg1002-a100-solvers.txt").write_text(_fixture("solvers.txt"))
    found, missing = find_result_files(tmp_path)
    parsed = _parse_all(found)
    checked = check_thresholds(parsed)
    results = build_results_dict(tmp_path, found, missing, parsed, checked)
    assert "date" in results
    assert "files_found" in results
    assert "files_missing" in results
    assert "metrics" in results
    assert "summary" in results
    assert "n_pass" in results["summary"]
