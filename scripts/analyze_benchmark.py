"""Stage 1: parse benchmark .txt outputs and check BENCHMARK.md §4 thresholds.

Usage:
    python scripts/analyze_benchmark.py [--date YYYY-MM-DD] [--dir /path/to/results]

Writes:
    <dir>/results.json   structured metrics + pass/fail
    <dir>/REPORT.md      human-readable tables

Exit codes:
    0  all present metrics PASS
    1  one or more metrics FAIL
    2  one or more expected files MISSING
"""
import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

EXPECTED_SCRIPTS = ["solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy", "planck_cl"]


class MetricStatus:
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    MISSING = "MISSING"


def find_result_files(result_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """Return (found: {script_name: path}, missing: [script_name])."""
    found = {}
    for path in result_dir.glob("*-a100-*.txt"):
        for name in EXPECTED_SCRIPTS:
            if path.name.endswith(f"-{name}.txt") and name not in found:
                found[name] = path
    missing = [s for s in EXPECTED_SCRIPTS if s not in found]
    return found, missing


def parse_solvers(text: str) -> dict:
    """Parse benchmark_solvers.py output."""
    m = {}
    plat = re.search(r"Platform: (\w+)", text)
    m["platform"] = plat.group(1) if plat else None

    rel = re.findall(r"vs kvaerno5: max=(\d+\.\d+)%, mean=(\d+\.\d+)%", text)
    rows = re.findall(r"^\s+(kvaerno5|rodas5|rosenbrock_batched)\s+(\d+\.\d+)\s+(\d+\.\d+)x",
                      text, re.MULTILINE)
    for row in rows:
        name, median, speedup = row
        m[f"{name}_median_s"] = float(median)
        m[f"{name}_speedup"] = float(speedup)

    if len(rel) >= 1:
        m["rodas5_vs_kv_max_pct"] = float(rel[0][0])
    if len(rel) >= 2:
        m["rosenbrock_vs_kv_max_pct"] = float(rel[1][0])
    return m


def parse_gradients(text: str) -> dict:
    """Parse benchmark_gradients.py output."""
    m = {}
    fwd = re.search(r"Cached:\s+(\d+\.\d+)s", text)
    m["fwd_cached_s"] = float(fwd.group(1)) if fwd else None

    ad = re.search(r"Median time:\s+(\d+\.\d+)s\s+\(\d+ repeats\)", text)
    m["ad_median_s"] = float(ad.group(1)) if ad else None

    bwdfwd = re.search(r"Backward/forward ratio:\s+(\d+\.\d+)x", text)
    m["bwd_fwd_ratio"] = float(bwdfwd.group(1)) if bwdfwd else None

    speedup = re.search(r"Speedup:\s+(\d+\.\d+)x\s+\(AD vs FD\)", text)
    m["ad_fd_speedup"] = float(speedup.group(1)) if speedup else None

    fd = re.search(r"FD grad:\s+(\d+\.\d+)s", text)
    m["fd_median_s"] = float(fd.group(1)) if fd else None

    params = ["h", "omega_b", "omega_cdm", "ln10A_s", "n_s"]
    m["ad_fd_agreement"] = {}
    for p in params:
        pat = rf"^\s+{re.escape(p)}\s+[\d.e+-]+\s+[\d.e+-]+\s+(\d+\.\d+)%"
        hit = re.search(pat, text, re.MULTILINE)
        if hit:
            m["ad_fd_agreement"][p] = float(hit.group(1))
    return m


def parse_speed(text: str) -> dict:
    """Parse benchmark_speed.py output (fit_cl or planck_cl)."""
    m = {}
    preset = re.search(r"BENCHMARK:\s+(\w+)", text)
    m["preset"] = preset.group(1) if preset else None

    cached_block = re.search(
        r"Second call \(cached\):(.*?)(?=\n\n|\Z)", text, re.DOTALL)
    if cached_block:
        block = cached_block.group(1)
        for stage in ["Background", "Thermodynamics", "Perturbations", "Harmonic"]:
            hit = re.search(rf"{stage}:\s+(\d+\.\d+)s", block)
            key = stage.lower() + "_cached_s"
            m[key] = float(hit.group(1)) if hit else None
        total = re.search(r"TOTAL \(2nd\):\s+(\d+\.\d+)s", block)
        m["total_cached_s"] = float(total.group(1)) if total else None
        m["pt_cached_s"] = m.get("perturbations_cached_s")

    m["cl_errs"] = {}
    for hit in re.finditer(
        r"^\s+(\d+)\s+([+-]\d+\.\d+)\s+([+-]\d+\.\d+)\s+([+-]\d+\.\d+)",
        text, re.MULTILINE
    ):
        l = int(hit.group(1))
        m["cl_errs"][l] = {
            "tt": float(hit.group(2)),
            "ee": float(hit.group(3)),
            "te": float(hit.group(4)),
        }
    return m


def parse_ept(text: str) -> dict:
    """Parse benchmark_ept.py output."""
    m = {}
    fwd = re.search(r"EPT forward at z=0\.38 \(cached\):\s+(\d+\.\d+)s", text)
    m["ept_fwd_cached_s"] = float(fwd.group(1)) if fwd else None

    bwdfwd = re.search(r"Backward/Forward ratio \(full pipeline\):\s+(\d+\.\d+)x", text)
    m["bwd_fwd_ratio"] = float(bwdfwd.group(1)) if bwdfwd else None

    grad = re.search(r"EPT scalar gradient \(full pipe\):\s+(\d+\.\d+)s", text)
    m["grad_cached_s"] = float(grad.group(1)) if grad else None

    pmm = re.search(r"P_mm real: max rel err = (\d+\.\d+)%", text)
    m["pmm_max_pct"] = float(pmm.group(1)) if pmm else None

    pgg = re.search(r"P_gg real.*: max rel err = (\d+\.\d+)%", text)
    m["pgg_max_pct"] = float(pgg.group(1)) if pgg else None

    upstream = re.search(r"Upstream BG\+TH\+PT \(compile \+ cached\):\s+(\d+\.\d+)s", text)
    m["upstream_s"] = float(upstream.group(1)) if upstream else None
    return m


def parse_clpp(text: str) -> dict:
    """Parse benchmark_clpp.py output."""
    m = {}
    for nl in ("none", "halofit", "ept"):
        hit = re.search(
            rf"compute_cl_pp\('{re.escape(nl)}',.*?\):\s+(\d+\.\d+)s", text)
        m[f"clpp_{nl}_s"] = float(hit.group(1)) if hit else None

    rel_diffs = re.findall(
        r"^\s+\d+\s*\|\s*[\d.e+-]+\s*\|\s*[\d.e+-]+\s*\|\s*([+-]\d+\.\d+)%",
        text, re.MULTILINE)
    m["clpp_linear_max_abspct"] = max(abs(float(x)) for x in rel_diffs) if rel_diffs else None

    m["bb_ratios"] = {}
    for hit in re.finditer(
        r"^\s+(\d+)\s*\|\s*[\d.e+-]+\s*\|\s*[\d.e+-]+\s*\|\s*([\d.]+|nan)",
        text, re.MULTILINE
    ):
        l = int(hit.group(1))
        val = hit.group(2)
        m["bb_ratios"][l] = float(val) if val != "nan" else None
    m["bb_ratios"] = {l: m["bb_ratios"].get(l) for l in [2, 10, 30, 50, 80, 100, 150, 200]}
    return m


def parse_accuracy(text: str) -> dict:
    """Parse accuracy_classpt.py output."""
    m = {"spectra": {}, "all_pass": None}
    for hit in re.finditer(
        r"^\s+(\w+)\s+([\d.]+)%\s+([\d.]+)%\s+[\d.]+\s+[✓✗]\s+(PASS|FAIL)",
        text, re.MULTILINE
    ):
        name = hit.group(1)
        m["spectra"][name] = {
            "max_pct": float(hit.group(2)),
            "mean_pct": float(hit.group(3)),
            "pass": hit.group(4) == "PASS",
        }
    m["all_pass"] = bool(re.search(r"ALL SPECTRA PASS", text))
    if re.search(r"FAILING spectra", text):
        m["all_pass"] = False
    return m


# All thresholds from BENCHMARK.md §4
_THRESHOLDS = {
    # §4.1 timing — GPU A100 targets
    "fit_cl.total_cached_s":        {"limit": 50.0,  "op": "lt", "warn_pct": 0.20},
    "planck_cl.total_cached_s":     {"limit": 600.0, "op": "lt", "warn_pct": 0.20},
    "fit_cl.pt_cached_s":           {"limit": 35.0,  "op": "lt", "warn_pct": 0.20},
    "planck_cl.pt_cached_s":        {"limit": 600.0, "op": "lt", "warn_pct": 0.20},
    "ept.ept_fwd_cached_s":         {"limit": 2.0,   "op": "lt", "warn_pct": 0.20},
    "gradients.bwd_fwd_ratio":      {"limit": 4.0,   "op": "lt", "warn_pct": 0.20},
    "ept.bwd_fwd_ratio":            {"limit": 5.0,   "op": "lt", "warn_pct": 0.20},
    # §4.1 solver
    "solvers.rodas5_vs_kv_max_pct": {"limit": 0.1,   "op": "lt", "warn_pct": 0.20},
    "solvers.rodas5_speedup":       {"limit": 2.0,   "op": "gt", "warn_pct": 0.20},
    # §4.2 accuracy
    "ept.pmm_max_pct":              {"limit": 1.0,   "op": "lt", "warn_pct": 0.20},
    "ept.pgg_max_pct":              {"limit": 1.0,   "op": "lt", "warn_pct": 0.20},
    "clpp.clpp_linear_max_abspct":  {"limit": 1.0,   "op": "lt", "warn_pct": 0.20},
    "accuracy.all_pass":            {"limit": True,  "op": "eq", "warn_pct": None},
}
_GRAD_PARAMS = ["h", "omega_b", "omega_cdm", "ln10A_s", "n_s"]
_BB_THRESHOLDS = {
    2: (0.95, 1.05), 10: (0.95, 1.05), 30: (0.95, 1.05),
    50: (0.95, 1.05), 80: (0.95, 1.05), 100: (0.95, 1.05),
    150: (0.90, 1.10), 200: (0.90, 1.10),
}
_CL_ERR_THRESHOLD = 0.5


def _scalar_check(value, threshold_def) -> str:
    if value is None:
        return MetricStatus.MISSING
    limit = threshold_def["limit"]
    op = threshold_def["op"]
    warn_pct = threshold_def.get("warn_pct")
    if op == "lt":
        passes = value < limit
        near = warn_pct and value >= limit * (1 - warn_pct) and value < limit
    elif op == "gt":
        passes = value > limit
        near = warn_pct and value <= limit * (1 + warn_pct) and value > limit
    elif op == "eq":
        passes = value == limit
        near = False
    else:
        raise ValueError(f"Unknown op: {op}")
    if not passes:
        return MetricStatus.FAIL
    if near:
        return MetricStatus.WARN
    return MetricStatus.PASS


def check_thresholds(metrics: dict) -> dict:
    """Check all metrics against BENCHMARK.md §4 thresholds."""
    checked = {}

    def _add(key, value, status, threshold, description=""):
        checked[key] = {
            "value": value,
            "status": status,
            "threshold": threshold,
            "description": description,
        }

    for key, tdef in _THRESHOLDS.items():
        script, attr = key.split(".", 1)
        script_metrics = metrics.get(script, {})
        value = script_metrics.get(attr) if isinstance(script_metrics, dict) else None
        status = _scalar_check(value, tdef)
        _add(key, value, status, tdef["limit"], "BENCHMARK.md §4")

    grad = metrics.get("gradients", {})
    agreement = grad.get("ad_fd_agreement", {}) if grad else {}
    for p in _GRAD_PARAMS:
        val = agreement.get(p)
        status = _scalar_check(val, {"limit": 1.0, "op": "lt", "warn_pct": 0.20})
        _add(f"gradients.ad_fd_{p}_pct", val, status, 1.0, "AD/FD < 1%")

    clpp = metrics.get("clpp", {})
    bb = clpp.get("bb_ratios", {}) if clpp else {}
    for l, (lo, hi) in _BB_THRESHOLDS.items():
        val = bb.get(l) if bb else None
        if val is None:
            status = MetricStatus.MISSING
        elif lo <= val <= hi:
            margin_lo = (val - lo) / (hi - lo)
            margin_hi = (hi - val) / (hi - lo)
            status = MetricStatus.WARN if min(margin_lo, margin_hi) < 0.20 else MetricStatus.PASS
        else:
            status = MetricStatus.FAIL
        _add(f"clpp.bb_ratio_l{l}", val, status, f"[{lo},{hi}]", f"BB ratio ℓ={l}")

    for preset in ("fit_cl", "planck_cl"):
        speed = metrics.get(preset, {})
        cl_errs = speed.get("cl_errs", {}) if speed else {}
        for l in [20, 100, 500, 1000]:
            errs = cl_errs.get(l, {}) if cl_errs else {}
            for spec in ("tt", "ee"):
                val = errs.get(spec)
                abs_val = abs(val) if val is not None else None
                status = _scalar_check(abs_val, {"limit": _CL_ERR_THRESHOLD, "op": "lt", "warn_pct": 0.20})
                _add(f"{preset}.cl_{spec}_l{l}_abspct", abs_val, status, _CL_ERR_THRESHOLD)

    return checked


STATUS_ICON = {
    MetricStatus.PASS: "✓",
    MetricStatus.FAIL: "✗",
    MetricStatus.WARN: "⚠",
    MetricStatus.MISSING: "·",
}


def build_results_dict(result_dir: Path, found: dict, missing: list,
                       parsed: dict, checked: dict) -> dict:
    node = None
    for p in result_dir.glob("*-a100-*.txt"):
        node = p.name.split("-a100-")[0]
        break
    summary = {
        "n_pass": sum(1 for v in checked.values() if v["status"] == MetricStatus.PASS),
        "n_fail": sum(1 for v in checked.values() if v["status"] == MetricStatus.FAIL),
        "n_warn": sum(1 for v in checked.values() if v["status"] == MetricStatus.WARN),
        "n_missing": sum(1 for v in checked.values() if v["status"] == MetricStatus.MISSING),
    }
    return {
        "date": str(result_dir.name) if result_dir.name[0].isdigit() else datetime.now().strftime("%Y-%m-%d"),
        "node": node,
        "result_dir": str(result_dir),
        "files_found": list(found.keys()),
        "files_missing": missing,
        "metrics": parsed,
        "thresholds": {k: {**v, "value": str(v["value"])} for k, v in checked.items()},
        "summary": summary,
        "analyzed_at": datetime.now().isoformat(),
    }


def write_report(results: dict, out_dir: Path) -> None:
    """Write results.json and REPORT.md to out_dir."""
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    lines = []
    d = results["date"]
    node = results.get("node", "unknown")
    s = results["summary"]
    lines += [
        f"# clax benchmark report — {d}",
        "",
        "## Platform",
        f"- Node: {node}",
        f"- Files present: {', '.join(results['files_found']) or 'none'}",
        f"- Files missing: {', '.join(results['files_missing']) or 'none'}",
        f"- Analyzed: {results['analyzed_at']}",
        "",
        f"## Scorecard: {s['n_pass']} PASS · {s['n_fail']} FAIL · "
        f"{s['n_warn']} WARN · {s['n_missing']} MISSING",
        "",
        "| Metric | Value | Status | Threshold |",
        "|---|---|---|---|",
    ]
    for key, v in results["thresholds"].items():
        icon = STATUS_ICON.get(v["status"], "?")
        val_str = f"{v['value']}" if v["value"] is not None else "—"
        lines.append(f"| `{key}` | {val_str} | {icon} {v['status']} | {v['threshold']} |")

    lines += ["", "## Raw output tails", ""]
    raw_dir = out_dir
    for script in ["solvers", "gradients", "fit_cl", "ept", "clpp", "accuracy", "planck_cl"]:
        paths = list(raw_dir.glob(f"*-a100-{script}.txt"))
        lines += [f"### {script}"]
        if paths:
            tail = paths[0].read_text().strip().split("\n")
            lines += ["```"] + tail[-15:] + ["```", ""]
        else:
            lines += ["*(not yet available)*", ""]

    (out_dir / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--dir", type=Path, default=None)
    args = parser.parse_args()

    result_dir = args.dir or Path(f"/ptmp/minh/clax-bench/{args.date}")
    if not result_dir.exists():
        print(f"Result directory not found: {result_dir}", file=sys.stderr)
        sys.exit(2)

    found, missing = find_result_files(result_dir)
    if not found:
        print(f"No result files found in {result_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"Found: {list(found.keys())}")
    print(f"Missing: {missing}")

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

    checked = check_thresholds(parsed)
    results = build_results_dict(result_dir, found, missing, parsed, checked)
    write_report(results, result_dir)

    n_fail = results["summary"]["n_fail"]
    n_missing = results["summary"]["n_missing"]
    print(f"\nReport written to {result_dir}/REPORT.md")
    print(f"Summary: {results['summary']}")

    if n_fail > 0:
        sys.exit(1)
    elif n_missing > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
