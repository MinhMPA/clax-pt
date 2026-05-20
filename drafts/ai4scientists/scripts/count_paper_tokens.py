#!/usr/bin/env python3
"""Audit Claude Code token usage for the clax-pt paper appendix.

Reads ~/.claude/projects/*/*.jsonl on the local machine, filters to sessions
in a date window, aggregates token usage per model, and computes USD cost
from a fixed pricing table. Designed to produce auditable, reproducible
numbers for the AI4Science paper's "57 sessions" and "33 of 57" claims.

Usage (run on the machine that did the clax-pt bring-up):
    python3 count_paper_tokens.py
    python3 count_paper_tokens.py --start 2026-03-29 --end 2026-04-12
    python3 count_paper_tokens.py --start 2026-04-02 --end 2026-04-12  # tighter window

Output:
    paper_tokens.json — top-level summary (counts, costs, breakdowns)
    paper_tokens_per_session.csv — one row per session

Hard rules:
    - Pricing constants below are mid-2026 estimates. VERIFY against Anthropic's
      published prices for the actual window before quoting in the paper.
    - This script does NOT use LLMs to classify sessions. Intervention detection
      is a coarse keyword heuristic; treat as a LOWER BOUND on supervision-heavy sessions.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import re
import sys

# ---------------------------------------------------------------------------
# Pricing (USD per million tokens). VERIFY before quoting.
# Source: https://www.anthropic.com/api/pricing
# ---------------------------------------------------------------------------
PRICING = {
    "claude-opus-4-7":            {"input": 15.0, "output": 75.0, "cache_create": 18.75, "cache_read": 1.50},
    "claude-opus-4-6":            {"input": 15.0, "output": 75.0, "cache_create": 18.75, "cache_read": 1.50},
    "claude-sonnet-4-6":          {"input":  3.0, "output": 15.0, "cache_create":  3.75, "cache_read": 0.30},
    "claude-sonnet-4-5":          {"input":  3.0, "output": 15.0, "cache_create":  3.75, "cache_read": 0.30},
    "claude-haiku-4-5-20251001":  {"input":  0.25,"output":  1.25,"cache_create":  0.30, "cache_read": 0.025},
}

# Keyword heuristic for "user invoked physics-domain language."
# LOWER BOUND only; tune as needed.
PHYSICS_KEYWORDS = re.compile(
    r"\b("
    r"anisotrop\w*|sigmatot|isotropic|fudge|architect\w*|"
    r"class[-_ ]?pt|nonlinear_pt|fftlog|ir resumm\w*|bao damping|"
    r"hexadecapole|monopole|quadrupole|legendre|gauss[-_ ]?legendre|"
    r"multipole|m22|m13|uv counterterm|sigma_v|stochastic\w*|"
    r"redshift[-_ ]?space|rsd|biased tracer|eft|eftoflss|"
    r"halofit|limber|bessel|kernel|tree[-_ ]?level|loop[-_ ]?level|"
    r"column[-_ ]?major|row[-_ ]?major|hermitian|symmetric|lapack|"
    r"backsolve|adjoint|jacobian|jvp|vjp|"
    r"convention|coefficient|placeholder"
    r")\b",
    re.IGNORECASE,
)


def find_clax_dirs(claude_root: pathlib.Path) -> list[pathlib.Path]:
    base = claude_root.expanduser() / "projects"
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and "clax" in p.name.lower())


def parse_iso_date(s):
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


def parse_session(jsonl_path: pathlib.Path):
    by_model: dict[str, dict[str, int]] = {}
    timestamps: list[str] = []
    user_blobs: list[str] = []

    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if ts := rec.get("timestamp"):
                    timestamps.append(ts)

                msg = rec.get("message") or {}
                if msg.get("role") == "user":
                    c = msg.get("content")
                    if isinstance(c, str):
                        user_blobs.append(c)
                    elif isinstance(c, list):
                        for item in c:
                            if isinstance(item, dict) and item.get("type") == "text":
                                user_blobs.append(item.get("text", ""))

                usage = msg.get("usage") or {}
                if usage:
                    model = msg.get("model") or "unknown"
                    slot = by_model.setdefault(model, {
                        "input": 0, "output": 0, "cache_create": 0, "cache_read": 0, "turns": 0
                    })
                    slot["input"]        += usage.get("input_tokens", 0) or 0
                    slot["output"]       += usage.get("output_tokens", 0) or 0
                    slot["cache_create"] += usage.get("cache_creation_input_tokens", 0) or 0
                    slot["cache_read"]   += usage.get("cache_read_input_tokens", 0) or 0
                    slot["turns"]        += 1
    except OSError:
        return None

    if not by_model:
        return None

    return {
        "path": str(jsonl_path),
        "session_id": jsonl_path.stem,
        "start": min(timestamps) if timestamps else None,
        "end":   max(timestamps) if timestamps else None,
        "by_model": by_model,
        "physics_keyword_present": any(PHYSICS_KEYWORDS.search(b) for b in user_blobs),
    }


def compute_cost(by_model, pricing=PRICING):
    total = 0.0
    breakdown = {}
    warnings: list[str] = []
    for model, u in by_model.items():
        prices = pricing.get(model)
        if not prices:
            warnings.append(f"unknown model {model!r}; using opus-4-7 fallback pricing")
            prices = pricing["claude-opus-4-7"]
        cost = (
            u["input"]        * prices["input"]
            + u["output"]      * prices["output"]
            + u["cache_create"]* prices["cache_create"]
            + u["cache_read"]  * prices["cache_read"]
        ) / 1e6
        breakdown[model] = {
            "cost_usd": round(cost, 4),
            "turns": u["turns"],
            "input_tokens": u["input"],
            "output_tokens": u["output"],
            "cache_create_tokens": u["cache_create"],
            "cache_read_tokens": u["cache_read"],
        }
        total += cost
    return round(total, 4), breakdown, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-03-29")
    ap.add_argument("--end",   default="2026-04-12")
    ap.add_argument("--claude-root", default="~/.claude")
    ap.add_argument("--output-json", default="paper_tokens.json")
    ap.add_argument("--output-csv",  default="paper_tokens_per_session.csv")
    ap.add_argument("--project-filter", default="clax")
    args = ap.parse_args()

    start = parse_iso_date(args.start)
    end   = parse_iso_date(args.end)
    if start is None or end is None or start > end:
        print(f"Bad date window: {args.start} .. {args.end}", file=sys.stderr)
        return 2

    dirs = [d for d in find_clax_dirs(pathlib.Path(args.claude_root))
            if args.project_filter.lower() in d.name.lower()]

    print(f"# Window: {start} .. {end}")
    print(f"# Found {len(dirs)} project dir(s) matching '{args.project_filter}':")
    for d in dirs:
        print(f"#   {d.name}: {len(list(d.glob('*.jsonl')))} JSONL files")

    in_window, out_window, no_usage = [], [], 0
    for d in dirs:
        for f in sorted(d.glob("*.jsonl")):
            s = parse_session(f)
            if s is None:
                no_usage += 1
                continue
            sd = parse_iso_date(s["start"])
            if sd is None or sd < start or sd > end:
                out_window.append(s)
            else:
                in_window.append(s)

    print(f"# In window:        {len(in_window)} sessions")
    print(f"# Outside window:   {len(out_window)} sessions")
    print(f"# No usage records: {no_usage} files (empty/aborted)")

    agg: dict[str, dict[str, int]] = {}
    intervention = 0
    for s in in_window:
        for m, u in s["by_model"].items():
            slot = agg.setdefault(m, {"input": 0, "output": 0, "cache_create": 0, "cache_read": 0, "turns": 0})
            for k in slot:
                slot[k] += u[k]
        if s["physics_keyword_present"]:
            intervention += 1

    total_cost, by_model_breakdown, warnings = compute_cost(agg)
    total_turns = sum(u["turns"] for u in agg.values())

    summary = {
        "window": {"start": str(start), "end": str(end)},
        "sessions_in_window": len(in_window),
        "sessions_outside_window": len(out_window),
        "sessions_no_usage_data": no_usage,
        "total_turns_in_window": total_turns,
        "total_cost_usd": total_cost,
        "intervention_heuristic": {
            "sessions_flagged": intervention,
            "total_in_window": len(in_window),
            "rate": (round(intervention / len(in_window), 3) if in_window else None),
            "method": "regex on user-message text only; LOWER BOUND",
        },
        "by_model": by_model_breakdown,
        "pricing_warnings": warnings,
        "pricing_source": "Anthropic pricing as of mid-2026 (USD/MTok). VERIFY before quoting.",
        "pricing_table": PRICING,
    }

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "session_id", "start", "end",
            "turns", "input_tokens", "output_tokens",
            "cache_create_tokens", "cache_read_tokens",
            "cost_usd", "physics_keyword_present", "models",
        ])
        for s in sorted(in_window, key=lambda x: x["start"] or ""):
            ti  = sum(u["input"]        for u in s["by_model"].values())
            to  = sum(u["output"]       for u in s["by_model"].values())
            tcc = sum(u["cache_create"] for u in s["by_model"].values())
            tcr = sum(u["cache_read"]   for u in s["by_model"].values())
            tt  = sum(u["turns"]        for u in s["by_model"].values())
            cost, _, _ = compute_cost(s["by_model"])
            w.writerow([
                s["session_id"], s["start"], s["end"],
                tt, ti, to, tcc, tcr, cost,
                s["physics_keyword_present"],
                ";".join(sorted(s["by_model"].keys())),
            ])

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {args.output_json}")
    print(f"Wrote: {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
