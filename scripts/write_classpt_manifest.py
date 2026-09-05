#!/usr/bin/env python
# scripts/write_classpt_manifest.py
"""Scan reference_data/classpt/**/*.npz and write MANIFEST.md (spec §4.8,
§6.4 "maps file -> exact invocation").

Runs in either env (numpy only).  Idempotent; rerun after any regeneration.
Skipped runs are appended from --skipped "path: reason" arguments.

A5 review fix round 1 (Blocking 2): every row now also carries `z_pk`
(asserted single-valued -- Ruling 19 makes single-z provenance the
load-bearing property of this file) and a reconstructed `invocation`: the
`generate_classpt_reference.py` CLI call that reproduces the row, derived
from the stored `ap`/`cb`/`omfid`/`use_ppf` fields, `bias_json` compared
against the case's fiducial/nonzero bias dicts, and the filename's tag
suffix. The single global `Patches:`/`CLASS-PT` header line is now asserted
common across every scanned file rather than read from the leaked `for`
loop variable (quality review minor 1 / Blocking 2b).
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from scripts import validation_cosmologies as vc

_USE_PPF_FLAG = {"default": None, "yes": True, "no": False}


def _bias_flag(bias_json: dict, stem: str) -> tuple[str, bool]:
    """(" --bias nonzero" or "", used_filename_fallback).

    Primary path: compare `bias_json` against vc.BIAS / vc.BIAS_NONZERO.
    Fallback (should not trigger on today's 48 files -- both dicts are
    exact literals and every row matches one of them): if `bias_json`
    matches neither, the stored keys can't disambiguate, so fall back to
    the filename's "_biasnz" suffix (spec's own file-layout convention,
    validation_cosmologies.reference_path) and flag the row as such.
    """
    if bias_json == vc.BIAS:
        return "", False
    if bias_json == vc.BIAS_NONZERO:
        return " --bias nonzero", False
    return (" --bias nonzero" if "_biasnz" in stem else ""), True


def _tag(stem: str, z: float, ap: bool, omfid: float, cb: bool, bias_nonzero: bool) -> str | None:
    """Filename tag suffix, by stripping the deterministic prefix that
    validation_cosmologies.reference_path() would build for these fields.
    None if the stem doesn't match that prefix at all (unexpected name)."""
    prefix = f"z{z:.3f}_" + (f"ap_omfid{omfid:g}" if ap else "noap") + ("_cb" if cb else "_m")
    if bias_nonzero:
        prefix += "_biasnz"
    if stem == prefix:
        return ""
    if stem.startswith(prefix + "_"):
        return stem[len(prefix) + 1:]
    return None


def _expected_params(case: str, z_pk: float, *, ap: bool, omfid: float, cb: bool, use_ppf: str) -> dict:
    """Reconstruct the classy params dict classpt_params_from() would build
    for this (case, z, ap, omfid, cb, use_ppf) with no --class-extra, for
    diffing against the row's actual params_json (see `invocation` below)."""
    if case == "legacy_fiducial":
        cosmo, yhe = vc.LEGACY_CLASSPT_FIDUCIAL, None
    else:
        cosmo, yhe = vc.cosmo_params(case), vc.Y_HE_CLAX
    return vc.classpt_params_from(cosmo, z_list=[z_pk], ap=ap, omfid=omfid, cb=cb,
                                   yhe=yhe, use_ppf=_USE_PPF_FLAG[use_ppf])


def _invocation(rel, z_pk_str: str, z_pk: float, ap: bool, omfid: float, cb: bool,
                 use_ppf: str, bias_json: dict, params: dict) -> tuple[str, bool]:
    """(invocation string, used_bias_filename_fallback)."""
    case = rel.parts[0]
    stem = rel.stem
    head = "--legacy" if case == "legacy_fiducial" else f"--cosmology {case}"
    parts = [head, f"--z-list {z_pk_str}", f"--ap {'yes' if ap else 'no'}"]
    if omfid != vc.OMFID:
        parts.append(f"--omfid {omfid:g}")
    parts.append(f"--cb {'yes' if cb else 'no'}")
    bias_flag, bias_fallback = _bias_flag(bias_json, stem)
    if bias_flag:
        parts.append(bias_flag.strip())
    tag = _tag(stem, z_pk, ap, omfid, cb, bool(bias_flag))
    note = ""
    if tag is None:
        note = "  <!-- filename does not match the expected z/ap/cb/bias prefix; tag undetermined -->"
    elif tag:
        parts.append(f"--tag {tag}")
    if use_ppf != "default":
        parts.append(f"--use-ppf {use_ppf}")
    expected = _expected_params(case, z_pk, ap=ap, omfid=omfid, cb=cb, use_ppf=use_ppf)
    extra = {k: params[k] for k in params if k not in expected or params[k] != expected[k]}
    extra.pop("z_pk", None)  # z_pk is already represented by --z-list
    if extra:
        note += (f"  <!-- params_json has keys not reproduced by these flags "
                 f"(likely --class-extra): {sorted(extra)} -->")
    return " ".join(parts) + note, bias_fallback


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skipped", nargs="*", default=[], help='"relative/path.npz: reason"')
    a = p.parse_args(argv)
    rows = []
    commits: set[str] = set()
    patch_blobs: set[str] = set()
    bias_fallback_files = []
    for path in sorted(vc.REFERENCE_ROOT.rglob("*.npz")):
        d = np.load(path)
        rel = path.relative_to(vc.REFERENCE_ROOT)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        params = json.loads(str(d["params_json"]))
        psha = hashlib.sha256(str(d["params_json"]).encode()).hexdigest()[:12]
        commits.add(str(d["classpt_commit"]))
        patch_blobs.add(str(d["patches_sha256"]))

        z_pk_str = str(params["z_pk"])
        z_pk_parts = z_pk_str.split(",")
        assert len(z_pk_parts) == 1, (
            f"z_pk has {len(z_pk_parts)} values in {rel} (params_json z_pk={z_pk_str!r}) -- "
            "Ruling 19 requires every campaign file to be a single-z CLASS-PT run; report this.")
        z_pk = float(z_pk_parts[0])

        ap, cb = bool(d["ap"]), bool(d["cb"])
        omfid = float(d["omfid"])
        use_ppf = str(d["use_ppf"])
        bias_json = json.loads(str(d["bias_json"]))
        invocation, bias_fallback = _invocation(rel, z_pk_parts[0], z_pk, ap, omfid, cb,
                                                 use_ppf, bias_json, params)
        if bias_fallback:
            bias_fallback_files.append(str(rel))

        rows.append(f"| `{rel}` | `{sha}` | `{d['classpt_commit']}` | {z_pk:g} | "
                    f"{float(d['hratio']):.6f} | {float(d['Dratio']):.6f} | {float(d['fz']):.6f} | "
                    f"`{psha}` | `{invocation}` |")

    if rows:
        assert len(commits) == 1, f"multiple classpt_commit values across scanned files: {commits}"
        assert len(patch_blobs) == 1, f"multiple patches_sha256 blobs across scanned files: {patch_blobs}"
        patches = json.loads(next(iter(patch_blobs)))
    else:
        patches = {}

    lines = ["# CLASS-PT reference manifest", "",
             "Generated by `scripts/write_classpt_manifest.py`; files by "
             "`scripts/generate_classpt_reference.py` in the `classpt` env "
             "(`scripts/setup_classpt_env.sh`). Layout: spec §4.8.", "",
             "Every campaign file is one `--z-list <z>` CLASS-PT run (Ruling 19); "
             "`z_pk` in each file's `params_json` is a single value.", "",
             "Patches: " + ", ".join(f"`{k}` `{v[:16]}`" for k, v in patches.items()), "",
             "| file | sha256[:16] | CLASS-PT | z_pk | hratio | Dratio | f | params sha[:12] | invocation |",
             "|---|---|---|---|---|---|---|---|---|", *rows, ""]
    if bias_fallback_files:
        lines += ["`bias_json` matched neither the fiducial nor the nonzero bias dict for: "
                  + ", ".join(f"`{f}`" for f in bias_fallback_files)
                  + " -- `--bias nonzero` above was inferred from the filename's `_biasnz` "
                  "suffix instead.", ""]
    if a.skipped:
        lines += ["## Skipped", "", *[f"- {s}" for s in a.skipped], ""]
    (vc.REFERENCE_ROOT / "MANIFEST.md").write_text("\n".join(lines))
    print(f"MANIFEST.md: {len(rows)} files, {len(a.skipped)} skipped")


if __name__ == "__main__":
    main()
