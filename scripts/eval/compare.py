#!/usr/bin/env python
"""Compare two evaluate_stitch.py reports (before vs after) and check SC thresholds.

Feature: stitching-artifact-correction (SC-002 seam >=90% reduction, SC-003 stripe
>=75% reduction, SC-004 in-strip content preserved). Exits non-zero if a gate fails,
so the dev loop fails loudly (constitution IV).
"""

from __future__ import annotations

import argparse
import json
import sys


def pct_reduction(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return 100.0 * (before - after) / before


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("before_json")
    ap.add_argument("after_json")
    ap.add_argument("--seam-target", type=float, default=90.0, help="%% reduction (SC-002)")
    ap.add_argument("--stripe-target", type=float, default=75.0, help="%% reduction (SC-003)")
    ap.add_argument("--instrip-tol", type=float, default=0.05, help="max rel. mean drift (SC-004)")
    args = ap.parse_args(argv)

    with open(args.before_json) as f:
        b = json.load(f)
    with open(args.after_json) as f:
        a = json.load(f)

    seam_red = pct_reduction(b["seam_discontinuity"], a["seam_discontinuity"])
    stripe_red = pct_reduction(b["stripe_energy"], a["stripe_energy"])
    mean_drift = abs(a["in_strip_mean"] - b["in_strip_mean"]) / (abs(b["in_strip_mean"]) + 1e-9)

    print(f"seam discontinuity : {b['seam_discontinuity']:.3f} -> {a['seam_discontinuity']:.3f} "
          f"({seam_red:.1f}% reduction; target >={args.seam_target}%) [SC-002]")
    print(f"stripe energy      : {b['stripe_energy']:.4f} -> {a['stripe_energy']:.4f} "
          f"({stripe_red:.1f}% reduction; target >={args.stripe_target}%) [SC-003]")
    print(f"in-strip mean drift: {mean_drift*100:.2f}% (tol {args.instrip_tol*100:.0f}%) [SC-004]")

    gates = {
        "SC-002 seam": seam_red >= args.seam_target,
        "SC-003 stripe": stripe_red >= args.stripe_target,
        "SC-004 in-strip": mean_drift <= args.instrip_tol,
    }
    failed = [k for k, ok in gates.items() if not ok]
    if failed:
        print(f"FAIL: {', '.join(failed)}")
        return 1
    print("PASS: all gates met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
