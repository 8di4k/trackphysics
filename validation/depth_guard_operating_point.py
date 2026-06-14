"""Operating-point validation of the opt-in depth-domination guard (§10 tier-hole).

The guard's worth is NOT the monotone "keep the top-K by aspect -> error drops" average (that
is near-tautological selection). It is the CONFUSION at the actual decision threshold: of the
metric emissions it downgrades, how many were truly 3D-blind (precision), how many genuinely
3D-honest arcs it wrongly sacrifices, and how many blind arcs it still misses (recall). The
asymmetry is deliberate: a downgrade is a safe under-claim, so the hard floor is tuned for high
precision / low false-downgrade, and the recall gap (ambiguous middle) is covered by the
CONTINUOUS confidence penalty rather than a cliff.

"3D-blind" label (point-only-agnostic, from GT): the recovered in-plane metric misses most of
the true 3-D motion, i.e. full3d_err > 0.5*inplane_truth + 0.5 m/s.

Run (from the repo root):  python -m validation.depth_guard_operating_point --root <eval>

NOTE: TT3D is all-rights-reserved — the printed numbers are an INTERNAL artifact (license).
This script (code) is fine to keep; do not commit its output.
"""

from __future__ import annotations

import argparse
import os

from validation.datasets_tt3d import VIEWS, load_tt3d
from validation.run_3d_validation import validate_trajectory

import trackphysics as tp
from trackphysics.core.grounding import DepthDominationGuard, GroundingContext


def run(root: str, guard: DepthDominationGuard, *, noisy: bool = False) -> dict[str, float]:
    tp_ = fp_ = fn_ = tn_ = 0
    off_err: list[float] = []
    on_err: list[float] = []
    for view in VIEWS:
        for traj in load_tt3d(root, view=view, noisy=noisy):
            off = tp.analyze(traj.track, preset="sphere", grounding=GroundingContext()).trajectory
            if off.tier is not tp.Tier.METRIC:
                continue
            r = validate_trajectory(traj)
            if r.full3d_error is None or r.inplane_truth is None:
                continue
            blind = r.full3d_error > (0.5 * r.inplane_truth + 0.5)
            on = tp.analyze(
                traj.track, preset="sphere", grounding=GroundingContext(depth_guard=guard)
            ).trajectory
            down = on.tier is not tp.Tier.METRIC
            tp_ += int(down and blind)
            fp_ += int(down and not blind)
            fn_ += int((not down) and blind)
            tn_ += int((not down) and not blind)
            off_err.append(r.full3d_error)
            if not down:
                on_err.append(r.full3d_error)
    prec = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
    rec = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
    return {
        "tp": tp_, "fp": fp_, "fn": fn_, "tn": tn_, "precision": prec, "recall": rec,
        "good_downgraded_frac": fp_ / (fp_ + tn_) if (fp_ + tn_) else 0.0,
        "retained_full3d_off": sum(off_err) / len(off_err) if off_err else float("nan"),
        "retained_full3d_on": sum(on_err) / len(on_err) if on_err else float("nan"),
        "n_metric_off": len(off_err), "n_metric_on": len(on_err),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("TT3D_EVAL_ROOT"))
    ap.add_argument("--soft", type=float, default=2.5)
    ap.add_argument("--hard", type=float, default=0.8)
    args = ap.parse_args()
    if not args.root:
        raise SystemExit("pass --root <tt3d>/data/evaluation (or set TT3D_EVAL_ROOT)")
    guard = DepthDominationGuard(enabled=True, soft_aspect=args.soft, hard_aspect=args.hard)
    s = run(args.root, guard)
    print(f"depth-domination guard @ soft={args.soft} hard={args.hard} (clean views):")
    print(f"  TP={s['tp']:.0f} FP={s['fp']:.0f} FN={s['fn']:.0f} TN={s['tn']:.0f}")
    print(f"  precision={s['precision']:.2f}  recall={s['recall']:.2f}  "
          f"good-arcs-wrongly-downgraded={s['good_downgraded_frac']:.2%}")
    print(
        f"  retained-METRIC mean full3D_err: {s['retained_full3d_off']:.2f} "
        f"(n={s['n_metric_off']:.0f}) -> {s['retained_full3d_on']:.2f} (n={s['n_metric_on']:.0f})"
    )


if __name__ == "__main__":
    main()
