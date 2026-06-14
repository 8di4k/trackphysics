"""Run trackphysics validation against the TT3D dataset (real multi-camera 3D truth).

Pass the dataset's ``data/evaluation`` directory as argv[1] or via the ``TT3D_EVAL_ROOT``
env var. For each viewpoint (clean + noisy) it loads the trajectories, runs the engine, and
reports per-view aggregates: metric-emission count, in-plane speed error (the fair monocular
number), full-3D error (exposes depth-blindness), and CI coverage rate.

    python -m validation.run_tt3d /path/to/tt3d/data/evaluation

Note: TT3D ships no license (all rights reserved) — internal validation only; clear terms
before publishing numbers. Table tennis is a spin/Magnus + depth stress regime for
gravity-as-a-ruler, so read low CI-coverage as "this regime is hard AND/OR the engine is
overconfident" — disambiguate with a no-spin heavy-ball anchor + a radar speed check.
"""

from __future__ import annotations

import os
import sys

from validation.datasets_tt3d import VIEWS, DatasetUnavailable, load_tt3d
from validation.run_3d_validation import run


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TT3D_EVAL_ROOT")
    if not root:
        print("usage: python -m validation.run_tt3d <data/evaluation>  (or set TT3D_EVAL_ROOT)")
        return
    header = (
        f"{'view':16s} {'metric/total':>13s} {'inplane(mean/med)':>18s} "
        f"{'full3D':>7s} {'CI_cover':>8s}"
    )
    print(header)
    for view in VIEWS:
        for noisy in (False, True):
            try:
                report = run(load_tt3d(root, view=view, noisy=noisy))
            except DatasetUnavailable as exc:
                print(f"  unavailable: {exc}")
                return
            s = report.summary()
            tag = f"{view}{'' if noisy else '_clean'}"
            if s.get("n_metric", 0) > 0:
                inplane = f"{s['mean_inplane_error_m_s']:.2f}/{s['median_inplane_error_m_s']:.2f}"
                print(
                    f"{tag:16s} {int(s['n_metric']):>5d}/{int(s['n_total']):<7d} "
                    f"{inplane:>18s} {s['mean_full3d_error_m_s']:>7.2f} {s['ci_coverage_rate']:>8.2f}"
                )
            else:
                print(f"{tag:16s} {int(s.get('n_metric', 0)):>5d}/{int(s['n_total']):<7d}  (no metric)")
    print(
        "\nin-plane = fair monocular number; full3D large on 'back' = depth-blindness (ball "
        "flies along the optical axis). Low CI coverage = engine overconfident on this "
        "spin/Magnus regime — confirm against a no-spin anchor before concluding it is a bug."
    )


if __name__ == "__main__":
    main()
