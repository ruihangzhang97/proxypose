import json
import argparse
from pathlib import Path

FIELDS = [
    ("ATE", "ate_mean_mm", "mm"),
    ("ARE", "are_mean_deg", "deg"),
    ("RPE-t", "rpe_trans_mean_mm", "mm"),
    ("RPE-r", "rpe_rot_mean_deg", "deg"),
    ("2D", "2d_mean_px", "px"),
]


def main():
    ap = argparse.ArgumentParser(description="Print the paper metrics for every metrics JSON in a directory.")
    ap.add_argument("--metrics_dir", default="output/metrics", help="Directory of evaluate_slam.py outputs")
    ap.add_argument("--latex", action="store_true", help="Print rows as LaTeX (values joined by '&')")
    args = ap.parse_args()

    paths = sorted(Path(args.metrics_dir).glob("*.json"))
    if not paths:
        print(f"no metrics JSONs found in {args.metrics_dir}")
        return

    width = max(len(p.stem) for p in paths)
    sep = " & " if args.latex else "  "
    header = sep.join(f"{name} ({unit})" for name, _, unit in FIELDS)
    print(f"{'':<{width}}  {header}")

    for path in paths:
        with open(path) as f:
            mean = json.load(f)["mean"]
        values = sep.join(
            f"{mean[k]:.4g}" if mean.get(k) is not None else "-" for _, k, _ in FIELDS
        )
        print(f"{path.stem:<{width}}: {values}" + (" \\\\" if args.latex else ""))


if __name__ == "__main__":
    main()
