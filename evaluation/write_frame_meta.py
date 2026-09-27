import json
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Write each scene's evaluation window to <scene>/frame_meta.json (read by proxypose-eval)."
    )
    ap.add_argument("--filter_json", required=True, help="Benchmark window file, e.g. evaluation/benchmarks/ho3d/w1_f49.json")
    ap.add_argument("--uniform_root", required=True, help="Root of the benchmark in uniform format")
    args = ap.parse_args()

    with open(args.filter_json) as f:
        data = json.load(f)

    dataset_name = data["dataset"]
    for scene in data["scenes"]:
        scene_path = Path(args.uniform_root) / f"{dataset_name}_{scene['scene_id']:05d}"
        if not scene_path.is_dir():
            print(f"skip {scene_path}, not found")
            continue

        with open(scene_path / "frame_meta.json", "w") as f:
            json.dump(scene, f, indent=4)
        print(f"wrote {scene_path / 'frame_meta.json'}")


if __name__ == "__main__":
    main()
