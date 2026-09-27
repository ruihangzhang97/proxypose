import cv2
import json
import trimesh
import numpy as np
from pathlib import Path


class UniformDataset:
    def __init__(
        self,
        scene_dir: str,
        obj_id: int,
        anchor_frame: int = 0,
        n_frames: int = 29,
        stride: int = 1,
        depth_source: str = "gt",
    ):
        self.scene_dir = Path(scene_dir)
        self.obj_id = obj_id
        self.anchor_frame = anchor_frame
        self.n_frames = n_frames
        self.stride = stride
        self.depth_src = depth_source
        self.da3_scale = 1.0

        with open(self.scene_dir / "scene_meta.json") as f:
            self.scene_meta = json.load(f)

        self.dataset_name = self.scene_meta["dataset"]
        self.scene_name = self.scene_meta["scene_name"]
        self.K = np.array(self.scene_meta["K"], dtype=np.float64)
        self.imw = self.scene_meta["image_width"]
        self.imh = self.scene_meta["image_height"]
        self.depth_scale_to_meters = self.scene_meta["depth_scale_to_meters"]
        self.source_frame_map = self.scene_meta.get("source_frame_map", {})

        if depth_source == "da3":
            self.depth_dir = self.scene_dir / "da3_depth"
            gt_path = self.scene_dir / "depth" / f"{anchor_frame:04d}_depth.png"
            da3_path = self.depth_dir / f"{anchor_frame:04d}_depth.png"
            if gt_path.exists() and da3_path.exists():
                gt = (
                    cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED).astype(np.float64)
                    * self.depth_scale_to_meters
                )
                da3 = (
                    cv2.imread(str(da3_path), cv2.IMREAD_UNCHANGED).astype(np.float64)
                    * self.depth_scale_to_meters
                )
                valid = (gt > 0) & (da3 > 0)
                if valid.sum() > 100:
                    self.da3_scale = float(np.median(gt[valid] / da3[valid]))
        else:
            self.depth_dir = self.scene_dir / "depth"

        if str(obj_id) not in self.scene_meta["objects"]:
            raise KeyError(f"obj_id {obj_id} not in scene {self.scene_name}")
        self.obj_info = self.scene_meta["objects"][str(obj_id)]

        self.frame_ids = [anchor_frame + i * stride for i in range(n_frames)]
        if max(self.frame_ids) >= self.scene_meta["n_frames"]:
            raise ValueError(
                f"Window [{self.frame_ids[0]}, {self.frame_ids[-1]}] exceeds scene length"
            )

        self.meta_cache = {}
        for fid in self.frame_ids:
            fmeta = self.load_frame_meta(fid)
            if str(obj_id) not in fmeta["objects"]:
                raise ValueError(
                    f"obj {obj_id} missing from frame {fid} in {self.scene_name}"
                )

        self.init_mesh = None

    def load_frame_meta(self, frame_id: int) -> dict:
        if frame_id in self.meta_cache:
            return self.meta_cache[frame_id]
        path = self.scene_dir / "meta" / f"{frame_id:04d}_meta.json"
        with open(path) as f:
            meta = json.load(f)
        self.meta_cache[frame_id] = meta
        return meta

    def idx_to_frame_id(self, idx: int) -> int:
        return self.frame_ids[idx]

    def source_to_uniform(self, source_frame_id) -> int:
        return self.source_frame_map[str(source_frame_id)]

    def get_color(self, idx: int) -> np.ndarray:
        fid = self.idx_to_frame_id(idx)
        bgr = cv2.imread(str(self.scene_dir / "color" / f"{fid:04d}_color.png"))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_depth(self, idx: int) -> np.ndarray:
        fid = self.idx_to_frame_id(idx)
        depth_raw = cv2.imread(
            str(self.depth_dir / f"{fid:04d}_depth.png"),
            cv2.IMREAD_UNCHANGED,
        )
        depth = depth_raw.astype(np.float32) * self.depth_scale_to_meters
        if self.depth_src == "da3":
            depth = depth * self.da3_scale

        return depth

    def get_mask(self, idx: int) -> np.ndarray:
        fid = self.idx_to_frame_id(idx)
        label = cv2.imread(
            str(self.scene_dir / "mask" / f"{fid:04d}_mask.png"),
            cv2.IMREAD_UNCHANGED,
        )
        return label == self.obj_id

    def get_gt_pose(self, idx: int) -> np.ndarray:
        fid = self.idx_to_frame_id(idx)
        fmeta = self.load_frame_meta(fid)
        entry = fmeta["objects"][str(self.obj_id)]
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.array(entry["R"])
        pose[:3, 3] = np.array(entry["t"])
        return pose

    def get_visib_fract(self, idx: int) -> float:
        fid = self.idx_to_frame_id(idx)
        fmeta = self.load_frame_meta(fid)
        return fmeta["objects"][str(self.obj_id)].get("visib_fract", 1.0)

    def get_bbox_visib(self, idx: int) -> list:
        fid = self.idx_to_frame_id(idx)
        fmeta = self.load_frame_meta(fid)
        return fmeta["objects"][str(self.obj_id)].get("bbox_visib", None)

    def has_mesh(self) -> bool:
        return self.obj_info.get("mesh_path") is not None

    def get_gt_mesh(self):
        if not self.has_mesh():
            return None
        if self.init_mesh is None:
            mesh_path = (self.scene_dir / self.obj_info["mesh_path"]).resolve()
            self.init_mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        return self.init_mesh

    def get_gt_mesh_diameter(self):
        return self.obj_info.get("diameter_m")

    def get_bbox_side_len(self):
        return self.obj_info.get("bbox_side_len")

    def get_object_class(self) -> str:
        return self.obj_info.get("class_name", f"obj_{self.obj_id}")

    def get_object_symmetries(self) -> dict:
        return self.obj_info.get("symmetries", {"discrete": [], "continuous": []})

    def is_symmetric(self) -> bool:
        sym = self.get_object_symmetries()
        return bool(sym["discrete"]) or bool(sym["continuous"])

    def get_all_object_ids(self) -> list:
        return [int(k) for k in self.scene_meta["objects"].keys()]

    def __len__(self) -> int:
        return len(self.frame_ids)
