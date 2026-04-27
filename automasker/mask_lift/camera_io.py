"""
COLMAP sparse モデル (cameras.bin / images.bin) の最小パーサ.

完全な pycolmap に依存しないよう、必要な要素だけを読む.
前提: PINHOLE or SIMPLE_PINHOLE カメラモデル. 魚眼や歪みは projection 時に無視される.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------------
# バイナリ読み込みヘルパ
# --------------------------------------------------------------------------
def _read(fid, n: int, fmt: str):
    data = fid.read(n)
    return struct.unpack("<" + fmt, data)


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # [fx, fy, cx, cy] などカメラモデル依存

    def intrinsics(self) -> np.ndarray:
        """3x3 の K 行列を返す. PINHOLE 以外では近似."""
        m = self.model
        p = self.params
        if m == "SIMPLE_PINHOLE":
            fx = fy = p[0]; cx, cy = p[1], p[2]
        elif m in ("PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"):
            fx = p[0]; fy = p[1] if m in ("PINHOLE", "OPENCV", "FULL_OPENCV") else p[0]
            if m == "PINHOLE":
                cx, cy = p[2], p[3]
            else:
                cx, cy = p[1], p[2] if m == "SIMPLE_RADIAL" else (p[2], p[3])
        else:
            fx = fy = p[0]; cx, cy = self.width / 2, self.height / 2

        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


@dataclass
class Image:
    id: int
    qvec: np.ndarray   # (4,) quaternion (w, x, y, z) -- world_to_camera
    tvec: np.ndarray   # (3,)
    camera_id: int
    name: str

    def world_to_cam(self) -> np.ndarray:
        """4x4 変換行列 (world → camera)."""
        R = qvec_to_rotmat(self.qvec)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = self.tvec
        return T


# --------------------------------------------------------------------------
CAMERA_MODEL_IDS = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
    4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV", 7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE",
}
CAMERA_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4, "RADIAL": 5,
    "OPENCV": 8, "OPENCV_FISHEYE": 8, "FULL_OPENCV": 12, "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4, "RADIAL_FISHEYE": 5, "THIN_PRISM_FISHEYE": 12,
}


def read_cameras_bin(path: Path) -> Dict[int, Camera]:
    cams: Dict[int, Camera] = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            cid = _read(f, 4, "i")[0]
            model_id = _read(f, 4, "i")[0]
            w, h = _read(f, 16, "QQ")
            model = CAMERA_MODEL_IDS[model_id]
            nparam = CAMERA_MODEL_NUM_PARAMS[model]
            params = np.array(_read(f, 8 * nparam, "d" * nparam), dtype=np.float32)
            cams[cid] = Camera(cid, model, w, h, params)
    return cams


def read_images_bin(path: Path) -> Dict[int, Image]:
    imgs: Dict[int, Image] = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            img_id = _read(f, 4, "i")[0]
            q = np.array(_read(f, 32, "dddd"), dtype=np.float32)
            t = np.array(_read(f, 24, "ddd"), dtype=np.float32)
            cam_id = _read(f, 4, "i")[0]
            # ヌル終端文字列
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            # point2D: skip
            num_pt = _read(f, 8, "Q")[0]
            f.read(24 * num_pt)
            imgs[img_id] = Image(img_id, q, t, cam_id, name.decode())
    return imgs


# --------------------------------------------------------------------------
def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) → 3x3 rotation."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)


def load_scene(scene_dir: Path) -> Tuple[Dict[int, Camera], Dict[int, Image]]:
    """
    scene/sparse/0/ に cameras.bin/images.bin が格納されている前提.
    gaussian-splatting の公式スクリプトが出力する構成.
    """
    sparse = scene_dir / "sparse" / "0"
    if not sparse.exists():
        sparse = scene_dir / "sparse"
    cams = read_cameras_bin(sparse / "cameras.bin")
    imgs = read_images_bin(sparse / "images.bin")
    return cams, imgs
