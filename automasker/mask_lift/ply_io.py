"""
3D Gaussian Splatting の point_cloud.ply 読み書き.

inria gaussian-splatting の慣習に従い、以下のプロパティを期待する:
    x, y, z                       : 中心位置
    nx, ny, nz                    : (使われない, 0)
    f_dc_0 .. f_dc_2              : SH 0 次 (色)
    f_rest_0 .. f_rest_{K-1}      : SH 高次
    opacity                       : pre-sigmoid の不透明度
    scale_0, scale_1, scale_2     : pre-exp のスケール (log-space)
    rot_0 .. rot_3                : 四元数 (w, x, y, z)

plyfile 依存で読み書きする. numpy 配列のタプルを返す.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class Gaussians:
    xyz: np.ndarray         # (N, 3)
    opacity: np.ndarray     # (N, 1) pre-sigmoid
    scales: np.ndarray      # (N, 3) pre-exp (log)
    rot: np.ndarray         # (N, 4) (w, x, y, z)
    features_dc: np.ndarray # (N, 3)
    features_rest: np.ndarray  # (N, K)
    # 元の PlyElement を保持して書き戻しに使う
    _raw_names: list = None  # type: ignore

    @property
    def n(self) -> int:
        return self.xyz.shape[0]

    def opacity_sigmoid(self) -> np.ndarray:
        """α (0..1) に変換された不透明度."""
        return 1.0 / (1.0 + np.exp(-self.opacity.squeeze(-1)))

    def scales_exp(self) -> np.ndarray:
        return np.exp(self.scales)


def load_ply(path: Path, max_size_mb: int = 4096) -> Gaussians:
    """
    3DGS の .ply を読み込む.

    SECURITY: 最大サイズを 4 GB に制限し、巨大ファイルによるメモリ枯渇を防ぐ.
    通常の 3DGS scene は 1-2 GB に収まるので十分な上限.
    信頼できるソースのみ読み込んでください (plyfile 自体は安全だが、
    本ツールは入力 .ply を検証しないため).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size > max_size_mb * 1024 * 1024:
        raise ValueError(
            f"PLY が大きすぎます ({size/1e9:.2f} GB > {max_size_mb/1024:.1f} GB). "
            f"意図した入力なら load_ply(path, max_size_mb=...) で上限を引き上げてください."
        )

    from plyfile import PlyData

    plydata = PlyData.read(str(path))
    v = plydata["vertex"]
    names = v.data.dtype.names

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    f_dc = np.stack(
        [v[f"f_dc_{i}"] for i in range(3)], axis=-1
    ).astype(np.float32)

    rest_names = sorted(
        [n for n in names if n.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    f_rest = (
        np.stack([v[n] for n in rest_names], axis=-1).astype(np.float32)
        if rest_names else np.zeros((xyz.shape[0], 0), np.float32)
    )

    opacity = np.asarray(v["opacity"], dtype=np.float32)[:, None]
    scales = np.stack(
        [v[f"scale_{i}"] for i in range(3)], axis=-1
    ).astype(np.float32)
    rot = np.stack(
        [v[f"rot_{i}"] for i in range(4)], axis=-1
    ).astype(np.float32)

    return Gaussians(
        xyz=xyz, opacity=opacity, scales=scales, rot=rot,
        features_dc=f_dc, features_rest=f_rest,
        _raw_names=list(names),
    )


def save_ply(path: Path, g: Gaussians, mask: np.ndarray | None = None) -> None:
    """
    mask が与えられた場合、True のガウスだけを書き出す (=オブジェクト抽出).
    """
    from plyfile import PlyData, PlyElement

    if mask is None:
        mask = np.ones(g.n, dtype=bool)
    mask = mask.astype(bool)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        *[(f"f_dc_{i}", "f4") for i in range(3)],
        *[(f"f_rest_{i}", "f4") for i in range(g.features_rest.shape[1])],
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    N = int(mask.sum())
    arr = np.zeros(N, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = g.xyz[mask].T
    # nx/ny/nz は 0
    for i in range(3):
        arr[f"f_dc_{i}"] = g.features_dc[mask, i]
    for i in range(g.features_rest.shape[1]):
        arr[f"f_rest_{i}"] = g.features_rest[mask, i]
    arr["opacity"] = g.opacity[mask, 0]
    for i in range(3):
        arr[f"scale_{i}"] = g.scales[mask, i]
    for i in range(4):
        arr[f"rot_{i}"] = g.rot[mask, i]

    el = PlyElement.describe(arr, "vertex")
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el], text=False).write(str(path))
