"""
ERP ⇄ perspective (tangent plane) 射影ユーティリティ.

参照座標系:
  - ERP 画像:  x = [0, W) → longitude φ = (x/W * 2π - π), 範囲 [-π, π)
               y = [0, H) → latitude   θ = (0.5 - y/H) * π, 範囲 [π/2, -π/2]
  - 3D ベクトル:  (X, Y, Z) = (cos θ cos φ, sin θ, cos θ sin φ)
    すなわち +Y が真上 (north pole), +Z が前方 φ=0, +X が φ=π/2 (右手)
  - perspective 画像はピンホール, yaw (水平角), pitch (仰俯角), roll=0 で定義

実装は OpenCV の remap (双線形) だけで完結させる. GPU も py360convert も不要.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class ErpSpec:
    """ERP 画像の定義 (解像度のみ)."""
    width: int
    height: int

    @classmethod
    def from_image(cls, img: np.ndarray) -> "ErpSpec":
        return cls(img.shape[1], img.shape[0])


@dataclass
class ViewSpec:
    """tangent perspective view の向き・画角・出力解像度."""
    yaw_deg: float
    pitch_deg: float
    fov_deg: float      # 水平 FOV
    width: int
    height: int

    def label(self) -> str:
        return f"yaw{int(self.yaw_deg):+04d}_pitch{int(self.pitch_deg):+03d}"


# --------------------------------------------------------------------------
# 基本演算
# --------------------------------------------------------------------------
def _rotation_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """world → view の回転行列. yaw=水平, pitch=仰俯 (+は上向き)."""
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    # yaw は +Y 軸回り (world Y = up). pitch は view 側の +X 軸回り.
    R_yaw = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]], dtype=np.float32)
    R_pitch = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]], dtype=np.float32)
    return R_pitch @ R_yaw


def _perspective_to_rays(W: int, H: int, fov_deg: float) -> np.ndarray:
    """
    perspective 画像の各ピクセルに対応する view-space 方向ベクトル (H, W, 3) を返す.
    +Z 前方, +X 右, +Y 下.
    """
    f = (W / 2) / np.tan(np.deg2rad(fov_deg) / 2)
    xs = (np.arange(W, dtype=np.float32) - W / 2) / f
    ys = (np.arange(H, dtype=np.float32) - H / 2) / f
    gx, gy = np.meshgrid(xs, ys)
    # Y 軸は画像下向きにしたいので +sign
    dirs = np.stack([gx, gy, np.ones_like(gx)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs  # (H, W, 3)


# --------------------------------------------------------------------------
# ERP → perspective
# --------------------------------------------------------------------------
def erp_to_perspective(
    erp: np.ndarray,
    view: ViewSpec,
    interp: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    ERP 画像を指定 yaw/pitch/fov で perspective 切り出しする.
    erp: (H, W, C) or (H, W) uint8/float
    """
    H_erp, W_erp = erp.shape[:2]
    dirs_view = _perspective_to_rays(view.width, view.height, view.fov_deg)
    R = _rotation_from_yaw_pitch(view.yaw_deg, view.pitch_deg)
    # world → view の R なので、view → world は R.T
    dirs_world = dirs_view @ R  # (H, W, 3)

    X, Y, Z = dirs_world[..., 0], dirs_world[..., 1], dirs_world[..., 2]
    # 注意: 我々のERP定義は +Z=φ0, +X=φ+π/2, +Y=上  (座標系は _perspective_to_rays と一致)
    # longitude φ, latitude θ
    phi = np.arctan2(X, Z)                                # [-π, π]
    theta = np.arcsin(np.clip(-Y, -1, 1))                 # [-π/2, π/2]

    u = (phi + np.pi) / (2 * np.pi) * W_erp               # [0, W)
    v = (np.pi / 2 - theta) / np.pi * H_erp               # [0, H)

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    # OpenCV の remap は境界外を黒埋めするので, longitude を wrap するために BORDER_WRAP
    return cv2.remap(erp, map_x, map_y, interpolation=interp,
                     borderMode=cv2.BORDER_WRAP)


# --------------------------------------------------------------------------
# perspective mask → ERP (マスク合成)
# --------------------------------------------------------------------------
def perspective_mask_to_erp(
    persp_mask: np.ndarray,
    view: ViewSpec,
    erp_spec: ErpSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    perspective 空間の mask (H_p, W_p) を ERP 空間へ 逆射影 する.
    戻り値: (erp_mask_float, erp_weight)
      - erp_mask_float: そのビューにおけるマスク値 (0..1). 視野外は 0.
      - erp_weight:     そのビューがカバーする ERP ピクセルの重み (0..1).
                        視野中心で 1, 境界で減衰するソフトウィンドウ.
    複数ビューを重ね合わせる際に weight で加重平均する.
    """
    W_erp, H_erp = erp_spec.width, erp_spec.height

    # ERP の各ピクセルを 3D 方向ベクトルに展開
    xs = np.arange(W_erp, dtype=np.float32)
    ys = np.arange(H_erp, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    phi = (gx / W_erp) * (2 * np.pi) - np.pi
    theta = (0.5 - gy / H_erp) * np.pi
    X = np.cos(theta) * np.sin(phi)
    Y = -np.sin(theta)
    Z = np.cos(theta) * np.cos(phi)
    dirs_world = np.stack([X, Y, Z], axis=-1)

    R = _rotation_from_yaw_pitch(view.yaw_deg, view.pitch_deg)
    # world → view
    dirs_view = dirs_world @ R.T
    zc = dirs_view[..., 2]

    # カメラ前方 (Z > 0) のみ有効
    valid = zc > 1e-3
    f = (view.width / 2) / np.tan(np.deg2rad(view.fov_deg) / 2)
    u = dirs_view[..., 0] * f / np.clip(zc, 1e-3, None) + view.width / 2
    v = dirs_view[..., 1] * f / np.clip(zc, 1e-3, None) + view.height / 2
    in_img = (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height) & valid

    # Remap でマスクをサンプリング
    map_x = np.where(in_img, u, -1).astype(np.float32)
    map_y = np.where(in_img, v, -1).astype(np.float32)
    sampled = cv2.remap(
        persp_mask.astype(np.float32),
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # weight: 視野中央で 1, 境界でソフトに減衰. ただし視野内は必ず > 0
    # (cubemap のような直交分割で隅が 0 になると被覆漏れするため)
    du = (u - view.width / 2) / (view.width / 2)
    dv = (v - view.height / 2) / (view.height / 2)
    r2 = du * du + dv * dv
    # 端でも 0.1 は残す (範囲 [0.1, 1.0])
    soft = 0.1 + 0.9 * np.clip(1.0 - 0.5 * r2, 0, 1)
    weight = np.where(in_img, soft, 0.0).astype(np.float32)

    if persp_mask.dtype == np.uint8:
        sampled = sampled / 255.0

    return sampled.astype(np.float32), weight


# --------------------------------------------------------------------------
# 視野プリセット
# --------------------------------------------------------------------------
def cubemap_views(face_size: int, fov_deg: float = 90.0) -> List[ViewSpec]:
    """標準 6 面 cubemap. 各面は正方形の perspective 画像."""
    configs = [
        ("front",  0,    0),
        ("right",  90,   0),
        ("back",   180,  0),
        ("left",   -90,  0),
        ("top",    0,    90),
        ("bottom", 0,    -90),
    ]
    return [ViewSpec(yaw, pitch, fov_deg, face_size, face_size)
            for (_, yaw, pitch) in configs]


def tangent_views(
    n_yaw: int = 8,
    n_pitch: int = 3,
    fov_deg: float = 90.0,
    width: int = 1024,
    height: int = 1024,
) -> List[ViewSpec]:
    """
    タンジェントプレーン視野. yaw を n_yaw 等分, pitch を n_pitch 段.
    既定: yaw=8 × pitch=3 (-45°, 0°, +45°) = 24 ビュー.
    隣接ビュー間でオーバーラップしてほしいので FOV を 90° 推奨.
    """
    yaws = np.linspace(-180, 180, n_yaw, endpoint=False).tolist()
    if n_pitch == 1:
        pitches = [0.0]
    else:
        pitches = np.linspace(-60, 60, n_pitch).tolist()

    views = []
    for p in pitches:
        for y in yaws:
            views.append(ViewSpec(float(y), float(p), fov_deg, width, height))
    return views
