"""
Mask-lift 本体 (Shen et al. ECCV 2024 の論文のアルゴリズムを独自実装): 2D マスク集合 → 各 Gaussian のラベル を閉形式で解く.

論文の核心 (式 2):
    l_i = 1  iff  Σ_{v,p} W_{i,v,p} (2 M_{v,p} - 1)  >  λ Σ_{v,p} W_{i,v,p}

ここで W_{i,v,p} はビュー v のピクセル p への Gaussian i の寄与
(= splatting 時の accumulated alpha = alpha_i * T_i).
λ は background bias (>0 にするとノイズマスクに強くなる).

本実装は完全な differentiable rasterizer は使わず、以下の近似で W を求める:
    1) Gaussian の中心を各ビューに projection
    2) depth で sort し、各 Gaussian の投影半径 r_i をスケールから推定
    3) pixel ごとに near-to-far の Gaussian を集計して alpha 合成し W_{i,v,p} を得る

この近似は論文と等価ではないが、以下の性質を保つので実用では十分機能する:
  - W は非負で、前方ガウスほど寄与が大きい (alpha blending)
  - 同じビューでも M=1 ピクセルに射影された Gaussian ほど「対象らしさ」が高く算出される

torch があれば GPU で動く. 無ければ numpy fallback. 何万 Gaussian × 数十枚
程度までなら GPU 上で数秒〜数十秒で解ける.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .camera_io import Camera, Image, qvec_to_rotmat
from .ply_io import Gaussians


@dataclass
class MaskLiftResult:
    labels: np.ndarray              # (N,) bool, True = 対象
    score: np.ndarray               # (N,) float, 生スコア (decision value)
    w_sum: np.ndarray               # (N,) float, 総寄与 Σ W
    w_mask_sum: np.ndarray          # (N,) float, 対象側への寄与 Σ W * M
    used_views: int


# --------------------------------------------------------------------------
# 低レベル: 1ビューで各 Gaussian の (Σ W, Σ W*M) 寄与を蓄積
# --------------------------------------------------------------------------
def _accumulate_view_numpy(
    xyz: np.ndarray,          # (N, 3)
    alphas: np.ndarray,       # (N,) pre-sigmoid の不透明度を sigmoid したもの
    scales_mean: np.ndarray,  # (N,) 3軸平均スケール (world units)
    camera: Camera,
    image: Image,
    mask: np.ndarray,         # (H, W) uint8
    near: float = 0.01,
    max_gauss_per_tile: int = 8,
    tile: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    単一ビューで各 Gaussian に (w_sum, w_mask_sum) を加算して返す.
    """
    H, W = mask.shape
    K = camera.intrinsics()
    # world → camera
    R = qvec_to_rotmat(image.qvec)
    t = image.tvec.astype(np.float32)
    cam_xyz = (xyz @ R.T) + t[None, :]          # (N, 3), camera frame
    z = cam_xyz[:, 2]
    valid = z > near
    if not np.any(valid):
        return np.zeros(xyz.shape[0], np.float32), np.zeros(xyz.shape[0], np.float32)

    # pinhole projection
    u = (cam_xyz[:, 0] * K[0, 0] / z) + K[0, 2]
    v = (cam_xyz[:, 1] * K[1, 1] / z) + K[1, 2]

    # 投影半径 (pixel 単位). Gaussian の σ (scales_mean) の 3倍を可視半径とする
    # (3σ で 99.7% の重みをカバー). fx と fy 双方を考慮.
    fx = K[0, 0]
    fy = K[1, 1]
    f_mean = 0.5 * (fx + fy)
    radius_px = 3.0 * scales_mean * f_mean / np.clip(z, near, None)

    in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid
    if not np.any(in_frame):
        return np.zeros(xyz.shape[0], np.float32), np.zeros(xyz.shape[0], np.float32)

    # depth-sort (near first). stable sort で同じ z のとき index 順を保ち決定論的に.
    idx = np.where(in_frame)[0]
    order = idx[np.argsort(z[idx], kind="stable")]

    N = xyz.shape[0]
    w_sum = np.zeros(N, np.float32)
    w_mask_sum = np.zeros(N, np.float32)

    # transmittance map. pixel 毎に残透過率を追跡
    T = np.ones((H, W), np.float32)

    # mask の bool 化
    M = (mask > 127).astype(np.float32)

    # 各 Gaussian を近い順に splat
    # O(N * r^2) なので、大きなGaussianが多い/高解像度なら重くなる.
    # ここでは簡易に円形 splat で近似 (論文の2D Gaussian splat の代わり)
    for i in order:
        r = radius_px[i]
        if r < 0.5:
            continue
        cu = u[i]
        cv = v[i]
        ru = int(np.clip(r, 1, 64))
        x0 = max(0, int(cu - ru)); x1 = min(W, int(cu + ru) + 1)
        y0 = max(0, int(cv - ru)); y1 = min(H, int(cv + ru) + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        # 2D Gaussian 重み (等方近似)
        ys = np.arange(y0, y1, dtype=np.float32)[:, None] - cv
        xs = np.arange(x0, x1, dtype=np.float32)[None, :] - cu
        sigma = max(r / 2.0, 1.0)
        gauss = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
        # 中心で alpha, 外側で 0 に減衰
        a = alphas[i] * gauss
        a = np.clip(a, 0.0, 0.99)

        t_patch = T[y0:y1, x0:x1]
        w_patch = a * t_patch             # 貢献度 W_{i,v,p}
        m_patch = M[y0:y1, x0:x1]

        w_sum[i] += float(w_patch.sum())
        w_mask_sum[i] += float((w_patch * m_patch).sum())

        # 透過率を減らす
        T[y0:y1, x0:x1] = t_patch * (1.0 - a)
        # 透過率がほぼゼロの領域はこれ以降ほぼ0寄与 → 打ち切りはしない(スキップは全体に効く)

    return w_sum, w_mask_sum


# --------------------------------------------------------------------------
# torch 加速版 (存在すれば使う)
# --------------------------------------------------------------------------
def _accumulate_view_torch(
    xyz, alphas, scales_mean, camera, image, mask, device="cuda", near=0.01,
):
    """
    ピクセル空間の accumulation を GPU で並列化. 計算は numpy 版と同じ.
    N が大きい場合はこちらを推奨.
    """
    import torch

    H, W = mask.shape
    K = torch.from_numpy(camera.intrinsics()).to(device)
    R = torch.from_numpy(qvec_to_rotmat(image.qvec)).to(device)
    t = torch.from_numpy(image.tvec).to(device)

    xyz_t = torch.from_numpy(xyz).to(device)
    cam_xyz = xyz_t @ R.T + t
    z = cam_xyz[:, 2]
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]

    u = cam_xyz[:, 0] * fx / z.clamp(min=near) + cx
    v = cam_xyz[:, 1] * fy / z.clamp(min=near) + cy
    # 3σ 半径. fx と fy の平均で単一スケーリング (等方近似)
    f_mean = 0.5 * (fx + fy)
    radius_px = 3.0 * torch.from_numpy(scales_mean).to(device) * f_mean / z.clamp(min=near)

    in_frame = (z > near) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (radius_px >= 0.5)
    order_idx = torch.nonzero(in_frame, as_tuple=False).squeeze(-1)
    if order_idx.numel() == 0:
        Nall = xyz.shape[0]
        return np.zeros(Nall, np.float32), np.zeros(Nall, np.float32)
    order_idx = order_idx[torch.argsort(z[order_idx], stable=True)]

    alphas_t = torch.from_numpy(alphas).to(device)
    M_t = (torch.from_numpy(mask).to(device) > 127).float()
    T = torch.ones((H, W), device=device)

    N = xyz.shape[0]
    w_sum = torch.zeros(N, device=device)
    w_mask_sum = torch.zeros(N, device=device)

    # ループ自体は Python だが、各 Gaussian のパッチ更新はベクトル化
    for i in order_idx.tolist():
        r = float(radius_px[i].item())
        ru = int(min(max(r, 1.0), 64.0))
        cu = float(u[i].item()); cvv = float(v[i].item())
        x0 = max(0, int(cu - ru)); x1 = min(W, int(cu + ru) + 1)
        y0 = max(0, int(cvv - ru)); y1 = min(H, int(cvv + ru) + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        ys = torch.arange(y0, y1, device=device, dtype=torch.float32).unsqueeze(1) - cvv
        xs = torch.arange(x0, x1, device=device, dtype=torch.float32).unsqueeze(0) - cu
        sigma = max(r / 2.0, 1.0)
        gauss = torch.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
        a = (alphas_t[i] * gauss).clamp(0.0, 0.99)

        t_patch = T[y0:y1, x0:x1]
        w = a * t_patch
        m = M_t[y0:y1, x0:x1]
        w_sum[i] = w_sum[i] + w.sum()
        w_mask_sum[i] = w_mask_sum[i] + (w * m).sum()
        T[y0:y1, x0:x1] = t_patch * (1.0 - a)

    return w_sum.cpu().numpy(), w_mask_sum.cpu().numpy()


# --------------------------------------------------------------------------
# トップレベル: マスク集合をリフト
# --------------------------------------------------------------------------
def lift_masks_to_gaussians(
    gaussians: Gaussians,
    cameras: Dict[int, Camera],
    images: Dict[int, Image],
    mask_loader,                    # name -> uint8 (H, W) mask
    bg_bias: float = 0.2,           # λ: 大きいほど対象判定が厳しくなる
    device: str = "auto",
    progress: Optional[callable] = None,
    max_views: Optional[int] = None,
) -> MaskLiftResult:
    """
    mask_loader(image_name) で画像名から 2D マスクを返す関数を受け取り、
    各 Gaussian のラベルを閉形式で計算する.

    bg_bias (= λ):
      論文式 2 の閾値. 0 だと「マスク内寄与 > マスク外寄与」で 1、
      大きくするほどマスク噪声に堅牢になる (例: 0.2 が既定, 0.3〜0.5 で厳しめ).
    """
    # device 解決
    use_torch = False
    if device == "auto":
        try:
            import torch
            use_torch = torch.cuda.is_available()
            device = "cuda" if use_torch else "cpu"
        except ImportError:
            use_torch = False
            device = "cpu"
    elif device in ("cuda", "mps"):
        import torch
        use_torch = True

    N = gaussians.n
    alphas = gaussians.opacity_sigmoid().astype(np.float32)
    # scales はログスペース. 3軸平均の exp
    scales_mean = np.exp(gaussians.scales).mean(axis=1).astype(np.float32)
    xyz = gaussians.xyz

    total_w = np.zeros(N, np.float32)
    total_wm = np.zeros(N, np.float32)

    # image_id 順に処理
    items = list(images.items())
    if max_views is not None:
        items = items[:max_views]
    total = len(items)

    used = 0
    for step, (img_id, img) in enumerate(items):
        mask = mask_loader(img.name)
        if mask is None:
            continue
        cam = cameras[img.camera_id]
        # mask の解像度が画像と違う場合は合わせる
        if mask.shape != (cam.height, cam.width):
            import cv2
            mask = cv2.resize(mask, (cam.width, cam.height),
                              interpolation=cv2.INTER_NEAREST)

        if use_torch:
            w, wm = _accumulate_view_torch(
                xyz, alphas, scales_mean, cam, img, mask, device=device,
            )
        else:
            w, wm = _accumulate_view_numpy(
                xyz, alphas, scales_mean, cam, img, mask,
            )
        total_w += w
        total_wm += wm
        used += 1
        if progress:
            progress(step + 1, total, img.name)

    # 論文式 2: Σ W (2M - 1) > λ Σ W  ⇔  2 Σ W*M - Σ W > λ Σ W
    score = 2.0 * total_wm - total_w - bg_bias * total_w
    labels = score > 0

    return MaskLiftResult(
        labels=labels,
        score=score,
        w_sum=total_w,
        w_mask_sum=total_wm,
        used_views=used,
    )
