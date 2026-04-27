"""
gsplat を用いた mask-lift の W (Shen et al. 論文のアイデア) 計算 (厳密版).

lift.py の既定実装は等方 circular splat による近似. 本モジュールは
nerfstudio-project/gsplat の differentiable rasterizer を用いて、
anisotropic 3D Gaussian splatting 過程そのものを使って W を計算する.

核心:
    - gsplat.rasterization は alpha-compose された RGB/depth を返すだけでなく、
      "各 Gaussian の per-pixel 寄与" を直接は返さない.
    - そこで、各 Gaussian の "色" を one-hot indicator に設定し、複数チャネルで
      レンダリングすることで W を取り出す.

⚠️  精度上の注意 (重要):

本実装は Gaussian を batch_size 個ずつのサブセットに分けて rasterize しますが、
これは **バッチ内のガウスのみで alpha 合成** する近似です. 論文の厳密な
FlashSplat W は「全ガウスを同時に rasterize したときの各ガウスの寄与」を
要求するため、バッチを跨いだ occlusion (手前のバッチのガウスが後ろのバッチの
ガウスを遮蔽する効果) は失われます.

実用上の影響:
    - 単純なシーン (1〜2 物体が分離) ではほぼ問題なし.
    - 密な遮蔽があるシーン (屋内、街並み等) では W が過大評価され、
      false positive が増える可能性がある.

対策:
    - batch_size を十分大きく (理想は全ガウス数) とり、1バッチで済ませる
      (メモリが許す場合は論文と等価).
    - あるいは近似版 (lift.py, 円形 splat) を使う. こちらも別の誤差がある.

Future work: 各バッチに "他の全ガウスを灰色背景として同時に渡す" 実装を書くと
正しい W が得られますが、未実装.

gsplat が無い環境ではこのファイルを import しても失敗するだけなので、
使用側で try/except してフォールバックする.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .camera_io import Camera, Image, qvec_to_rotmat
from .lift import MaskLiftResult
from .ply_io import Gaussians


def _require_torch_and_gsplat():
    import torch
    try:
        from gsplat.rendering import rasterization
    except ImportError as e:
        raise RuntimeError(
            "gsplat が見つかりません. pip install gsplat でインストールしてください."
        ) from e
    return torch, rasterization


def _gaussians_to_tensors(g: Gaussians, device):
    """PLY パラメータを gsplat が期待する Tensor 形式に変換."""
    import torch

    means = torch.from_numpy(g.xyz).to(device).float()
    # scales はログスペースなので exp する
    scales = torch.from_numpy(np.exp(g.scales)).to(device).float()
    # rot は (w, x, y, z). gsplat も (w, x, y, z) を受け取る
    quats = torch.from_numpy(g.rot).to(device).float()
    # normalize (学習時は自動正規化されているが安全のため)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    opacities = torch.from_numpy(g.opacity_sigmoid()).to(device).float()  # (N,)
    return means, quats, scales, opacities


def _make_viewmat(camera: Camera, image: Image, device) -> "torch.Tensor":
    import torch
    R = torch.from_numpy(qvec_to_rotmat(image.qvec)).to(device).float()
    t = torch.from_numpy(image.tvec).to(device).float()
    view = torch.eye(4, device=device)
    view[:3, :3] = R
    view[:3, 3] = t
    return view


def _make_K(camera: Camera, device) -> "torch.Tensor":
    import torch
    return torch.from_numpy(camera.intrinsics()).to(device).float()


# --------------------------------------------------------------------------
def _accumulate_view_gsplat(
    means, quats, scales, opacities,
    camera: Camera, image: Image, mask_np: np.ndarray,
    device, batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1 ビューで、全 Gaussian の (w_sum_i, w_mask_sum_i) を gsplat 経由で取得する.

    戦略: Gaussian を batch_size 個ずつのグループに分けて、色 (channel ==group_idx)
    に indicator を入れてレンダリングする. 結果チャネル c の各ピクセル値 =
    グループ c に属する Gaussian の累積 W_{i, p}. それを mask で内積するだけで、
    グループ内総和として w_sum / w_mask_sum が得られる. 個別 i に分解するには
    batch 内で 1 個ずつ順に one-hot を立てることになり重いので、本実装では
    batch 内の Gaussian については等分して寄与を割り当てる近似は使わず、
    "1 Gaussian = 1 channel" で厳密にやる. チャネル数は O(N) と巨大になるので
    ミニバッチでループする.
    """
    import torch
    from gsplat.rendering import rasterization

    H, W = mask_np.shape
    K = _make_K(camera, device).unsqueeze(0)          # (1, 3, 3)
    viewmat = _make_viewmat(camera, image, device).unsqueeze(0)  # (1, 4, 4)
    mask_t = torch.from_numpy(mask_np).to(device).float()
    mask_t = (mask_t > 127).float()

    N = means.shape[0]
    w_sum = torch.zeros(N, device=device)
    w_mask_sum = torch.zeros(N, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        b = end - start
        # カラー Tensor: shape (N, 3) のうち、この batch だけ one-hot-ish に
        # …ではメモリが爆発するので、代わりに "batch メンバーだけレンダリング" する.
        # gsplat は「レンダリングに使うガウスの subset」を直接渡せるので、
        # そのぶん軽量. ただしチャネルは "batch 内のガウスを単位ベクトル" にする.
        sub_means = means[start:end]
        sub_quats = quats[start:end]
        sub_scales = scales[start:end]
        sub_op = opacities[start:end]

        # 各 Gaussian を独立チャネルに割り当てる (b チャネル)
        colors = torch.eye(b, device=device).unsqueeze(0)  # (1, b, b) ... 不正
        # 正しくは colors shape (N_subset, b) — gsplat は (N, C) を期待
        colors = torch.eye(b, device=device)  # (b, b), 行 i = Gaussian i の色

        # b チャネルで rasterize
        # gsplat.rasterization signature (v1.5.x):
        # rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height,
        #               near_plane=..., far_plane=..., render_mode="RGB", ...)
        # -> render_colors (1, H, W, C), render_alphas (1, H, W, 1), meta
        render_colors, render_alphas, _ = rasterization(
            means=sub_means,
            quats=sub_quats,
            scales=sub_scales,
            opacities=sub_op,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            render_mode="RGB",
        )
        # render_colors: (1, H, W, b)
        out = render_colors[0]  # (H, W, b)

        # チャネル c ごとに sum(out[:,:,c]) = Σ_p W_{c, p}, と sum(out * mask) = Σ W*M
        w_sum[start:end] = out.sum(dim=(0, 1))
        w_mask_sum[start:end] = (out * mask_t.unsqueeze(-1)).sum(dim=(0, 1))

    return w_sum.detach().cpu().numpy(), w_mask_sum.detach().cpu().numpy()


# --------------------------------------------------------------------------
def lift_masks_to_gaussians_gsplat(
    gaussians: Gaussians,
    cameras: Dict[int, Camera],
    images: Dict[int, Image],
    mask_loader,
    bg_bias: float = 0.2,
    batch_size: int = 2048,
    progress=None,
    max_views: Optional[int] = None,
    device: str = "cuda",
) -> MaskLiftResult:
    """
    gsplat (anisotropic, differentiable rasterizer) を用いて mask-lift を解く.

    lift.py の lift_masks_to_gaussians と API 互換. 計算が重くなる分精度は高い.
    """
    torch, _ = _require_torch_and_gsplat()

    means, quats, scales, opacities = _gaussians_to_tensors(gaussians, device)
    N = means.shape[0]

    total_w = torch.zeros(N, device=device)
    total_wm = torch.zeros(N, device=device)

    items = list(images.items())
    if max_views is not None:
        items = items[:max_views]

    used = 0
    import cv2
    for step, (_, img) in enumerate(items):
        mask = mask_loader(img.name)
        if mask is None:
            continue
        cam = cameras[img.camera_id]
        if mask.shape != (cam.height, cam.width):
            mask = cv2.resize(mask, (cam.width, cam.height),
                              interpolation=cv2.INTER_NEAREST)

        w, wm = _accumulate_view_gsplat(
            means, quats, scales, opacities,
            cam, img, mask, device=device, batch_size=batch_size,
        )
        total_w += torch.from_numpy(w).to(device)
        total_wm += torch.from_numpy(wm).to(device)
        used += 1
        if progress:
            progress(step + 1, len(items), img.name)

    # 論文式 2
    score = 2.0 * total_wm - total_w - bg_bias * total_w
    labels = (score > 0).cpu().numpy()

    return MaskLiftResult(
        labels=labels,
        score=score.cpu().numpy(),
        w_sum=total_w.cpu().numpy(),
        w_mask_sum=total_wm.cpu().numpy(),
        used_views=used,
    )
