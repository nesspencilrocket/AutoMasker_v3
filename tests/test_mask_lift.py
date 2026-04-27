"""
mask-lift の非 I/O ロジック (camera projection, closed-form solver) の検証.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automasker.mask_lift.camera_io import Camera, Image, qvec_to_rotmat
from automasker.mask_lift.lift import _accumulate_view_numpy, lift_masks_to_gaussians
from automasker.mask_lift.ply_io import Gaussians


def _make_synthetic_scene(seed=0):
    """
    2つの Gaussian クラスタを用意する:
      - A: (0,0,5) 付近 → 画像中央に映る対象
      - B: (3,0,5) 付近 → 右側に映る非対象
    """
    rng = np.random.default_rng(seed)
    A = np.array([0.0, 0.0, 5.0]) + rng.normal(scale=0.05, size=(50, 3))
    B = np.array([3.0, 0.0, 5.0]) + rng.normal(scale=0.05, size=(50, 3))
    xyz = np.concatenate([A, B]).astype(np.float32)
    N = xyz.shape[0]

    g = Gaussians(
        xyz=xyz,
        opacity=np.full((N, 1), 3.0, np.float32),           # sigmoid(3) ≈ 0.95
        scales=np.full((N, 3), np.log(0.05), np.float32),   # 約 5cm 相当
        rot=np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
        features_dc=np.zeros((N, 3), np.float32),
        features_rest=np.zeros((N, 0), np.float32),
    )
    return g, N, 50  # Gaussians, 全体 N, 前半 = 対象 50 個


def _make_camera(W=320, H=240, f=400.0):
    cam = Camera(
        id=1, model="PINHOLE", width=W, height=H,
        params=np.array([f, f, W / 2, H / 2], np.float32),
    )
    # world → camera の identity: tvec=0, qvec=(1,0,0,0)
    img = Image(
        id=1, qvec=np.array([1, 0, 0, 0], np.float32),
        tvec=np.zeros(3, np.float32),
        camera_id=1, name="00001.jpg",
    )
    return cam, img


def test_qvec_to_rotmat_identity():
    R = qvec_to_rotmat(np.array([1, 0, 0, 0], np.float32))
    assert np.allclose(R, np.eye(3), atol=1e-6)
    print("  ✔ qvec_to_rotmat(identity) OK")


def test_projection_center():
    """cluster A が画像中央に落ちることを単体で確認."""
    g, N, N_target = _make_synthetic_scene()
    cam, img = _make_camera()

    # 対象 cluster A の中心を画像に落とす: (0,0,5) → u=W/2, v=H/2
    K = cam.intrinsics()
    p_cam = np.array([0, 0, 5], dtype=np.float32)
    u = p_cam[0] * K[0, 0] / p_cam[2] + K[0, 2]
    v = p_cam[1] * K[1, 1] / p_cam[2] + K[1, 2]
    assert abs(u - cam.width / 2) < 1 and abs(v - cam.height / 2) < 1
    print("  ✔ pinhole projection maps (0,0,5) to image center OK")


def test_accumulate_puts_weight_on_masked_pixels():
    g, N, N_target = _make_synthetic_scene()
    cam, img = _make_camera()
    # 左半分のみマスク=1 にする → 対象 (cluster A, 中央に投影) が主に対象側にヒット
    mask = np.zeros((cam.height, cam.width), np.uint8)
    mask[:, : cam.width // 2 + 20] = 255  # 中央より少し右まで含む

    alphas = g.opacity_sigmoid()
    scales_mean = np.exp(g.scales).mean(axis=1)
    w, wm = _accumulate_view_numpy(g.xyz, alphas, scales_mean, cam, img, mask)

    A_mask_ratio = wm[:N_target].sum() / max(w[:N_target].sum(), 1e-8)
    B_mask_ratio = wm[N_target:].sum() / max(w[N_target:].sum(), 1e-8)
    # A は中央 (マスク内) に投影されるので mask_ratio が高いはず
    assert A_mask_ratio > 0.5, f"A should be mostly in mask, got {A_mask_ratio:.3f}"
    # B は右側 (マスク外) に投影されるので低いはず
    assert B_mask_ratio < 0.5, f"B should be mostly outside mask, got {B_mask_ratio:.3f}"
    print(f"  ✔ accumulate: A_in_mask={A_mask_ratio:.2f}  B_in_mask={B_mask_ratio:.2f}")


def test_lift_solves_labels_correctly():
    g, N, N_target = _make_synthetic_scene()
    cam, img = _make_camera()
    mask = np.zeros((cam.height, cam.width), np.uint8)
    mask[:, : cam.width // 2 + 20] = 255  # 中央寄り左半分

    cameras = {1: cam}
    images = {1: img}

    def loader(name):
        return mask

    # λ=0.0 で明確に分けられるはず (A=対象, B=非対象)
    result = lift_masks_to_gaussians(
        g, cameras, images, loader, bg_bias=0.0, device="cpu",
    )
    n_A_true = int(result.labels[:N_target].sum())
    n_B_true = int(result.labels[N_target:].sum())
    # A の大多数が True、B の大多数が False であること
    assert n_A_true > 0.7 * N_target, f"A recall too low: {n_A_true}/{N_target}"
    assert n_B_true < 0.3 * (N - N_target), f"B FP too high: {n_B_true}/{N - N_target}"
    print(f"  ✔ lift labels: A true={n_A_true}/{N_target}  B true={n_B_true}/{N-N_target}")


def main():
    print("mask-lift tests")
    test_qvec_to_rotmat_identity()
    test_projection_center()
    test_accumulate_puts_weight_on_masked_pixels()
    test_lift_solves_labels_correctly()
    print("\nAll mask-lift tests passed.")


if __name__ == "__main__":
    main()
