"""
ERP ⇄ perspective の幾何的性質をチェックするテスト.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automasker.pano.projection import (
    ErpSpec,
    ViewSpec,
    cubemap_views,
    erp_to_perspective,
    perspective_mask_to_erp,
    tangent_views,
)


def _make_test_erp(W=1024, H=512):
    """
    ERP 画像として、左右と上下でグラデーションが違うパターンを作る.
    "正面" 方向 (yaw=0) に十字マークを描く.
    """
    img = np.zeros((H, W, 3), np.uint8)
    # 水平グラデーション
    img[:, :, 0] = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
    # 垂直グラデーション
    img[:, :, 1] = np.tile(np.linspace(255, 0, H, dtype=np.uint8)[:, None], (1, W))
    # 正面 (x = W/2) に白十字
    cv2.line(img, (W // 2, 0), (W // 2, H - 1), (255, 255, 255), 4)
    cv2.line(img, (0, H // 2), (W - 1, H // 2), (255, 255, 255), 4)
    return img


def test_perspective_center_hits_same_erp_center():
    erp = _make_test_erp()
    H, W = erp.shape[:2]
    view = ViewSpec(yaw_deg=0, pitch_deg=0, fov_deg=90, width=512, height=512)
    persp = erp_to_perspective(erp, view)
    # perspective 画像の中心は ERP (W/2, H/2) と同じ色のはず
    pc = persp[persp.shape[0] // 2, persp.shape[1] // 2]
    ec = erp[H // 2, W // 2]
    # 内部で INTER_LINEAR が走るので完全一致ではなく近いことを確認
    assert np.linalg.norm(pc.astype(int) - ec.astype(int)) < 20, \
        f"center color mismatch: persp={pc} vs erp={ec}"
    print(f"  ✔ center color: persp={tuple(pc)} erp={tuple(ec)}")


def test_cubemap_covers_sphere():
    """6面cubemap + 逆射影で、ERP 全面が重みで埋まること."""
    erp_spec = ErpSpec(width=512, height=256)
    views = cubemap_views(face_size=256, fov_deg=90)
    total_weight = np.zeros((erp_spec.height, erp_spec.width), np.float32)
    for v in views:
        dummy_mask = np.ones((v.height, v.width), np.uint8) * 255
        _, w = perspective_mask_to_erp(dummy_mask, v, erp_spec)
        total_weight += w

    # 全ピクセルで少なくとも 1 つのビューからカバーされている
    uncovered = (total_weight <= 0).mean()
    assert uncovered < 0.02, f"uncovered ratio too high: {uncovered:.3f}"
    print(f"  ✔ cubemap covers sphere (uncovered={uncovered*100:.2f}%)")


def test_roundtrip_mask_persists():
    """
    ERP 空間に矩形マスクを置く → 各tangentへ perspective 化 → 逆射影 で
    元と同じおおよその位置に戻ること (IoU 測定).
    """
    erp_spec = ErpSpec(width=1024, height=512)
    # "正面中央" に 100x100 のマスクを置く
    gt = np.zeros((erp_spec.height, erp_spec.width), np.uint8)
    cx, cy = erp_spec.width // 2, erp_spec.height // 2
    gt[cy - 30:cy + 30, cx - 30:cx + 30] = 255

    views = tangent_views(n_yaw=8, n_pitch=3, fov_deg=90, width=512, height=512)

    accum = np.zeros(gt.shape, np.float32)
    weight = np.zeros(gt.shape, np.float32)
    for v in views:
        # ERP に直接 remap ではないので、各ビューで GT を perspective 化 → そのまま逆射影
        persp_gt = erp_to_perspective(gt, v, interp=cv2.INTER_NEAREST)
        m, w = perspective_mask_to_erp(persp_gt, v, erp_spec)
        accum += m * w
        weight += w
    avg = np.where(weight > 1e-6, accum / np.maximum(weight, 1e-6), 0)
    recovered = (avg > 0.5).astype(np.uint8) * 255

    inter = ((gt > 0) & (recovered > 0)).sum()
    union = ((gt > 0) | (recovered > 0)).sum()
    iou = inter / max(union, 1)
    assert iou > 0.6, f"roundtrip IoU too low: {iou:.3f}"
    print(f"  ✔ roundtrip IoU = {iou:.3f}")


def test_tangent_views_count():
    views = tangent_views(n_yaw=8, n_pitch=3)
    assert len(views) == 24
    yaws = sorted({v.yaw_deg for v in views})
    assert len(yaws) == 8
    print(f"  ✔ tangent_views: 8x3 = 24, yaws={yaws}")


def main():
    print("Panorama projection tests")
    test_tangent_views_count()
    test_perspective_center_hits_same_erp_center()
    test_cubemap_covers_sphere()
    test_roundtrip_mask_persists()
    print("\nAll panorama tests passed.")


if __name__ == "__main__":
    main()
