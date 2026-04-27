"""
mask-lift CLI: AutoMasker が生成した masks/ と、学習済み point_cloud.ply を
入力に、対象オブジェクトに属する Gaussian だけを抜き出した .ply を出力する.

典型ワークフロー:
    # 1) 先に AutoMasker でマスクを生成
    python -m automasker.cli -i scene/images -o scene/masks --prompt "person"

    # 2) 3DGS を学習 (通常フロー). scene/point_cloud.ply が出力される想定.

    # 3) mask-lift でリフト
    python -m automasker.mask_lift.lift_cli \
        --scene scene \
        --ply   scene/point_cloud/iteration_30000/point_cloud.ply \
        --masks scene/masks \
        --output-dir scene/mask_lift_out \
        --mode extract        # 対象 Gaussian だけ残す (物体抽出)
        # あるいは --mode remove で対象を消した ply を出力 (物体除去)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from .camera_io import load_scene
from .lift import lift_masks_to_gaussians
from .ply_io import load_ply, save_ply


def build_mask_loader(masks_dir: Path, invert: bool = False):
    """
    AutoMasker 既定 (対象=0, 背景=255) → "対象=255" に変換して返すローダを作る.
    invert=True は入力 png 自体が "対象=255" 仕様の場合に使う.
    """
    cache = {}

    def _load(image_name: str):
        stem = Path(image_name).stem
        # 複数拡張子に対応
        for ext in (".png", ".jpg", ".jpeg"):
            p = masks_dir / f"{stem}{ext}"
            if p.exists():
                break
        else:
            return None
        if p in cache:
            return cache[p]
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if not invert:
            # AutoMasker 既定: 対象=0 → 反転して対象=255 にする
            m = 255 - m
        cache[p] = m
        return m

    return _load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True,
                    help="COLMAP scene directory (sparse/ を含む)")
    ap.add_argument("--ply", type=Path, required=True,
                    help="学習済み point_cloud.ply")
    ap.add_argument("--masks", type=Path, required=True,
                    help="AutoMasker 出力の masks/ フォルダ")
    ap.add_argument("--output-dir", type=Path, default=Path("mask_lift_out"))

    ap.add_argument("--mode", choices=["extract", "remove", "both"], default="both",
                    help="extract=対象抽出 / remove=対象除去 / both=両方保存")
    ap.add_argument("--bg-bias", type=float, default=0.2,
                    help="λ. 大きいほど対象判定が厳しくなる (0.1-0.5 が実用域)")
    ap.add_argument("--max-views", type=int, default=None,
                    help="デバッグ用: 先頭 N ビューだけ使う")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", choices=["approx", "gsplat"], default="approx",
                    help="approx = 等方 circular splat 近似 (外部依存なし). "
                         "gsplat = nerfstudio-project/gsplat で厳密 anisotropic 計算.")
    ap.add_argument("--batch-size", type=int, default=2048,
                    help="gsplat 使用時のガウス batch サイズ")
    ap.add_argument("--mask-already-target255", action="store_true",
                    help="入力マスクが既に 対象=255 になっている場合に指定")
    ap.add_argument("--save-score-npy", action="store_true",
                    help="各 Gaussian のスコアを .npy でも保存する")
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] COLMAP scene:  {args.scene}")
    cameras, images = load_scene(args.scene)
    print(f"  cameras: {len(cameras)}  images: {len(images)}")

    print(f"[load] PLY:  {args.ply}")
    g = load_ply(args.ply)
    print(f"  gaussians: {g.n}")

    mask_loader = build_mask_loader(args.masks, invert=args.mask_already_target255)

    def _report(i, total, name):
        if i == 1 or i == total or i % 10 == 0:
            print(f"  [{i}/{total}] {name}", flush=True)

    print(f"[lift] solving closed-form assignment (λ={args.bg_bias}, backend={args.backend})…")
    if args.backend == "gsplat":
        from .lift_gsplat import lift_masks_to_gaussians_gsplat
        result = lift_masks_to_gaussians_gsplat(
            g, cameras, images, mask_loader,
            bg_bias=args.bg_bias, device=args.device,
            batch_size=args.batch_size,
            progress=_report, max_views=args.max_views,
        )
    else:
        result = lift_masks_to_gaussians(
            g, cameras, images, mask_loader,
            bg_bias=args.bg_bias, device=args.device,
            progress=_report, max_views=args.max_views,
        )
    n_target = int(result.labels.sum())
    print(f"  used_views = {result.used_views}")
    print(f"  labeled target: {n_target} / {g.n}  ({100 * n_target / g.n:.1f}%)")

    # 出力
    if args.mode in ("extract", "both"):
        out = args.output_dir / "point_cloud_target.ply"
        print(f"[save] target only → {out}")
        save_ply(out, g, mask=result.labels)
    if args.mode in ("remove", "both"):
        out = args.output_dir / "point_cloud_without_target.ply"
        print(f"[save] target removed → {out}")
        save_ply(out, g, mask=~result.labels)

    if args.save_score_npy:
        np.savez(
            args.output_dir / "labels.npz",
            labels=result.labels, score=result.score,
            w_sum=result.w_sum, w_mask_sum=result.w_mask_sum,
        )
        print("[save] labels.npz")

    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
