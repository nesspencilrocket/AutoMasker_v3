"""
典型的な COLMAP / 3D Gaussian Splatting シーン構造に対して
AutoMasker を一括でかけるサンプルレシピ.

期待するディレクトリ構造:
    scene/
    ├── images/          ← 入力画像
    ├── sparse/          ← COLMAP 出力 (触らない)
    └── masks/           ← AutoMasker が書き出す (無ければ作成)

使い方:
    python scripts/mask_scene.py /path/to/scene  --prompt "person . tripod"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# パッケージを import pathに
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automasker.config import Config
from automasker.pipeline import ImagePipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path, help="COLMAP scene directory")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--box-thresh", type=float, default=0.30)
    ap.add_argument("--text-thresh", type=float, default=0.25)
    ap.add_argument("--dilate", type=int, default=5,
                    help="境界を少し広めに消したい場合は 5-10 を推奨")
    ap.add_argument("--invert", action="store_true")
    args = ap.parse_args()

    images = args.scene / "images"
    masks = args.scene / "masks"
    if not images.is_dir():
        print(f"ERROR: {images} not found", file=sys.stderr)
        sys.exit(1)

    cfg = Config()
    cfg.box_threshold = args.box_thresh
    cfg.text_threshold = args.text_thresh
    cfg.mask_dilate = args.dilate
    cfg.invert = args.invert

    print(f"[scene]  {args.scene}")
    print(f"[device] {cfg.device}")
    print(f"[prompt] {args.prompt!r}")

    pipe = ImagePipeline(cfg)
    n = pipe.run_folder(images, masks, args.prompt,
                        progress=lambda i, t, p: print(f"  [{i}/{t}] {p.name}"))
    print(f"Done. {n} mask(s) written to {masks}")


if __name__ == "__main__":
    main()
