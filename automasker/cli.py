"""
CLI エントリポイント.

使用例:
    # 画像フォルダ
    python -m automasker.cli -i scene/images -o scene/masks \
        --prompt "person . tripod"

    # 動画 (VOSで時間伝搬)
    python -m automasker.cli -i input.mp4 -o out/masks \
        --prompt "person" --video --frames-cache /tmp/frames
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from .config import Config


def _default_frames_cache() -> Path:
    """OS に応じた一時ディレクトリの下にセッションごとのサブディレクトリを作る."""
    return Path(tempfile.gettempdir()) / "automasker_frames"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="automasker",
        description="Text-prompt driven auto masking for COLMAP / 3DGS training.",
    )
    p.add_argument("-i", "--input", required=True, type=Path,
                   help="入力フォルダ or 動画ファイル")
    p.add_argument("-o", "--output", required=True, type=Path,
                   help="masks/ を書き出すフォルダ")
    p.add_argument("--prompt", required=True,
                   help='検出対象 (ピリオド区切り). 例: "person . tripod"')
    p.add_argument("--video", action="store_true",
                   help="入力が動画ファイル、または連番画像で時間伝搬したい場合")
    p.add_argument("--frames-cache", type=Path, default=None,
                   help="動画をフレーム展開する作業ディレクトリ "
                        "(既定: OSの一時ディレクトリ)")
    p.add_argument("--stride", type=int, default=1, help="動画のフレーム間引き")
    p.add_argument("--max-frames", type=int, default=None,
                   help="動画処理時の最大フレーム数")

    p.add_argument("--box-thresh", type=float, default=0.30)
    p.add_argument("--text-thresh", type=float, default=0.25)
    p.add_argument("--dilate", type=int, default=3, help="膨張 (px)")
    p.add_argument("--erode", type=int, default=0, help="収縮 (px)")
    p.add_argument("--min-area", type=int, default=128)

    p.add_argument("--invert", action="store_true",
                   help="対象だけを残す (マスク反転)")
    p.add_argument("--device", default=None, help="cuda / mps / cpu")
    p.add_argument("--backend", choices=["torch", "onnx", "trt"], default="torch",
                   help="推論バックエンド. onnx/trt は事前に export スクリプトが必要.")
    p.add_argument("--use-trt-ep", action="store_true",
                   help="onnx バックエンド時に TensorRT ExecutionProvider を使う")

    # --- パノラマ ---
    p.add_argument("--pano", action="store_true",
                   help="入力を equirectangular (360度) として扱い、"
                        "tangent patch ごとに推論→ERPへ合成する")
    p.add_argument("--pano-mode", choices=["tangent", "cubemap"], default="tangent")
    p.add_argument("--pano-n-yaw", type=int, default=8)
    p.add_argument("--pano-n-pitch", type=int, default=3)
    p.add_argument("--pano-fov", type=float, default=90.0)
    p.add_argument("--pano-patch-size", type=int, default=1024)
    p.add_argument("--pano-save-patches", action="store_true",
                   help="デバッグ用に各パッチの画像とマスクを出力する")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = Config()
    if args.device:
        cfg.device = args.device
    cfg.box_threshold = args.box_thresh
    cfg.text_threshold = args.text_thresh
    cfg.mask_dilate = args.dilate
    cfg.mask_erode = args.erode
    cfg.mask_min_area = args.min_area
    cfg.invert = args.invert
    cfg.backend_detector = args.backend
    cfg.backend_segmenter = args.backend
    cfg.use_trt_ep = args.use_trt_ep

    print(f"[AutoMasker] device = {cfg.device}")
    print(f"[AutoMasker] prompt = {args.prompt!r}")

    def report(i, total, path):
        print(f"  [{i}/{total}] {path.name}", flush=True)

    if args.pano and args.video:
        print("WARN: --pano と --video が同時指定されました. --pano を優先します.",
              file=sys.stderr)

    if args.pano:
        from .pano import ErpMaskPipeline
        from .pano.pipeline import ErpMaskOptions
        opts = ErpMaskOptions(
            mode=args.pano_mode,
            n_yaw=args.pano_n_yaw,
            n_pitch=args.pano_n_pitch,
            fov_deg=args.pano_fov,
            patch_size=args.pano_patch_size,
        )
        pipe = ErpMaskPipeline(cfg, opts)
        if args.input.is_dir():
            n = pipe.run_folder(args.input, args.output, args.prompt,
                                progress=report,
                                save_patches=args.pano_save_patches)
        else:
            import cv2
            erp_bgr = cv2.imread(str(args.input))
            if erp_bgr is None:
                print(f"ERROR: cannot read image: {args.input}", file=sys.stderr)
                return 1
            erp = cv2.cvtColor(erp_bgr, cv2.COLOR_BGR2RGB)
            debug_dir = (args.output.parent / "_debug") if args.pano_save_patches else None
            binary = pipe.run_erp_image(erp, args.prompt, save_patches_dir=debug_dir)
            from . import io_utils
            args.output.mkdir(parents=True, exist_ok=True)
            # mask_path_for を経由して null-byte / path traversal からサニタイズ
            out = io_utils.mask_path_for(args.input, args.output)
            io_utils.write_mask(out, binary, invert=cfg.invert)
            n = 1
    elif args.video or args.input.suffix.lower() in cfg.video_exts:
        from .pipeline import VideoPipeline
        pipe = VideoPipeline(cfg)
        if args.input.is_file():
            frames_cache = args.frames_cache or _default_frames_cache()
            n = pipe.run_video(
                args.input, args.output, args.prompt,
                frames_cache=frames_cache,
                stride=args.stride,
                max_frames=args.max_frames,
                progress=report,
            )
        else:
            n = pipe.run_image_sequence(args.input, args.output, args.prompt, progress=report)
    else:
        from .pipeline import ImagePipeline
        pipe = ImagePipeline(cfg)
        n = pipe.run_folder(args.input, args.output, args.prompt, progress=report)

    print(f"[AutoMasker] done. {n} mask(s) written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
