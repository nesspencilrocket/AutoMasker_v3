"""
ONNX → TensorRT Engine ビルダ.

2 つのモードをサポート:
  1) trtexec (デフォルト, 推奨) — NVIDIA 付属の CLI を subprocess で叩く
  2) python  — TensorRT Python API で自前ビルド (trtexec が PATH に無い時のフォールバック)

使い方:
    # SAM2 encoder + decoder を一括変換
    python -m export.build_trt --target sam2 --fp16

    # GDINO
    python -m export.build_trt --target gdino --fp16

    # 任意の ONNX
    python -m export.build_trt --onnx foo.onnx --engine foo.engine --fp16
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def build_with_trtexec(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    workspace_mb: int = 4096,
    min_shapes: Optional[str] = None,
    opt_shapes: Optional[str] = None,
    max_shapes: Optional[str] = None,
) -> None:
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError(
            "trtexec が PATH に見つかりません. TensorRT をインストールするか "
            "--backend python を指定してください."
        )
    # SECURITY: ファイルパスが "-" で始まるとオプション扱いされうるので弾く.
    for p in (onnx_path, engine_path):
        s = str(p)
        if s.startswith("-"):
            raise ValueError(f"path starts with '-', refusing: {s!r}")

    # 絶対パスに解決しておく (trtexec の CWD 依存を避ける)
    onnx_abs = Path(onnx_path).resolve()
    engine_abs = Path(engine_path).resolve()

    cmd: List[str] = [
        trtexec,
        f"--onnx={onnx_abs}",
        f"--saveEngine={engine_abs}",
        f"--memPoolSize=workspace:{int(workspace_mb)}",
    ]
    if fp16:
        cmd.append("--fp16")
    if min_shapes:
        cmd.append(f"--minShapes={min_shapes}")
    if opt_shapes:
        cmd.append(f"--optShapes={opt_shapes}")
    if max_shapes:
        cmd.append(f"--maxShapes={max_shapes}")

    print(" ".join(cmd))
    # nosec B603: cmd は trtexec (shutil.which で解決済み) + 検証済みの引数のみ.
    # shell=True は使わず、パスの先頭が "-" のものは上で弾いている.
    subprocess.run(cmd, check=True)  # nosec B603


def build_with_python_api(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    workspace_mb: int = 4096,
    profile_shapes: Optional[dict] = None,
) -> None:
    """
    TensorRT Python API による engine ビルド.
    profile_shapes: {"input_name": ((min,), (opt,), (max,)), ...}
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    if profile_shapes:
        profile = builder.create_optimization_profile()
        for name, (mn, opt, mx) in profile_shapes.items():
            profile.set_shape(name, mn, opt, mx)
        config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"Saved engine → {engine_path}")


# --------------------------------------------------------------------------
# 高レベルヘルパ: target ごとの推奨設定
# --------------------------------------------------------------------------
def build_sam2(output_dir: Path, backend: str, fp16: bool):
    enc_onnx = output_dir / "sam2_encoder.onnx"
    dec_onnx = output_dir / "sam2_decoder.onnx"
    enc_eng = output_dir / "sam2_encoder.engine"
    dec_eng = output_dir / "sam2_decoder.engine"

    if not enc_onnx.exists() or not dec_onnx.exists():
        raise FileNotFoundError(
            "先に python -m export.sam2_onnx で ONNX を出力してください."
        )

    # encoder は固定形状. そのまま変換して OK.
    _build(enc_onnx, enc_eng, backend=backend, fp16=fp16)

    # decoder は num_boxes が動的. 1-16 程度でプロファイルを設定する.
    decoder_shapes = {
        "point_coords":   ((1, 2, 2), (4, 2, 2), (16, 2, 2)),
        "point_labels":   ((1, 2),   (4, 2),   (16, 2)),
        "mask_input":     ((1, 1, 256, 256), (4, 1, 256, 256), (16, 1, 256, 256)),
        "has_mask_input": ((1,),     (4,),     (16,)),
    }
    if backend == "trtexec":
        mn = ",".join(f"{k}:{'x'.join(map(str, v[0]))}" for k, v in decoder_shapes.items())
        opt = ",".join(f"{k}:{'x'.join(map(str, v[1]))}" for k, v in decoder_shapes.items())
        mx = ",".join(f"{k}:{'x'.join(map(str, v[2]))}" for k, v in decoder_shapes.items())
        build_with_trtexec(dec_onnx, dec_eng, fp16=fp16,
                           min_shapes=mn, opt_shapes=opt, max_shapes=mx)
    else:
        build_with_python_api(dec_onnx, dec_eng, fp16=fp16,
                              profile_shapes=decoder_shapes)


def build_gdino(output_dir: Path, backend: str, fp16: bool):
    onnx_path = output_dir / "gdino.onnx"
    engine_path = output_dir / "gdino.engine"
    if not onnx_path.exists():
        raise FileNotFoundError(
            "先に python -m export.gdino_onnx で ONNX を出力してください."
        )
    shapes = {
        "image":          ((1, 3, 600, 800), (1, 3, 800, 1200), (1, 3, 1333, 1333)),
        "input_ids":      ((1, 32), (1, 128), (1, 256)),
        "attention_mask": ((1, 32), (1, 128), (1, 256)),
        "token_type_ids": ((1, 32), (1, 128), (1, 256)),
    }
    if backend == "trtexec":
        def _join(idx):
            return ",".join(f"{k}:{'x'.join(map(str, v[idx]))}" for k, v in shapes.items())
        build_with_trtexec(onnx_path, engine_path, fp16=fp16,
                           min_shapes=_join(0), opt_shapes=_join(1), max_shapes=_join(2))
    else:
        build_with_python_api(onnx_path, engine_path, fp16=fp16, profile_shapes=shapes)


def _build(onnx_path: Path, engine_path: Path, *, backend: str, fp16: bool):
    if backend == "trtexec":
        build_with_trtexec(onnx_path, engine_path, fp16=fp16)
    else:
        build_with_python_api(onnx_path, engine_path, fp16=fp16)


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["sam2", "gdino", "custom"], default="custom")
    ap.add_argument("--onnx", type=Path, help="custom target の入力")
    ap.add_argument("--engine", type=Path, help="custom target の出力")
    ap.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--backend", choices=["trtexec", "python"], default="trtexec")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args(argv)

    fp16 = args.fp16 and not args.fp32

    if args.target == "sam2":
        build_sam2(args.output_dir, args.backend, fp16)
    elif args.target == "gdino":
        build_gdino(args.output_dir, args.backend, fp16)
    else:
        if not args.onnx or not args.engine:
            ap.error("--target custom needs --onnx and --engine")
        _build(args.onnx, args.engine, backend=args.backend, fp16=fp16)


if __name__ == "__main__":
    sys.exit(main())
