"""
各バックエンドの推論速度を比較するためのベンチマーク.

使い方:
    python -m export.benchmark --image sample.jpg --prompt "person" \
        --backends torch onnx trt --warmup 3 --iters 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# automasker 本体
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from automasker.config import Config


def bench_one(name: str, cfg_patch: dict, image: np.ndarray, prompt: str,
              warmup: int, iters: int) -> dict:
    cfg = Config()
    for k, v in cfg_patch.items():
        setattr(cfg, k, v)
    print(f"\n=== backend: {name} ===")
    try:
        from automasker.backends import load_detector, load_segmenter
        det = load_detector(cfg)
        seg = load_segmenter(cfg)
    except Exception as e:
        print(f"  [skip] {type(e).__name__}: {e}")
        return {"backend": name, "skipped": True, "reason": str(e)}

    # warmup
    for _ in range(warmup):
        dets = det.detect(image, prompt, cfg.box_threshold, cfg.text_threshold)
        if dets:
            boxes = np.stack([d.box_xyxy for d in dets])
            seg.segment(image, boxes)

    # measure
    t_det = []
    t_seg = []
    last_n = 0
    for _ in range(iters):
        t0 = time.perf_counter()
        dets = det.detect(image, prompt, cfg.box_threshold, cfg.text_threshold)
        t1 = time.perf_counter()
        if dets:
            boxes = np.stack([d.box_xyxy for d in dets])
            seg.segment(image, boxes)
            last_n = len(dets)
        t2 = time.perf_counter()
        t_det.append((t1 - t0) * 1000)
        t_seg.append((t2 - t1) * 1000)

    result = {
        "backend": name,
        "detect_ms":  float(np.median(t_det)),
        "segment_ms": float(np.median(t_seg)),
        "total_ms":   float(np.median(t_det) + np.median(t_seg)),
        "num_boxes":  last_n,
    }
    print(f"  detect  median: {result['detect_ms']:.1f} ms")
    print(f"  segment median: {result['segment_ms']:.1f} ms")
    print(f"  total   median: {result['total_ms']:.1f} ms  ({last_n} boxes)")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--backends", nargs="+",
                    default=["torch", "onnx", "trt"],
                    choices=["torch", "onnx", "trt"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args(argv)

    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(args.image)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    PATCHES = {
        "torch": {"backend_detector": "torch", "backend_segmenter": "torch"},
        "onnx":  {"backend_detector": "onnx",  "backend_segmenter": "onnx"},
        "trt":   {"backend_detector": "trt",   "backend_segmenter": "trt"},
    }

    results = []
    for name in args.backends:
        results.append(bench_one(name, PATCHES[name], image_rgb, args.prompt,
                                 args.warmup, args.iters))

    # サマリ表
    print("\n" + "=" * 60)
    print(f"{'backend':<10} {'detect':>10} {'segment':>10} {'total':>10} {'speedup':>10}")
    print("-" * 60)
    base = next((r["total_ms"] for r in results
                 if not r.get("skipped") and r["backend"] == "torch"), None)
    for r in results:
        if r.get("skipped"):
            print(f"{r['backend']:<10} {'-':>10} {'-':>10} {'-':>10} {'skipped':>10}")
            continue
        sp = f"{base / r['total_ms']:.2f}x" if base else "-"
        print(f"{r['backend']:<10} "
              f"{r['detect_ms']:>10.1f} "
              f"{r['segment_ms']:>10.1f} "
              f"{r['total_ms']:>10.1f} "
              f"{sp:>10}")


if __name__ == "__main__":
    sys.exit(main())
