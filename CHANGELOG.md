# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0b1] — 2026-04-19 (beta)

Initial public release.

### Added
- Core mask generation pipeline: Grounding DINO for text-prompted detection +
  SAM 2 for mask prediction, with COLMAP/3DGS-compatible output format
  (`masks/<stem>.png`).
- CLI (`automasker`) and PySide6 GUI for batch processing.
- Video support via SAM 2 VOS for temporally consistent masks across frames.
- 360° equirectangular image support via tangent patches (8 yaw × 3 pitch
  by default) or cubemap (6 faces), with ERP↔perspective round-trip of
  IoU ≥ 0.99.
- 2D→3D mask lifting for 3D Gaussian Splatting scenes
  (`automasker.mask_lift`), implementing the closed-form label assignment
  from Shen et al. ECCV 2024 "FlashSplat" paper. Two backends:
    - `approx`: isotropic circular splat (pure numpy/torch, no external deps)
    - `gsplat`: anisotropic rasterization via nerfstudio-project/gsplat
- ONNX Runtime and TensorRT backends for 2-4× inference speedup.
- HTTP inference server (`automasker.server`) with Bearer token auth.
- C++/Dear ImGui low-latency frontend (`cpp_frontend/`) that talks to the
  HTTP server.
- Export pipeline for converting SAM2 and Grounding DINO to ONNX/TensorRT.

### Security
- `torch.load(..., weights_only=True)` throughout to prevent pickle RCE.
- HTTP server defaults to 127.0.0.1-only; requires `--allow-remote` and
  `--token` to bind to non-loopback.
- 25 MB request body cap and 50M pixel image cap on `/infer` endpoint.
- Magic byte validation for uploaded images (PNG/JPEG/WebP only).
- Path sanitization on mask output to defend against null-byte and
  path-traversal in filenames (`io_utils._sanitize_stem`).
- Hardcoded HF tokenizer revision for bert-base-uncased to pin against
  upstream tampering.
- Subprocess args (trtexec) guarded against `-`-prefix option injection.
- `.ply` loader has a 4 GB default size cap.
- Download script enforces HTTPS-only URLs.
- CI runs bandit (Medium+) and pip-audit on every push/PR.

### Known limitations
- `mask_lift` is an independent reimplementation of the paper's
  algorithm — not a port of the official `florinshen/FlashSplat` code
  (which is under an INRIA non-commercial research license).
  Numerical agreement with the paper is **not** guaranteed.
- gsplat backend batches gaussians independently; cross-batch occlusion
  is not modeled. Set `batch_size` to the full gaussian count (if VRAM
  allows) to get the paper-equivalent result.
- ONNX/TensorRT backends reimplement preprocessing and postprocessing and
  are close to but not exactly equivalent to the torch backend.
- Grounding DINO is English-trained; CJK prompts will have degraded accuracy.
- SAM 2 video predictor holds all masks in memory; for 1080p × 1000 frames
  budget ~2 GB RAM. Use `--stride` or `--max-frames` for long videos.
- HTTP server serializes inference via `_INFERENCE_LOCK` (SAM 2 is not
  thread-safe). Concurrent requests are handled but not parallelized.
