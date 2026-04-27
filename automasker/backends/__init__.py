"""
推論バックエンド抽象化.

同じインタフェースで torch / onnxruntime / tensorrt を切り替えられるようにする.
AutoMasker のパイプラインからは `load_detector(cfg)` / `load_segmenter(cfg)` しか
呼ばない設計にして、中身の差し替えを一箇所に閉じ込める.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DetectorBackend(Protocol):
    """Grounding DINO 相当のインタフェース."""

    def detect(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> list:
        """[Detection(box_xyxy, score, label), ...] を返す."""
        ...


@runtime_checkable
class SegmenterBackend(Protocol):
    """SAM2 image predictor 相当のインタフェース."""

    def segment(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """boxes から (H, W) uint8 の2値マスクを返す (対象=255)."""
        ...


def load_detector(cfg) -> DetectorBackend:
    """cfg.backend_detector に従って実装を選ぶ."""
    name = getattr(cfg, "backend_detector", "torch").lower()
    if name == "torch":
        from ..detector import GroundingDINODetector
        return GroundingDINODetector(cfg)
    if name == "onnx":
        from .gdino_onnx import GroundingDINOONNX
        return GroundingDINOONNX(cfg)
    if name in ("trt", "tensorrt"):
        from .gdino_trt import GroundingDINOTRT
        return GroundingDINOTRT(cfg)
    raise ValueError(f"Unknown detector backend: {name!r}")


def load_segmenter(cfg) -> SegmenterBackend:
    name = getattr(cfg, "backend_segmenter", "torch").lower()
    if name == "torch":
        from ..segmenter import SAM2Segmenter
        return SAM2Segmenter(cfg)
    if name == "onnx":
        from .sam2_onnx import SAM2ONNX
        return SAM2ONNX(cfg)
    if name in ("trt", "tensorrt"):
        from .sam2_trt import SAM2TRT
        return SAM2TRT(cfg)
    raise ValueError(f"Unknown segmenter backend: {name!r}")
