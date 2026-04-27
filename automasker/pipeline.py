"""
統合パイプライン.

- ImagePipeline: 各画像について  Detect → Segment → Refine  を実行し、
                 COLMAP 互換の masks/ に出力する.
- VideoPipeline: 1フレーム目で Detect し、SAM2 の VOS で全フレーム伝搬する.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from tqdm import tqdm

from . import io_utils
from .config import Config
from .refine import refine_mask


@dataclass
class FrameResult:
    image_path: Path
    mask: np.ndarray            # uint8, 対象=255
    num_detections: int


class ImagePipeline:
    """画像 (フォルダ) 向け."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # バックエンド選択 (torch/onnx/trt) は cfg.backend_* で切替
        from .backends import load_detector, load_segmenter
        self.detector = load_detector(cfg)
        self.segmenter = load_segmenter(cfg)

    def run_single(self, image_path: Path, prompt: str) -> FrameResult:
        img = io_utils.read_image(image_path)
        dets = self.detector.detect(
            img, prompt,
            box_threshold=self.cfg.box_threshold,
            text_threshold=self.cfg.text_threshold,
        )
        if len(dets) == 0:
            mask = np.zeros(img.shape[:2], np.uint8)
        else:
            boxes = np.stack([d.box_xyxy for d in dets], axis=0)
            mask = self.segmenter.segment(img, boxes)
            mask = refine_mask(
                mask,
                dilate=self.cfg.mask_dilate,
                erode=self.cfg.mask_erode,
                min_area=self.cfg.mask_min_area,
            )
        return FrameResult(image_path=image_path, mask=mask, num_detections=len(dets))

    def run_folder(
        self,
        input_dir: Path,
        output_dir: Path,
        prompt: str,
        progress: Optional[Callable[[int, int, Path], None]] = None,
    ) -> int:
        """
        フォルダ内の画像を一括処理し masks/<stem>.png を保存する.
        progress(i, total, current_path) で進捗を通知.
        """
        images = io_utils.list_images(input_dir, self.cfg)
        total = len(images)
        output_dir.mkdir(parents=True, exist_ok=True)

        it = enumerate(images)
        if progress is None:
            it = tqdm(it, total=total, desc="Masking")

        for i, p in it:
            res = self.run_single(p, prompt)
            out = io_utils.mask_path_for(p, output_dir)
            io_utils.write_mask(out, res.mask, invert=self.cfg.invert)
            if progress is not None:
                progress(i + 1, total, p)
        return total


class VideoPipeline:
    """動画 / 連番画像向け. 1フレーム目の検出結果を時間方向に伝搬."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        from .backends import load_detector
        self.detector = load_detector(cfg)
        # video predictor は SAM2 の内部メモリ機構を使う専用実装のみ
        from .segmenter import SAM2VideoSegmenter
        self.video_seg = SAM2VideoSegmenter(cfg)

    def run_video(
        self,
        video_path: Path,
        output_dir: Path,
        prompt: str,
        frames_cache: Path,
        stride: int = 1,
        max_frames: Optional[int] = None,
        progress: Optional[Callable[[int, int, Path], None]] = None,
    ) -> int:
        frame_paths, _ = io_utils.video_to_frames(
            video_path, frames_cache, stride=stride, max_frames=max_frames,
        )
        return self._run_sequence(frame_paths, frames_cache, output_dir, prompt, progress)

    def run_image_sequence(
        self,
        input_dir: Path,
        output_dir: Path,
        prompt: str,
        progress: Optional[Callable[[int, int, Path], None]] = None,
    ) -> int:
        """
        入力が既に連番画像フォルダの場合 (例: COLMAPの images/).
        SAM2 video predictor は "000000.jpg" 形式を要求するので、
        必要に応じてシンボリックリンク/コピーで整形する必要がある。
        ここでは直接渡せる前提. 整形が必要な場合はCLI側で対応する.
        """
        images = io_utils.list_images(input_dir, self.cfg)
        return self._run_sequence(images, input_dir, output_dir, prompt, progress)

    def _run_sequence(
        self,
        frame_paths: List[Path],
        frames_dir: Path,
        output_dir: Path,
        prompt: str,
        progress: Optional[Callable[[int, int, Path], None]],
    ) -> int:
        if not frame_paths:
            return 0
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1フレーム目で検出
        first = io_utils.read_image(frame_paths[0])
        dets = self.detector.detect(
            first, prompt,
            box_threshold=self.cfg.box_threshold,
            text_threshold=self.cfg.text_threshold,
        )
        if not dets:
            raise RuntimeError(
                "1フレーム目で対象が検出できませんでした。"
                "box_threshold を下げるか prompt を見直してください。"
            )
        boxes = [d.box_xyxy for d in dets]

        # SAM2 で全フレーム伝搬
        masks_by_idx = self.video_seg.propagate(frames_dir, boxes)

        total = len(frame_paths)
        for i, p in enumerate(frame_paths):
            mask = masks_by_idx.get(i, np.zeros(first.shape[:2], np.uint8))
            mask = refine_mask(
                mask,
                dilate=self.cfg.mask_dilate,
                erode=self.cfg.mask_erode,
                min_area=self.cfg.mask_min_area,
            )
            out = io_utils.mask_path_for(p, output_dir)
            io_utils.write_mask(out, mask, invert=self.cfg.invert)
            if progress is not None:
                progress(i + 1, total, p)
        return total
