"""
SAM 2 ラッパ.

- SAM2ImagePredictor:  単一画像用。box/points prompt から mask を生成する。
- SAM2VideoPredictor:  フレーム連番ディレクトリを渡し、1フレーム目で与えた
                        box/mask を時間方向に伝搬する (VOS / 時間的整合性).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


class SAM2Segmenter:
    """単一画像向けのラッパ."""

    def __init__(self, cfg):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(cfg.sam2_cfg, cfg.sam2_ckpt, device=cfg.device)
        self.predictor = SAM2ImagePredictor(model)
        self.device = cfg.device

    def segment(
        self,
        image_rgb: np.ndarray,
        boxes_xyxy: np.ndarray,  # shape (N,4)
    ) -> np.ndarray:
        """
        N 個のボックスから N 枚のマスクを作り、OR 合成して1枚に畳む.
        返り値: uint8 の2値マスク (H,W), 対象=255.
        """
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            h, w = image_rgb.shape[:2]
            return np.zeros((h, w), np.uint8)

        self.predictor.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(
            box=boxes_xyxy,
            multimask_output=False,
        )
        # masks shape: (N, 1, H, W) or (N, H, W)
        if masks.ndim == 4:
            masks = masks[:, 0]
        combined = np.any(masks > 0, axis=0).astype(np.uint8) * 255
        return combined


class SAM2VideoSegmenter:
    """
    動画 / 画像シーケンス向け. 最初のフレームで与えたボックス・マスクを
    SAM2 のメモリ機構で全フレームに伝搬する.
    """

    def __init__(self, cfg):
        from sam2.build_sam import build_sam2_video_predictor
        self.predictor = build_sam2_video_predictor(
            cfg.sam2_cfg, cfg.sam2_ckpt, device=cfg.device,
        )
        self.device = cfg.device

    def propagate(
        self,
        frames_dir: Path,
        boxes_per_object_first_frame: List[np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """
        frames_dir: 連番 JPEG が入ったディレクトリ (000000.jpg, 000001.jpg, ...)
        boxes_per_object_first_frame: 1フレーム目での対象ボックス一覧.
            各ボックスを独立オブジェクトとして追跡する.
        戻り値: {frame_idx: uint8 mask (H,W)} の辞書 (全オブジェクトを OR 合成)
        """
        state = self.predictor.init_state(video_path=str(frames_dir))

        # 1フレーム目に全ボックスを与える
        for obj_id, box in enumerate(boxes_per_object_first_frame):
            self.predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                box=box.astype(np.float32),
            )

        # 伝搬
        result: Dict[int, np.ndarray] = {}
        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
            # mask_logits: (num_objects, 1, H, W) float tensor
            combined = None
            for i in range(mask_logits.shape[0]):
                m = (mask_logits[i, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                combined = m if combined is None else np.maximum(combined, m)
            if combined is not None:
                result[frame_idx] = combined
        return result
