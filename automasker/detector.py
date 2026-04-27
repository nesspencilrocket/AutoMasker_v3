"""
Grounding DINO ラッパ.

テキストプロンプト (例: "person . tripod . camera") から画像内の候補を
バウンディングボックスで返す. 返り値は常に画像ピクセル座標系 (x1,y1,x2,y2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .config import Config


@dataclass
class Detection:
    box_xyxy: np.ndarray   # shape (4,), pixel coords
    score: float
    label: str


class GroundingDINODetector:
    """groundingdino-py の predict API をラップ."""

    def __init__(self, cfg: Config):
        from groundingdino.util.inference import load_model, predict  # noqa
        self._predict = predict

        if not Path(cfg.gdino_cfg).exists() or not Path(cfg.gdino_ckpt).exists():
            raise FileNotFoundError(
                f"Grounding DINO のチェックポイントが見つかりません: "
                f"{cfg.gdino_ckpt}\n"
                "scripts/download_checkpoints.py を実行してください."
            )

        self.model = load_model(cfg.gdino_cfg, cfg.gdino_ckpt, device=cfg.device)
        self.device = cfg.device

    @staticmethod
    def _preprocess(image_rgb: np.ndarray):
        """GroundingDINO が要求する前処理 (Resize + Normalize)."""
        import torchvision.transforms.functional as F
        from PIL import Image

        pil = Image.fromarray(image_rgb)
        # 800x1333 へリサイズ (短辺基準)
        w, h = pil.size
        scale = 800.0 / min(w, h)
        if max(w, h) * scale > 1333:
            scale = 1333.0 / max(w, h)
        new_size = (int(round(h * scale)), int(round(w * scale)))
        resized = F.resize(pil, new_size)
        tensor = F.to_tensor(resized)
        tensor = F.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return tensor

    def detect(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> List[Detection]:
        """
        prompt はピリオド区切りの小文字推奨: "person . tripod . dog"
        """
        h, w = image_rgb.shape[:2]
        caption = prompt.strip().lower()
        if not caption.endswith("."):
            caption += "."

        image_tensor = self._preprocess(image_rgb)

        boxes, logits, phrases = self._predict(
            model=self.model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        # GroundingDINO の boxes は cxcywh (正規化, [0,1]) で返ってくる
        dets: List[Detection] = []
        for i in range(len(boxes)):
            cx, cy, bw, bh = boxes[i].tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
            xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, w - 1)
            xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, h - 1)
            dets.append(Detection(
                box_xyxy=xyxy,
                score=float(logits[i]),
                label=phrases[i] if i < len(phrases) else "",
            ))
        return dets
