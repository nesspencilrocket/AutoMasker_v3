"""
Grounding DINO の TensorRT Engine 版バックエンド.

ONNX版と処理の流れは同じで、InferenceSession を _TRTRunner に差し替えただけ.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from ..detector import Detection
from .gdino_onnx import _preprocess
from .sam2_trt import _TRTRunner


class GroundingDINOTRT:
    def __init__(self, cfg):
        from transformers import AutoTokenizer

        engine = Path(getattr(cfg, "gdino_engine", "checkpoints/gdino.engine"))
        if not engine.exists():
            raise FileNotFoundError(
                f"TensorRT engine が見つかりません: {engine}\n"
                "python -m export.build_trt --target gdino で生成してください."
            )
        self.runner = _TRTRunner(engine)

        tokenizer_name = getattr(cfg, "gdino_tokenizer", "bert-base-uncased")
        # SECURITY: gdino_onnx.py と同じコミットハッシュに pin.
        tokenizer_rev = getattr(cfg, "gdino_tokenizer_revision",
                                "86b5e0934494bd15c9632b12f734a8a67f723594")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, revision=tokenizer_rev,
        )
        self.max_text_len = getattr(cfg, "gdino_max_text_len", 256)

    def _tokenize(self, caption: str) -> dict:
        if not caption.endswith("."):
            caption += "."
        tokens = self.tokenizer(
            [caption], padding="max_length", max_length=self.max_text_len,
            truncation=True, return_tensors="np",
        )
        return {
            "input_ids": tokens["input_ids"].astype(np.int64),
            "attention_mask": tokens["attention_mask"].astype(np.int64),
            "token_type_ids": tokens.get(
                "token_type_ids",
                np.zeros_like(tokens["input_ids"])
            ).astype(np.int64),
        }

    def detect(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> List[Detection]:
        h, w = image_rgb.shape[:2]
        img = _preprocess(image_rgb)
        tok = self._tokenize(prompt)

        inputs = {"image": img, **tok}
        inputs = {k: v for k, v in inputs.items() if k in self.runner.input_names}

        outs = self.runner.run(inputs)
        # 出力順は engine ビルド時に固定. ここでは先頭2つを boxes / logits と仮定.
        keys = list(outs.keys())
        boxes_cxcywh = outs[keys[0]]
        logits = outs[keys[1]]
        if boxes_cxcywh.ndim == 3:
            boxes_cxcywh = boxes_cxcywh[0]
            logits = logits[0]

        probs = 1.0 / (1.0 + np.exp(-logits))
        max_text_prob = probs.max(axis=1)
        keep = max_text_prob > box_threshold

        dets: List[Detection] = []
        for idx in np.where(keep)[0]:
            cx, cy, bw, bh = boxes_cxcywh[idx].tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            xyxy = np.clip(
                np.array([x1, y1, x2, y2], dtype=np.float32),
                [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1],
            )
            tok_mask = probs[idx] > text_threshold
            label_ids = tok["input_ids"][0][tok_mask]
            label = self.tokenizer.decode(label_ids).strip()
            dets.append(Detection(xyxy, float(max_text_prob[idx]), label))
        return dets
