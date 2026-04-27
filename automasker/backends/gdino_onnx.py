"""
Grounding DINO の ONNX Runtime バックエンド.

Grounding DINO は BERT ベースの caption encoder と Swin backbone を持つ重いモデル
のため、naive な export では精度が落ちやすい. 本実装は IDEA-Research の
公式 `demo/export_onnx.py` 相当で出したモデル (入力: image, caption embeddings)
を前提にする. トークナイズ自体は Python 側で HuggingFace tokenizer を使用.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from ..detector import Detection


# Grounding DINO の既定解像度 (短辺 800, 長辺 1333 以下)
GDINO_MIN_SIZE = 800
GDINO_MAX_SIZE = 1333


def _preprocess(image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = GDINO_MIN_SIZE / min(h, w)
    if max(h, w) * scale > GDINO_MAX_SIZE:
        scale = GDINO_MAX_SIZE / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255
    arr = (resized.astype(np.float32) - mean) / std
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)  # (1, 3, H, W)


class GroundingDINOONNX:
    def __init__(self, cfg):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        ckpt = Path(getattr(cfg, "gdino_onnx", "checkpoints/gdino.onnx"))
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Grounding DINO ONNX が見つかりません: {ckpt}\n"
                "python -m export.gdino_onnx で生成してください."
            )

        providers = self._select_providers(cfg)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(ckpt), so, providers=providers)

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        tokenizer_name = getattr(cfg, "gdino_tokenizer", "bert-base-uncased")
        # SECURITY: HF からダウンロードする際は revision を固定して
        # 上流リポジトリの改ざんに備える. bert-base-uncased の特定 commit に pin.
        tokenizer_rev = getattr(cfg, "gdino_tokenizer_revision",
                                "86b5e0934494bd15c9632b12f734a8a67f723594")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, revision=tokenizer_rev,
        )
        self.max_text_len = getattr(cfg, "gdino_max_text_len", 256)

    @staticmethod
    def _select_providers(cfg) -> list:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        wanted: list = []
        if getattr(cfg, "use_trt_ep", False) and "TensorrtExecutionProvider" in avail:
            wanted.append((
                "TensorrtExecutionProvider",
                {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                 "trt_engine_cache_path": ".trt_cache"},
            ))
        if "CUDAExecutionProvider" in avail and getattr(cfg, "device", "cuda") != "cpu":
            wanted.append("CUDAExecutionProvider")
        wanted.append("CPUExecutionProvider")
        return wanted

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
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

        feed = {"image": img, **tok}
        # export 時に違う命名を使っている場合に備えて未知キーを除外
        feed = {k: v for k, v in feed.items() if k in self.input_names}

        outs = self.session.run(self.output_names, feed)
        # 多くの export 実装は (boxes_cxcywh_norm, logits) を返す
        boxes_cxcywh = outs[0]  # (num_queries, 4)
        logits = outs[1]        # (num_queries, num_tokens)
        if boxes_cxcywh.ndim == 3:
            boxes_cxcywh = boxes_cxcywh[0]
            logits = logits[0]

        # 各クエリのスコアを "対象テキストトークンに対する最大値" で決める流儀
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
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
            # ラベルはおおざっぱに閾値越えトークン列を文字列化
            tok_mask = probs[idx] > text_threshold
            label_ids = tok["input_ids"][0][tok_mask]
            label = self.tokenizer.decode(label_ids).strip()
            dets.append(Detection(xyxy, float(max_text_prob[idx]), label))
        return dets
