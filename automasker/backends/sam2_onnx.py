"""
SAM2 を ONNX Runtime で推論するバックエンド.

Tier IV 方式に倣い、encoder と decoder を別ONNXに分けてエクスポートしておく
(export/sam2_onnx.py を参照). 推論はこの2つを順に流す:

    image (1024x1024 RGB)
        → [encoder.onnx]  → high_res_feats (2個) + image_embed
        → [decoder.onnx]  ← + box_prompt
        → low_res_masks   → 後処理でフル解像度マスク

ONNX Runtime は CUDAExecutionProvider + TensorrtExecutionProvider を順に試す.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# SAM2 はすべて 1024x1024 に揃えてから encoder に入れる
SAM2_INPUT_SIZE = 1024


def _imagenet_normalize(img01: np.ndarray) -> np.ndarray:
    """0..1 正規化済みの CHW に ImageNet 平均/標準偏差を適用."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return (img01 - mean) / std


def _resize_longest_side(image_rgb: np.ndarray, target: int) -> Tuple[np.ndarray, float]:
    """
    SAM と同様、長辺を target に揃えてアスペクト維持でリサイズし、
    右下を 0 でパディングして target x target 正方形にする.
    戻り値: (正方形画像, 使用されたスケール)
    """
    h, w = image_rgb.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((target, target, 3), dtype=image_rgb.dtype)
    padded[:nh, :nw] = resized
    return padded, scale


class SAM2ONNX:
    """
    SAM2 image predictor の ONNX Runtime 実装.
    TorchバックエンドとAPIを揃える (segment メソッドのシグネチャが同じ).
    """

    def __init__(self, cfg):
        import onnxruntime as ort

        enc_path = Path(getattr(cfg, "sam2_encoder_onnx",
                                "checkpoints/sam2_encoder.onnx"))
        dec_path = Path(getattr(cfg, "sam2_decoder_onnx",
                                "checkpoints/sam2_decoder.onnx"))
        if not enc_path.exists() or not dec_path.exists():
            raise FileNotFoundError(
                f"SAM2 ONNX が見つかりません: {enc_path}, {dec_path}\n"
                "python -m export.sam2_onnx で生成してください."
            )

        providers = self._select_providers(cfg)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.encoder = ort.InferenceSession(str(enc_path), so, providers=providers)
        self.decoder = ort.InferenceSession(str(dec_path), so, providers=providers)

        # encoder の出力名を記録 (エクスポート時の命名に合わせる)
        self.enc_output_names = [o.name for o in self.encoder.get_outputs()]
        # decoder の入力/出力名
        self.dec_input_names = [i.name for i in self.decoder.get_inputs()]
        self.dec_output_names = [o.name for o in self.decoder.get_outputs()]

    # ------------------------------------------------------------------
    @staticmethod
    def _select_providers(cfg) -> list:
        """TRT > CUDA > CPU の優先順で Providers を並べる."""
        import onnxruntime as ort
        avail = ort.get_available_providers()
        wanted: list = []

        # cfg.use_trt_ep が True かつ TRT EP が入っていれば最優先
        if getattr(cfg, "use_trt_ep", False) and "TensorrtExecutionProvider" in avail:
            wanted.append((
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": ".trt_cache",
                },
            ))
        if "CUDAExecutionProvider" in avail and getattr(cfg, "device", "cuda") != "cpu":
            wanted.append("CUDAExecutionProvider")
        wanted.append("CPUExecutionProvider")
        return wanted

    # ------------------------------------------------------------------
    def _encode(self, image_rgb: np.ndarray) -> Tuple[dict, float, Tuple[int, int]]:
        """
        画像をエンコーダに通す.
        戻り値: ({encoder outputs dict}, scale, (orig_h, orig_w))
        """
        orig_h, orig_w = image_rgb.shape[:2]
        padded, scale = _resize_longest_side(image_rgb, SAM2_INPUT_SIZE)
        chw = padded.astype(np.float32).transpose(2, 0, 1) / 255.0
        chw = _imagenet_normalize(chw)
        batch = chw[None, ...].astype(np.float32)

        outs = self.encoder.run(self.enc_output_names, {"image": batch})
        feats = dict(zip(self.enc_output_names, outs))
        return feats, scale, (orig_h, orig_w)

    # ------------------------------------------------------------------
    def segment(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """
        boxes_xyxy: (N, 4) の pixel 座標. N=0 の場合は空マスクを返す.
        戻り値: (H, W) uint8 の2値マスク.
        """
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return np.zeros(image_rgb.shape[:2], np.uint8)

        feats, scale, (orig_h, orig_w) = self._encode(image_rgb)

        # ボックスを 1024x1024 座標系へスケール
        boxes = boxes_xyxy.astype(np.float32) * scale  # (N, 4)
        # SAM2 decoder は point_coords + point_labels でボックスを受け付ける流儀
        # (top-left, bottom-right を2点として与え、labels=[2, 3])
        N = boxes.shape[0]
        point_coords = np.zeros((N, 2, 2), dtype=np.float32)
        point_coords[:, 0, 0] = boxes[:, 0]   # x1
        point_coords[:, 0, 1] = boxes[:, 1]   # y1
        point_coords[:, 1, 0] = boxes[:, 2]   # x2
        point_coords[:, 1, 1] = boxes[:, 3]   # y2
        point_labels = np.tile(np.array([[2, 3]], dtype=np.float32), (N, 1))

        # decoder に渡す dict. export 時の命名と揃える必要あり.
        dec_inputs = {
            "image_embed":      feats.get("image_embed"),
            "high_res_feats_0": feats.get("high_res_feats_0"),
            "high_res_feats_1": feats.get("high_res_feats_1"),
            "point_coords":     point_coords,
            "point_labels":     point_labels,
            "mask_input":       np.zeros((N, 1, 256, 256), np.float32),
            "has_mask_input":   np.zeros((N,), np.float32),
            "orig_im_size":     np.array([orig_h, orig_w], dtype=np.int64),
        }
        # 存在するキーのみ渡す (export 版によって入力名が少し違う場合あり)
        dec_inputs = {k: v for k, v in dec_inputs.items() if k in self.dec_input_names}

        outs = self.decoder.run(None, dec_inputs)
        # outs[0] は (N, 1, H', W') の logits.
        # SAM2 decoder の出力解像度は export 実装により 2 通りある:
        #   (a) orig_im_size を受け取って内部でリサイズ → (N, 1, orig_h, orig_w)
        #   (b) 1024x1024 の固定サイズ                → (N, 1, 1024, 1024)
        # 実際のサイズを見て分岐する.
        logits = outs[0]
        if logits.ndim == 4:
            logits = logits[:, 0]  # (N, H', W')

        out_h, out_w = logits.shape[-2], logits.shape[-1]
        at_orig = (out_h == orig_h and out_w == orig_w)

        valid_h = int(round(orig_h * scale))
        valid_w = int(round(orig_w * scale))
        binary: np.ndarray | None = None
        for i in range(logits.shape[0]):
            m = logits[i]
            if at_orig:
                # すでにオリジナル解像度. リサイズ不要.
                pass
            else:
                # 1024×1024 空間の mask と仮定. パディング解除→元解像度.
                if m.shape != (SAM2_INPUT_SIZE, SAM2_INPUT_SIZE):
                    m = cv2.resize(m, (SAM2_INPUT_SIZE, SAM2_INPUT_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
                m = m[:valid_h, :valid_w]
                m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            b = (m > 0.0).astype(np.uint8) * 255
            binary = b if binary is None else np.maximum(binary, b)

        return binary if binary is not None else np.zeros((orig_h, orig_w), np.uint8)
