"""
SAM2 を TensorRT Engine で直接推論するバックエンド.

ONNX Runtime の TRT ExecutionProvider よりもオーバーヘッドが少なく、
純粋に TensorRT + CUDA Stream で回せるので最速. ただし engine ファイルを
事前にビルドしておく必要がある (export/build_trt.py 参照).

依存: `tensorrt` (NVIDIA製, pip install tensorrt) と `pycuda` (pip install pycuda)
これらが無い環境では SAM2ONNX にフォールバックさせるのが無難.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from .sam2_onnx import (
    SAM2_INPUT_SIZE,
    _resize_longest_side,
    _imagenet_normalize,
)


class _TRTRunner:
    """
    TensorRT engine を 1 つロードして、名前付きの入出力バッファで run する薄いラッパ.
    動的シェイプ対応 (decoder のボックス数 N がフレーム毎に変わるため).
    """

    def __init__(self, engine_path: Path):
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401  (side effect: CUDA context init)
        import pycuda.driver as cuda

        self._trt = trt
        self._cuda = cuda

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # 入出力テンソル名を列挙
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        trt = self._trt
        cuda = self._cuda

        # 1) 入力シェイプを確定し、HostのバッファをDeviceへコピー
        dev_ptrs: Dict[str, int] = {}
        host_outs: Dict[str, np.ndarray] = {}

        for name, arr in inputs.items():
            arr = np.ascontiguousarray(arr)
            self.context.set_input_shape(name, arr.shape)
            dev = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod_async(dev, arr, self.stream)
            self.context.set_tensor_address(name, int(dev))
            dev_ptrs[name] = dev

        # 2) 出力バッファを確保
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.empty(shape, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            dev_ptrs[name] = dev
            host_outs[name] = host

        # 3) 実行
        self.context.execute_async_v3(self.stream.handle)

        # 4) 出力を Host へ戻す
        for name in self.output_names:
            cuda.memcpy_dtoh_async(host_outs[name], dev_ptrs[name], self.stream)
        self.stream.synchronize()

        return host_outs


class SAM2TRT:
    """SAM2ONNX と同じ API を提供する TensorRT 版."""

    def __init__(self, cfg):
        enc = Path(getattr(cfg, "sam2_encoder_engine",
                           "checkpoints/sam2_encoder.engine"))
        dec = Path(getattr(cfg, "sam2_decoder_engine",
                           "checkpoints/sam2_decoder.engine"))
        if not enc.exists() or not dec.exists():
            raise FileNotFoundError(
                f"TensorRT engine が見つかりません: {enc}, {dec}\n"
                "python -m export.build_trt で生成してください."
            )
        self.encoder = _TRTRunner(enc)
        self.decoder = _TRTRunner(dec)

    # ------------------------------------------------------------------
    def _encode(self, image_rgb: np.ndarray) -> Tuple[dict, float, Tuple[int, int]]:
        orig_h, orig_w = image_rgb.shape[:2]
        padded, scale = _resize_longest_side(image_rgb, SAM2_INPUT_SIZE)
        chw = padded.astype(np.float32).transpose(2, 0, 1) / 255.0
        chw = _imagenet_normalize(chw)
        batch = chw[None, ...].astype(np.float32)
        outs = self.encoder.run({"image": batch})
        return outs, scale, (orig_h, orig_w)

    # ------------------------------------------------------------------
    def segment(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return np.zeros(image_rgb.shape[:2], np.uint8)

        feats, scale, (orig_h, orig_w) = self._encode(image_rgb)
        boxes = boxes_xyxy.astype(np.float32) * scale
        N = boxes.shape[0]

        point_coords = np.zeros((N, 2, 2), dtype=np.float32)
        point_coords[:, 0, 0] = boxes[:, 0]
        point_coords[:, 0, 1] = boxes[:, 1]
        point_coords[:, 1, 0] = boxes[:, 2]
        point_coords[:, 1, 1] = boxes[:, 3]
        point_labels = np.tile(np.array([[2, 3]], dtype=np.float32), (N, 1))

        inputs = {
            "image_embed":      feats.get("image_embed"),
            "high_res_feats_0": feats.get("high_res_feats_0"),
            "high_res_feats_1": feats.get("high_res_feats_1"),
            "point_coords":     point_coords,
            "point_labels":     point_labels,
            "mask_input":       np.zeros((N, 1, 256, 256), np.float32),
            "has_mask_input":   np.zeros((N,), np.float32),
            "orig_im_size":     np.array([orig_h, orig_w], dtype=np.int64),
        }
        inputs = {k: v for k, v in inputs.items() if k in self.decoder.input_names}

        outs = self.decoder.run(inputs)
        # 出力名は 'masks' or 'low_res_masks' をそのまま使う
        mask_key = next(iter(outs))
        logits = outs[mask_key]
        if logits.ndim == 4:
            logits = logits[:, 0]

        # SAM2 の export によって出力サイズが異なる (sam2_onnx.py と同様の分岐)
        out_h, out_w = logits.shape[-2], logits.shape[-1]
        at_orig = (out_h == orig_h and out_w == orig_w)

        valid_h = int(round(orig_h * scale))
        valid_w = int(round(orig_w * scale))
        binary: np.ndarray | None = None
        for i in range(logits.shape[0]):
            m = logits[i]
            if at_orig:
                pass
            else:
                if m.shape != (SAM2_INPUT_SIZE, SAM2_INPUT_SIZE):
                    m = cv2.resize(m, (SAM2_INPUT_SIZE, SAM2_INPUT_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
                m = m[:valid_h, :valid_w]
                m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            b = (m > 0.0).astype(np.uint8) * 255
            binary = b if binary is None else np.maximum(binary, b)
        return binary if binary is not None else np.zeros((orig_h, orig_w), np.uint8)
