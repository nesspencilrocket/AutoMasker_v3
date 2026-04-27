"""
SAM2 の PyTorch モデルを encoder / decoder の 2 つの ONNX に分割してエクスポートする.

使い方:
    python -m export.sam2_onnx \
        --cfg configs/sam2.1/sam2.1_hiera_l.yaml \
        --ckpt checkpoints/sam2.1_hiera_large.pt \
        --output-dir checkpoints \
        --opset 17

出力:
    checkpoints/sam2_encoder.onnx
    checkpoints/sam2_decoder.onnx

参考: tier4/sam2_pytorch2onnx の実装方針を踏襲.
  - repeat_interleave を tile に置換しないと TensorRT 変換時に転ける
  - 動的軸は decoder 側の "num_boxes" のみ
  - encoder は 1x3x1024x1024 固定で OK
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


class SAM2EncoderWrapper(nn.Module):
    """
    SAM2 の image encoder (Hiera + FPN) を単体でエクスポートするためのラッパ.
    forward(image) -> (image_embed, high_res_feats_0, high_res_feats_1)
    """

    def __init__(self, sam2):
        super().__init__()
        self.sam2 = sam2

    def forward(self, image: torch.Tensor):
        # sam2.forward_image は dict を返すので、整形して tuple にする
        backbone_out = self.sam2.forward_image(image)
        # SAM2 内部の典型的な key: 'vision_features', 'backbone_fpn'
        image_embed = backbone_out["vision_features"]
        feats = backbone_out["backbone_fpn"]
        # 下流の mask decoder が使うのは通常 feats[0], feats[1]
        return image_embed, feats[0], feats[1]


class SAM2DecoderWrapper(nn.Module):
    """
    image_embed + ボックス prompt → mask logits の変換.
    ONNX 化しやすいよう、point_coords/point_labels 形式でボックスを扱う
    (SAM2 の ImagePredictor の内部処理に合わせる).
    """

    def __init__(self, sam2):
        super().__init__()
        self.model = sam2
        self.mask_decoder = sam2.sam_mask_decoder
        self.prompt_encoder = sam2.sam_prompt_encoder

    def forward(
        self,
        image_embed: torch.Tensor,            # (1, C, 64, 64)
        high_res_feats_0: torch.Tensor,       # (1, C0, 256, 256)
        high_res_feats_1: torch.Tensor,       # (1, C1, 128, 128)
        point_coords: torch.Tensor,           # (N, 2, 2)
        point_labels: torch.Tensor,           # (N, 2)
        mask_input: torch.Tensor,             # (N, 1, 256, 256)
        has_mask_input: torch.Tensor,         # (N,)
    ):
        # prompt encoder はボックス2点 (top-left=2, bottom-right=3) を受け付ける
        sparse_emb, dense_emb = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=mask_input if has_mask_input.any() else None,
        )
        # image_embed を N 個に "tile" で複製 (repeat_interleave は TRT が嫌う)
        N = point_coords.shape[0]
        img_emb = image_embed.tile(N, 1, 1, 1)
        hr0 = high_res_feats_0.tile(N, 1, 1, 1)
        hr1 = high_res_feats_1.tile(N, 1, 1, 1)

        low_res_masks, iou_preds, _, _ = self.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[hr0, hr1],
        )
        # low_res_masks: (N, 1, 256, 256). 後段で拡大して使う.
        return low_res_masks, iou_preds


def _export_encoder(sam2, out_path: Path, opset: int, device: str):
    wrapper = SAM2EncoderWrapper(sam2).to(device).eval()
    dummy = torch.randn(1, 3, 1024, 1024, device=device)
    print(f"[encoder] exporting → {out_path}")
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["image_embed", "high_res_feats_0", "high_res_feats_1"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,  # fixed shape
    )


def _export_decoder(sam2, out_path: Path, opset: int, device: str):
    wrapper = SAM2DecoderWrapper(sam2).to(device).eval()
    # dummy shape. C は hiera によって異なるので、encoder 出力から取る
    with torch.no_grad():
        dummy_img = torch.randn(1, 3, 1024, 1024, device=device)
        enc = SAM2EncoderWrapper(sam2).to(device).eval()(dummy_img)
    image_embed, hr0, hr1 = enc

    N = 1  # dummy
    point_coords = torch.zeros(N, 2, 2, device=device)
    point_labels = torch.tensor([[2, 3]] * N, dtype=torch.float32, device=device)
    mask_input = torch.zeros(N, 1, 256, 256, device=device)
    has_mask_input = torch.zeros(N, device=device)

    print(f"[decoder] exporting → {out_path}")
    torch.onnx.export(
        wrapper,
        (image_embed, hr0, hr1, point_coords, point_labels, mask_input, has_mask_input),
        str(out_path),
        input_names=[
            "image_embed", "high_res_feats_0", "high_res_feats_1",
            "point_coords", "point_labels", "mask_input", "has_mask_input",
        ],
        output_names=["low_res_masks", "iou_predictions"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={
            "point_coords": {0: "num_boxes"},
            "point_labels": {0: "num_boxes"},
            "mask_input":   {0: "num_boxes"},
            "has_mask_input": {0: "num_boxes"},
            "low_res_masks": {0: "num_boxes"},
            "iou_predictions": {0: "num_boxes"},
        },
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="SAM2 model config (yaml path inside sam2 package)")
    ap.add_argument("--ckpt", required=True, help="SAM2 checkpoint (.pt)")
    ap.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--simplify", action="store_true",
                    help="onnxsim でグラフを簡約化する (精度影響なし, 推論高速化)")
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from sam2.build_sam import build_sam2
    print(f"[load] {args.cfg} / {args.ckpt}")
    sam2 = build_sam2(args.cfg, args.ckpt, device=args.device).eval()

    enc_path = args.output_dir / "sam2_encoder.onnx"
    dec_path = args.output_dir / "sam2_decoder.onnx"

    with torch.no_grad():
        _export_encoder(sam2, enc_path, args.opset, args.device)
        _export_decoder(sam2, dec_path, args.opset, args.device)

    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify
            for p in (enc_path, dec_path):
                print(f"[simplify] {p}")
                model = onnx.load(str(p))
                simplified, ok = simplify(model)
                if not ok:
                    raise RuntimeError(f"onnxsim failed on {p}")
                onnx.save(simplified, str(p))
        except ImportError:
            print("onnxsim が見つからないのでスキップ (pip install onnxsim)")

    print("\nDone. Next step: python -m export.build_trt --target sam2")


if __name__ == "__main__":
    sys.exit(main())
