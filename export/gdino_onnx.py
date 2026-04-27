"""
Grounding DINO の ONNX エクスポート.

IDEA-Research 本家の `demo/export_openvino.py` / `demo/export_onnx.py` に
近い手順で、以下の入力を持つ単一 ONNX として保存する:
    input_ids: (1, L) int64
    attention_mask: (1, L) int64
    token_type_ids: (1, L) int64
    image: (1, 3, H, W) float32

出力:
    boxes_cxcywh: (1, num_queries, 4)
    logits:       (1, num_queries, L)

使い方:
    python -m export.gdino_onnx \
        --cfg checkpoints/GroundingDINO_SwinT_OGC.py \
        --ckpt checkpoints/groundingdino_swint_ogc.pth \
        --output checkpoints/gdino.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


class GDINOExportWrapper(nn.Module):
    """
    Grounding DINO 本体は内部で HF tokenizer を呼ぶが、それは ONNX 化できない.
    そこで tokenize 済みテンソルを受け取る形に整形する.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, input_ids, attention_mask, token_type_ids):
        # groundingdino.models.GroundingDINO.forward に合わせた呼び出し
        # 実装差を吸収するため、samples/captions 両方の呼び出しを試みる
        samples = image
        text_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        # 本家実装は captions (list[str]) を取るが、ONNX 化のために
        # 内部 API の forward(samples, text_dict) をそのまま叩く.
        # この部分は GroundingDINO の実装バージョンに強く依存するため、
        # 失敗した場合は model.forward の代わりに
        # model.transformer.forward を直接呼ぶように書き換える必要がある.
        out = self.model(samples, text_dict=text_dict)
        # out の代表 key: 'pred_logits' (N,Q,L), 'pred_boxes' (N,Q,4)
        return out["pred_boxes"], out["pred_logits"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", type=Path, default=Path("checkpoints/gdino.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-text-len", type=int, default=256)
    args = ap.parse_args(argv)

    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.models import build_model

    print(f"[load] {args.cfg} / {args.ckpt}")
    slcfg = SLConfig.fromfile(args.cfg)
    slcfg.device = args.device
    model = build_model(slcfg)
    # SECURITY: weights_only=True で pickle RCE を防ぐ.
    # Grounding DINO の公式 .pth は state dict のみなのでこれで読める.
    # もし古い checkpoint で失敗したら、その checkpoint を信頼できる場合のみ
    # weights_only=False に変更すること (その場合は必ずハッシュ検証を行う).
    try:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError(
            f"チェックポイント読み込みに失敗: {e}\n"
            "torch.load(weights_only=True) で読める形式である必要があります. "
            "信頼できないソースからダウンロードした .pth は使わないでください."
        ) from e
    model.load_state_dict(clean_state_dict(state["model"]), strict=False)
    model = model.to(args.device).eval()

    wrapper = GDINOExportWrapper(model).to(args.device).eval()

    # dummy inputs
    dummy_img = torch.randn(1, 3, 800, 1200, device=args.device)
    L = args.max_text_len
    input_ids = torch.zeros(1, L, dtype=torch.long, device=args.device)
    attn = torch.ones(1, L, dtype=torch.long, device=args.device)
    tti = torch.zeros(1, L, dtype=torch.long, device=args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] → {args.output}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_img, input_ids, attn, tti),
            str(args.output),
            input_names=["image", "input_ids", "attention_mask", "token_type_ids"],
            output_names=["boxes_cxcywh", "logits"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes={
                "image": {2: "H", 3: "W"},
                "input_ids": {1: "L"},
                "attention_mask": {1: "L"},
                "token_type_ids": {1: "L"},
                "logits": {2: "L"},
            },
        )
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
