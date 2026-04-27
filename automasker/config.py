"""
設定値の一元管理。チェックポイントのパスや既定の閾値を定義する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# プロジェクトルート (このファイルの2つ上)
ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"


@dataclass
class Config:
    # --- モデル / チェックポイント ---------------------------------
    # SAM2 の Hiera-Large。VRAM が少ない場合は hiera_s (Small) に変更
    sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_ckpt: str = str(CHECKPOINT_DIR / "sam2.1_hiera_large.pt")

    # SAM2 ONNX / TensorRT (export スクリプトで生成)
    sam2_encoder_onnx: str = str(CHECKPOINT_DIR / "sam2_encoder.onnx")
    sam2_decoder_onnx: str = str(CHECKPOINT_DIR / "sam2_decoder.onnx")
    sam2_encoder_engine: str = str(CHECKPOINT_DIR / "sam2_encoder.engine")
    sam2_decoder_engine: str = str(CHECKPOINT_DIR / "sam2_decoder.engine")

    # Grounding DINO (pip版: groundingdino-py)
    gdino_cfg: str = str(CHECKPOINT_DIR / "GroundingDINO_SwinT_OGC.py")
    gdino_ckpt: str = str(CHECKPOINT_DIR / "groundingdino_swint_ogc.pth")
    gdino_onnx: str = str(CHECKPOINT_DIR / "gdino.onnx")
    gdino_engine: str = str(CHECKPOINT_DIR / "gdino.engine")
    gdino_tokenizer: str = "bert-base-uncased"
    gdino_max_text_len: int = 256

    # --- バックエンド選択 ---------------------------------------
    # "torch" / "onnx" / "trt" を個別に切替可能
    backend_detector: str = "torch"
    backend_segmenter: str = "torch"
    # onnxruntime の TensorRT ExecutionProvider を使うか
    use_trt_ep: bool = False

    # --- 推論パラメータ -------------------------------------------
    box_threshold: float = 0.30      # Grounding DINO のボックス信頼度
    text_threshold: float = 0.25     # Grounding DINO のテキスト信頼度
    mask_dilate: int = 3             # モルフォロジー膨張のカーネル半径 (px)
    mask_erode: int = 0              # 収縮のカーネル半径 (px)
    mask_min_area: int = 128         # 小さいマスクを除去する面積閾値 (px^2)

    # --- I/O ------------------------------------------------------
    image_exts: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    video_exts: tuple = (".mp4", ".mov", ".avi", ".mkv")

    # --- ランタイム ----------------------------------------------
    device: str = field(default_factory=lambda: _auto_device())
    # True のとき "対象を残す" / False で "対象を消す" (COLMAP用は通常 False)
    invert: bool = False


def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# GUIで編集可能なパラメータのメタ情報 (name, min, max, step)
TUNABLE_PARAMS = [
    ("box_threshold", 0.05, 0.95, 0.01),
    ("text_threshold", 0.05, 0.95, 0.01),
    ("mask_dilate", 0, 30, 1),
    ("mask_erode", 0, 30, 1),
]
