"""
ファイル I/O ユーティリティ。
COLMAP / 3D Gaussian Splatting 学習コードが期待する命名規則に合わせる。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .config import Config


def list_images(folder: Path, cfg: Config) -> List[Path]:
    """フォルダ内の画像ファイルをソート順で列挙する."""
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.suffix.lower() in cfg.image_exts]
    # COLMAP の命名 (IMG_0001, ..., IMG_9999) で並ぶよう辞書順ソート
    return sorted(files, key=lambda p: p.name)


def read_image(path: Path) -> np.ndarray:
    """BGR→RGB に変換して返す (SAM2 / GDINO が RGB を要求するため)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_mask(path: Path, mask: np.ndarray, invert: bool = False) -> None:
    """
    マスクを2値PNGで保存する。
    既定: 対象=0 (黒), 背景=255 (白) … 3DGS 学習で「背景のみ使う」場合の慣習。
    invert=True: 対象=255, 背景=0
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    if not invert:
        # 「対象を除去」したいので 対象=0 / 保持領域=255
        out = np.where(mask > 0, 0, 255).astype(np.uint8)
    else:
        # 対象のみ残す
        out = np.where(mask > 0, 255, 0).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


def _sanitize_stem(stem: str) -> str:
    """
    SECURITY: ファイル名に含まれうる危険文字を取り除く.
      - null byte (\\x00)        → filesystem によっては path truncation に使われる
      - パス区切り (/, \\)         → ディレクトリ脱出の原因になる
      - 制御文字 (0x01-0x1F)     → 一部端末/ログの破壊につながる
    実用上は stem が空になることは考えにくいが、空になる場合は "_" で置換する.
    """
    # path 区切り・null・制御文字を削除
    bad = set(range(0, 32))
    bad.update(ord(c) for c in ("/", "\\"))
    cleaned = "".join(ch for ch in stem if ord(ch) not in bad)
    return cleaned or "_"


def mask_path_for(image_path: Path, output_root: Path) -> Path:
    """image 名から mask パスを決める. 拡張子は必ず .png.

    SECURITY: stem を _sanitize_stem でクリーン化し、null byte / パス区切り等に
    よる output_root 脱出を防ぐ. さらに最終パスが output_root の下に収まることを
    念のため resolve 後に検証する.
    """
    safe_stem = _sanitize_stem(image_path.stem)
    out = output_root / (safe_stem + ".png")
    # 二重防衛: resolve 後の位置が output_root 配下であることを確認
    try:
        out.resolve().relative_to(output_root.resolve())
    except ValueError:
        raise ValueError(
            f"refusing to write mask outside output root: {out}"
        ) from None
    return out


def video_to_frames(
    video_path: Path,
    out_dir: Path,
    stride: int = 1,
    max_frames: int | None = None,
) -> Tuple[List[Path], float]:
    """
    動画をフレーム画像に展開する (SAM2 video predictor の入力形式に合わせる).
    戻り値: (frame_paths, fps)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        paths: List[Path] = []
        idx, saved = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                # SAM2 video predictor は連番 JPEG を期待する
                p = out_dir / f"{saved:06d}.jpg"
                cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                paths.append(p)
                saved += 1
                if max_frames is not None and saved >= max_frames:
                    break
            idx += 1
    finally:
        # ファイルハンドルを確実に解放 (Windows でのファイルロックを防ぐ)
        cap.release()
    return paths, fps


def combine_masks(masks: Iterable[np.ndarray]) -> np.ndarray:
    """複数の2値マスクを OR で合成する."""
    out = None
    for m in masks:
        if m is None:
            continue
        b = (m > 0).astype(np.uint8) * 255
        out = b if out is None else np.maximum(out, b)
    return out if out is not None else np.zeros((1, 1), np.uint8)
