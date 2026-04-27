"""
マスクの後処理 (Refinement).
SAM2 の生マスクは境界が甘かったり、小さなノイズが出たりすることがあるので
モルフォロジー演算と小面積除去で整形する。オプションで GrabCut も掛けられる。
"""
from __future__ import annotations

import cv2
import numpy as np


def _kernel(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    size = 2 * r + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def refine_mask(
    mask: np.ndarray,
    dilate: int = 0,
    erode: int = 0,
    min_area: int = 0,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    2値マスクを整形する.
    dilate > 0  -> 境界を外側へ広げる (対象を少し大きめに消したいとき便利)
    erode > 0   -> 内側へ縮める
    min_area    -> その面積以下の連結成分を消す
    fill_holes  -> マスク内部の穴を埋める
    """
    m = (mask > 0).astype(np.uint8) * 255

    if erode > 0:
        m = cv2.erode(m, _kernel(erode), iterations=1)
    if dilate > 0:
        m = cv2.dilate(m, _kernel(dilate), iterations=1)

    if fill_holes:
        # m: 前景=255 / 背景=0 / 穴(内部の背景)=0
        # inv: 前景=0 / 外側背景=255 / 穴=255
        # 外側背景だけ 0 で埋めると、残った 255 がちょうど「内部の穴」になる.
        # ただし (0,0) が前景の場合もあるので、端を 1px 外側へパディングしてから行う.
        h, w = m.shape
        padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        inv = cv2.bitwise_not(padded)  # 外側も含めて背景=255
        ff_mask = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), np.uint8)
        # (0,0) は必ず外側背景 (255) なので、そこから 0 で flood-fill
        cv2.floodFill(inv, ff_mask, (0, 0), 0)
        # この時点で inv に残る 255 は内部の穴だけ
        holes = inv[1:h + 1, 1:w + 1]
        m = cv2.bitwise_or(m, holes)

    if min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = np.zeros_like(m)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        m = out

    return m


def grabcut_refine(image_rgb: np.ndarray, mask: np.ndarray, iters: int = 3) -> np.ndarray:
    """
    SAMマスクを初期値として GrabCut を適用し、境界をより正確にする.
    重い処理なのでプレビュー中は呼ばず、Export時のみ使う想定.
    """
    h, w = mask.shape[:2]
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    m = (mask > 0).astype(np.uint8)
    gc_mask[m == 1] = cv2.GC_PR_FGD

    # マスクの内側を強制 FGD にして収束を早める
    eroded = cv2.erode(m, _kernel(5), iterations=1)
    gc_mask[eroded == 1] = cv2.GC_FGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, gc_mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # マスクが空 or 全面の場合は grabCut が失敗するので元を返す
        return (m * 255).astype(np.uint8)

    out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0)
    return out.astype(np.uint8)
