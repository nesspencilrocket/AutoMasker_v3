"""
mask_lift: 2D マスク集合を 3D Gaussian Splatting シーンへリフト.

本モジュールは、2D セグメンテーションマスクと学習済み 3DGS から、
各 Gaussian のオブジェクトラベルを閉形式で決定する方法を実装する.

アルゴリズムの核心は Shen et al., ECCV 2024 の論文 "FlashSplat" にある
閉形式ラベル割当 (論文の式 2):

    l_i = 1  iff  Σ_{v,p} W_{i,v,p} (2 M_{v,p} - 1)  >  λ Σ_{v,p} W_{i,v,p}

ここで W_{i,v,p} はビュー v のピクセル p への Gaussian i の寄与.
論文: https://arxiv.org/abs/2409.08270
論文本体のアイデアへの credit は著者に帰属する.

## 本実装について
本パッケージは**論文のアルゴリズム (数式) を独自に再実装**したものであり、
論文著者の公式実装 (https://github.com/florinshen/FlashSplat) からの
コード流用は一切ありません. 公式実装は INRIA の 3DGS ベース研究ライセンス
(非商用・研究評価用途のみ) 下にあるため、本実装はそれとは独立です.

本パッケージのライセンスは親プロジェクトと同じく MIT です.
ただし、論文のアルゴリズム自体を商用利用する場合は、
FlashSplat 論文著者および INRIA 研究ライセンスの有無を各自ご確認ください.
"""
from .lift import lift_masks_to_gaussians, MaskLiftResult

__all__ = ["lift_masks_to_gaussians", "MaskLiftResult"]
