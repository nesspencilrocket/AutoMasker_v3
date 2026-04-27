"""
360 度 (Equirectangular / ERP) 画像対応.

ERP 画像はそのまま SAM2 / Grounding DINO に通すと、極付近の強い歪みが原因で
検出とマスクの両方が劣化する. 本パッケージは以下の2方式で問題を緩和する:

    1) cubemap 分割 (6面 + オプションで 18面のオーバーラップ)
    2) tangent-plane パッチ (yaw/pitch のグリッドに perspective 画像として切り出し)

各パッチで推論して、最後に逆射影で ERP に合成し、2値マスクを得る.
"""
from .projection import (
    ErpSpec,
    erp_to_perspective,
    perspective_mask_to_erp,
    cubemap_views,
    tangent_views,
    ViewSpec,
)

__all__ = [
    "ErpSpec",
    "ViewSpec",
    "erp_to_perspective",
    "perspective_mask_to_erp",
    "cubemap_views",
    "tangent_views",
    "ErpMaskPipeline",
]


def __getattr__(name):
    """ErpMaskPipeline は重い依存 (tqdm 等) を引くので遅延読み込み."""
    if name == "ErpMaskPipeline":
        from .pipeline import ErpMaskPipeline as _E
        return _E
    raise AttributeError(name)
