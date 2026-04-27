"""
ERP 画像のマスク生成パイプライン.

各 tangent view で ImagePipeline を呼び、得られた perspective マスクを
逆射影で ERP に合成する. 合成は per-pixel の加重平均 (weight: 視野中央重視).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ..config import Config
from ..pipeline import ImagePipeline
from ..refine import refine_mask
from .projection import (
    ErpSpec,
    ViewSpec,
    cubemap_views,
    erp_to_perspective,
    perspective_mask_to_erp,
    tangent_views,
)


@dataclass
class ErpMaskOptions:
    mode: str = "tangent"          # "tangent" or "cubemap"
    n_yaw: int = 8
    n_pitch: int = 3
    fov_deg: float = 90.0
    patch_size: int = 1024
    vote_threshold: float = 0.5    # 合成後の連続値を 2値化する閾値


class ErpMaskPipeline:
    """
    ERP 画像 1 枚に対して、
        1. 複数の tangent view を切り出し
        2. 各 view で detect + segment (ImagePipeline に委譲)
        3. 逆射影 + 加重平均で ERP 全天マスクに合成
        4. (任意) 2値化 + refinement
    """

    def __init__(self, cfg: Config, opts: ErpMaskOptions | None = None):
        self.cfg = cfg
        self.opts = opts or ErpMaskOptions()
        self.inner = ImagePipeline(cfg)
        self.views: List[ViewSpec] = self._make_views()

    def _make_views(self) -> List[ViewSpec]:
        if self.opts.mode == "cubemap":
            return cubemap_views(self.opts.patch_size, self.opts.fov_deg)
        return tangent_views(
            n_yaw=self.opts.n_yaw,
            n_pitch=self.opts.n_pitch,
            fov_deg=self.opts.fov_deg,
            width=self.opts.patch_size,
            height=self.opts.patch_size,
        )

    # ------------------------------------------------------------------
    def run_erp_image(
        self,
        erp_rgb: np.ndarray,
        prompt: str,
        save_patches_dir: Optional[Path] = None,
    ) -> np.ndarray:
        """
        ERP RGB (H, W, 3) uint8 → ERP 2値マスク (H, W) uint8 (対象=255).
        save_patches_dir を与えれば各ビューの perspective 画像とマスクもダンプする.
        """
        erp_spec = ErpSpec.from_image(erp_rgb)
        accum = np.zeros((erp_spec.height, erp_spec.width), dtype=np.float32)
        weight = np.zeros_like(accum)

        for i, view in enumerate(self.views):
            patch = erp_to_perspective(erp_rgb, view)
            # ImagePipeline.run_single は path から読むので、メモリ上の実行 API を使う
            det = self.inner.detector.detect(
                patch, prompt,
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
            )
            if len(det) == 0:
                persp_mask = np.zeros(patch.shape[:2], np.uint8)
            else:
                boxes = np.stack([d.box_xyxy for d in det], axis=0)
                persp_mask = self.inner.segmenter.segment(patch, boxes)
                persp_mask = refine_mask(
                    persp_mask,
                    dilate=self.cfg.mask_dilate,
                    erode=self.cfg.mask_erode,
                    min_area=self.cfg.mask_min_area,
                )

            if save_patches_dir is not None:
                save_patches_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_patches_dir / f"{i:02d}_{view.label()}_img.jpg"),
                            cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_patches_dir / f"{i:02d}_{view.label()}_mask.png"),
                            persp_mask)

            erp_m, erp_w = perspective_mask_to_erp(persp_mask, view, erp_spec)
            accum += erp_m * erp_w
            weight += erp_w

        # weight == 0 の領域 (どのビューからも見えない = 極付近のみなど) は 0
        avg = np.where(weight > 1e-6, accum / np.maximum(weight, 1e-6), 0.0)
        binary = (avg >= self.opts.vote_threshold).astype(np.uint8) * 255

        # 極近傍の継ぎ目を軽く smoothen
        binary = refine_mask(
            binary,
            dilate=max(self.cfg.mask_dilate, 1),
            erode=0,
            min_area=self.cfg.mask_min_area,
            fill_holes=True,
        )
        return binary

    # ------------------------------------------------------------------
    def run_folder(
        self,
        input_dir: Path,
        output_dir: Path,
        prompt: str,
        progress=None,
        save_patches: bool = False,
    ) -> int:
        """フォルダ丸ごと処理."""
        from .. import io_utils
        images = io_utils.list_images(input_dir, self.cfg)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, p in enumerate(images):
            erp = io_utils.read_image(p)
            debug_dir = (output_dir / "_debug" / p.stem) if save_patches else None
            binary = self.run_erp_image(erp, prompt, save_patches_dir=debug_dir)

            out = io_utils.mask_path_for(p, output_dir)
            io_utils.write_mask(out, binary, invert=self.cfg.invert)
            if progress:
                progress(i + 1, len(images), p)
        return len(images)
