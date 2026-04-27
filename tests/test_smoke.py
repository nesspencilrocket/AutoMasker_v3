"""
非ML部分 (I/O, Refinement, Config) の動作確認用スクリプト.
実行: python tests/test_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import cv2

# プロジェクトルートを import path へ
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automasker.config import Config
from automasker import io_utils
from automasker.refine import refine_mask


def test_write_mask_colmap_convention():
    """COLMAP向け出力: 対象=0 (消す), 背景=255 (保持) になっているか."""
    mask = np.zeros((100, 100), np.uint8)
    mask[30:70, 30:70] = 255  # 対象

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "img001.png"

        # デフォルト (invert=False) → 対象の場所は 0 になるはず
        io_utils.write_mask(out, mask, invert=False)
        written = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
        assert written[50, 50] == 0, "対象ピクセルは 0 (消す) でなくてはならない"
        assert written[5, 5] == 255, "非対象ピクセルは 255 (保持) でなくてはならない"
        print("  ✔ write_mask(invert=False) OK")

        # invert=True → 対象=255
        io_utils.write_mask(out, mask, invert=True)
        written = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
        assert written[50, 50] == 255
        assert written[5, 5] == 0
        print("  ✔ write_mask(invert=True) OK")


def test_mask_path_for():
    # Path comparison not string (Windows-compatible)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out_root = Path(td)
        p = out_root.parent / "images" / "IMG_0042.jpg"
        mp = io_utils.mask_path_for(p, out_root)
        assert mp == out_root / "IMG_0042.png", mp
        assert mp.suffix == ".png"
        print("  ✔ mask_path_for OK")


def test_mask_path_for_rejects_null_byte():
    """SECURITY: 入力に null バイトが含まれても output_root から脱出しない."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out_root = Path(td)
        # null byte を含むパス
        p = Path("foo\x00bar.jpg")
        result = io_utils.mask_path_for(p, out_root)
        # null byte が剥がれていること
        assert "\x00" not in str(result), f"null byte leaked: {result!r}"
        # output_root 配下に収まっていること
        result.resolve().relative_to(out_root.resolve())
        print(f"  ✔ null byte sanitized: {p.name!r} -> {result.name!r}")


def test_mask_path_for_rejects_path_separators():
    """stem に / や \\ が含まれていても output_root を脱出しない."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out_root = Path(td)
        # pathlib は "../x.jpg" を "../x" + ".jpg" に分けるので、
        # もう一段手前の悪意あるケースとして、stem 自体に区切りが入る異常ケースをテスト
        p = Path("../../etc/passwd.jpg")
        result = io_utils.mask_path_for(p, out_root)
        # Path.stem は最後のコンポーネントの拡張子なし部分のみ取るので、"passwd" になる
        assert result.name == "passwd.png", f"unexpected result: {result}"
        result.resolve().relative_to(out_root.resolve())
        print(f"  ✔ path traversal neutralized: {p!r} -> {result.name!r}")


def test_sanitize_stem_empty_becomes_underscore():
    """stem が全て危険文字で空になるケース."""
    from automasker.io_utils import _sanitize_stem
    assert _sanitize_stem("\x00\x01\x02") == "_"
    assert _sanitize_stem("") == "_"
    assert _sanitize_stem("normal") == "normal"
    print("  ✔ _sanitize_stem edge cases OK")


def test_list_images_sorted():
    cfg = Config()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name in ["c.jpg", "a.jpg", "b.png", "readme.txt"]:
            if name.endswith(".txt"):
                (td / name).write_text("hi")
            else:
                cv2.imwrite(str(td / name), np.zeros((4, 4, 3), np.uint8))
        files = io_utils.list_images(td, cfg)
        assert [p.name for p in files] == ["a.jpg", "b.png", "c.jpg"]
        print("  ✔ list_images sorted OK")


def test_refine_mask_dilate_and_minarea():
    m = np.zeros((200, 200), np.uint8)
    m[50:60, 50:60] = 255       # 大きな領域 (100 px)
    m[10:12, 10:12] = 255       # 小さなノイズ (4 px)

    out = refine_mask(m, dilate=0, erode=0, min_area=20, fill_holes=False)
    assert out[11, 11] == 0, "小さな領域は除去されるべき"
    assert out[55, 55] == 255
    print("  ✔ refine_mask(min_area) removes small components OK")

    out = refine_mask(m, dilate=5, erode=0, min_area=0, fill_holes=False)
    # 元の矩形は [50:60, 50:60]. 膨張5px (楕円カーネル) で少なくとも元+数px外側は白
    assert out[55, 48] == 255, "横方向へ膨張しているべき"
    assert out[48, 55] == 255, "縦方向へ膨張しているべき"
    # 元領域の外周より十分離れた場所は依然 0
    assert out[0, 0] == 0
    print("  ✔ refine_mask(dilate) expands OK")


def test_refine_mask_fill_holes():
    m = np.zeros((100, 100), np.uint8)
    cv2.rectangle(m, (20, 20), (80, 80), 255, -1)  # 塗りつぶし矩形
    cv2.rectangle(m, (45, 45), (55, 55), 0, -1)    # 中に穴
    out = refine_mask(m, fill_holes=True)
    assert out[50, 50] == 255, "穴は埋まるべき"
    print("  ✔ refine_mask(fill_holes) OK")


def main():
    print("AutoMasker smoke tests")
    test_mask_path_for()
    test_mask_path_for_rejects_null_byte()
    test_mask_path_for_rejects_path_separators()
    test_sanitize_stem_empty_becomes_underscore()
    test_list_images_sorted()
    test_write_mask_colmap_convention()
    test_refine_mask_dilate_and_minarea()
    test_refine_mask_fill_holes()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
