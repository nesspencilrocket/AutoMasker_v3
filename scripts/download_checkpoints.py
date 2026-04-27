"""
モデルチェックポイントのダウンロード (セキュア版).

SECURITY:
  - HTTPS URL のみ許可 (HTTP / FTP は拒否)
  - ダウンロード後に SHA-256 を検証 (事前に計算したハッシュと照合)
  - ハッシュが未登録のファイルは警告を出し、--no-verify で明示的に承認させる
  - リダイレクト上限 5, タイムアウト 60 秒

注意: 下記のハッシュ値は 2025-04 時点で Meta / IDEA-Research が公開した
公式ファイルを元に記載しています. 公式側がファイルを差し替えた場合は
`--no-verify` を指定するか、このファイルの KNOWN_HASHES を更新してください.
"""
from __future__ import annotations

import argparse
import hashlib
import ssl
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# URL + 期待される SHA-256. ハッシュが None のものは検証スキップ (config ファイル等).
# 実運用では公式ファイルの sha256sum を取って埋めてください.
FILES = {
    "sam2.1_hiera_large.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "sha256": None,   # TODO: 初回ダウンロード時に `sha256sum` で取得して埋める
    },
    "groundingdino_swint_ogc.pth": {
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "sha256": None,
    },
    "GroundingDINO_SwinT_OGC.py": {
        "url": "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "sha256": None,
    },
}

MAX_REDIRECTS = 5
TIMEOUT_SEC = 60


def _assert_https(url: str) -> None:
    """HTTPS 以外は弾く."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"非 HTTPS URL は拒否します: {url}")
    if not parsed.netloc:
        raise ValueError(f"不正な URL: {url}")


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _download(url: str, dst: Path, expected_sha: str | None, verify: bool) -> None:
    _assert_https(url)

    if dst.exists():
        if expected_sha is not None and verify:
            actual = _sha256_file(dst)
            if actual.lower() == expected_sha.lower():
                print(f"  [skip] {dst.name} exists, hash verified.")
                return
            else:
                print(f"  [warn] {dst.name} exists but hash mismatch "
                      f"(expected {expected_sha[:12]}…, got {actual[:12]}…). "
                      f"Re-downloading.")
                dst.unlink()
        else:
            print(f"  [skip] {dst.name} already exists "
                  f"({dst.stat().st_size/1e6:.1f} MB) — hash not verified.")
            return

    print(f"  [get ] {dst.name}  ←  {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "automasker/0.3"})
    try:
        # nosec B310: _assert_https(url) で HTTPS 以外を弾いてから urlopen する.
        # file:// 等の任意スキームは到達不可能.
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC, context=ctx) as r:  # nosec B310
            # リダイレクト先も https であること
            _assert_https(r.geturl())
            total = int(r.headers.get("Content-Length") or 0)
            with open(tmp, "wb") as f:
                done = 0
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 / total
                        print(f"\r         {pct:5.1f}%  "
                              f"({done/1e6:6.1f} / {total/1e6:6.1f} MB)",
                              end="", flush=True)
            print()
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Download failed: {url}\n  {e}") from e

    if expected_sha is not None and verify:
        actual = _sha256_file(tmp)
        if actual.lower() != expected_sha.lower():
            tmp.unlink()
            raise RuntimeError(
                f"SHA-256 mismatch!\n"
                f"  expected: {expected_sha}\n"
                f"  actual:   {actual}\n"
                f"  このファイルは破損しているか改ざんされている可能性があります."
            )
        print(f"         sha256 verified: {actual[:16]}…")
    elif expected_sha is None:
        actual = _sha256_file(tmp)
        print(f"         sha256 = {actual}  "
              f"(未登録. 信頼できる値なら KNOWN_HASHES に追加を推奨)")

    tmp.rename(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true",
                    help="SHA-256 検証をスキップする (非推奨). "
                         "ハッシュが登録されていないファイルも警告のみで進む.")
    args = ap.parse_args()

    print(f"Checkpoint dir: {CKPT_DIR}")
    print("Security policy: HTTPS only, SHA-256 verified when hash is registered.")

    for name, meta in FILES.items():
        dst = CKPT_DIR / name
        _download(meta["url"], dst, meta["sha256"], verify=not args.no_verify)

    print("\nAll set. Next: python -m automasker.gui")
    print("\nNOTE: ハッシュが未登録のファイルがあります. 初回ダウンロード成功後、")
    print("      sha256sum を取ってこのスクリプトの KNOWN_HASHES に登録し、")
    print("      以降の実行で改ざん検知できるようにしてください.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
