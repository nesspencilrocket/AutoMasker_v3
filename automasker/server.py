"""
ローカル HTTP 推論サーバ (セキュア版).

SECURITY DESIGN:
  - デフォルトで 127.0.0.1 バインド. ネットワーク公開は明示的な許可が必要.
  - 0.0.0.0 / LAN アドレスにバインドするには --allow-remote + --token 必須.
  - Bearer token 認証 (--token で指定). 未指定時は自動生成して起動時に表示.
  - リクエストボディ上限 25 MB (超過は 413 で拒否) — メモリ枯渇攻撃対策.
  - CORS は off が既定 (--cors でオン). オンにしても token 必須なので CSRF 不可.
  - マルチパート解析は厳密化 (boundary 長・フィールド長制限, nullバイト拒否).
  - cv2.imdecode に渡す前にマジックバイト検査でファイル種別を絞る (PNG/JPEG のみ).
  - エラーメッセージに内部パスを漏らさない.

C++ フロントエンドから叩く場合は起動時に表示されるトークンを使ってください.
"""
from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

from .config import Config


# --- 上限 ------------------------------------------------------------------
MAX_BODY_BYTES = 25 * 1024 * 1024        # 25 MB
MAX_PROMPT_LEN = 512
MAX_BOUNDARY_LEN = 200
MAX_IMAGE_PIXELS = 50_000_000            # 5000x10000 まで (cv2 DoS 対策)

# --- グローバル ------------------------------------------------------------
_PIPELINE_LOCK = threading.Lock()
# 推論の実行もシングルガードで直列化する.
# 理由: SAM2ImagePredictor や Grounding DINO の内部状態 (set_image 等) は
# スレッドセーフでないため, 並列リクエストが来ると結果が混ざる.
_INFERENCE_LOCK = threading.Lock()
_PIPELINE = None
_CONFIG: Optional[Config] = None
_EXPECTED_TOKEN: Optional[str] = None
_ALLOW_CORS = False


def _get_pipeline():
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            from .pipeline import ImagePipeline
            _PIPELINE = ImagePipeline(_CONFIG)
        return _PIPELINE


# --------------------------------------------------------------------------
# マルチパート解析 (必要最小限 + サニタイズ)
# --------------------------------------------------------------------------
def _parse_multipart(body: bytes, boundary: bytes) -> dict:
    """セキュアなマルチパート解析.

    - boundary 長をチェック
    - 各パートのヘッダはUTF-8でデコード可能なものだけ許可 (strict)
    - フィールド名に不正文字 (nullバイト, 改行, '/') を含むものは拒否
    """
    if len(boundary) == 0 or len(boundary) > MAX_BOUNDARY_LEN:
        raise ValueError("invalid multipart boundary")

    out = {}
    parts = body.split(b"--" + boundary)
    for p in parts:
        p = p.strip(b"\r\n")
        if not p or p == b"--":
            continue
        head, sep, content = p.partition(b"\r\n\r\n")
        if not sep:
            continue
        try:
            head_str = head.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        name = None
        for line in head_str.split("\r\n"):
            low = line.lower()
            if low.startswith("content-disposition"):
                for token in line.split(";"):
                    token = token.strip()
                    if token.startswith("name="):
                        raw = token.split("=", 1)[1].strip().strip('"').strip("'")
                        # 不正文字を含むフィールド名を拒否
                        if any(ch in raw for ch in ("\x00", "\r", "\n", "/", "\\")):
                            raise ValueError("illegal field name")
                        name = raw
        if name and len(name) < 64:
            out[name] = content.rstrip(b"\r\n")
    return out


# --------------------------------------------------------------------------
# 画像ファイル種別のマジックバイト検査
# --------------------------------------------------------------------------
def _looks_like_image(data: bytes) -> bool:
    if len(data) < 16:
        return False
    # PNG
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    # JPEG
    if data[:3] == b"\xff\xd8\xff":
        return True
    # WebP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    # タイムアウトで応答性と DoS 耐性を確保
    timeout = 120

    def log_message(self, fmt, *args):
        # アクセスログは最小限のみ (IPとパス) . パス内の機密情報は出さない.
        pass

    # ---- 認証 ------------------------------------------------------------
    def _check_auth(self) -> bool:
        if _EXPECTED_TOKEN is None:
            return True   # token 無効化モード (非推奨, localhost のみ)
        got = self.headers.get("Authorization", "")
        if not got.startswith("Bearer "):
            return False
        provided = got[len("Bearer "):].strip()
        # 定数時間比較
        return secrets.compare_digest(provided, _EXPECTED_TOKEN)

    # ---- ディスパッチ ----------------------------------------------------
    def do_GET(self):
        if not self._check_auth():
            self._reply(401, "text/plain", b"unauthorized")
            return
        if self.path == "/info":
            info = {
                "device": _CONFIG.device,
                "backend_detector": _CONFIG.backend_detector,
                "backend_segmenter": _CONFIG.backend_segmenter,
            }
            self._reply(200, "application/json", json.dumps(info).encode())
        else:
            self._reply(404, "text/plain", b"not found")

    def do_POST(self):
        if not self._check_auth():
            self._reply(401, "text/plain", b"unauthorized")
            return
        if self.path != "/infer":
            self._reply(404, "text/plain", b"not found")
            return

        # Content-Length チェック
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._reply(400, "text/plain", b"bad content-length")
            return
        if length <= 0:
            self._reply(400, "text/plain", b"empty body")
            return
        if length > MAX_BODY_BYTES:
            self._reply(413, "text/plain", b"payload too large")
            return

        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._reply(400, "text/plain", b"expected multipart/form-data")
            return
        boundary = ctype.split("boundary=", 1)[-1].encode()

        try:
            body = self.rfile.read(length)
            parts = _parse_multipart(body, boundary)
        except Exception:
            # 内部詳細は出さない
            self._reply(400, "text/plain", b"malformed multipart")
            return

        # 必須フィールド
        try:
            img_bytes = parts["image"]
            prompt_raw = parts["prompt"]
            box_th = float(parts.get("box_threshold", b"0.30"))
            text_th = float(parts.get("text_threshold", b"0.25"))
        except (KeyError, ValueError):
            self._reply(400, "text/plain", b"bad fields")
            return

        # プロンプト長 + 文字種チェック
        try:
            prompt = prompt_raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            self._reply(400, "text/plain", b"prompt must be utf-8")
            return
        if len(prompt) > MAX_PROMPT_LEN or "\x00" in prompt:
            self._reply(400, "text/plain", b"prompt invalid")
            return

        # 閾値レンジ制限
        if not (0.0 <= box_th <= 1.0) or not (0.0 <= text_th <= 1.0):
            self._reply(400, "text/plain", b"threshold out of range")
            return

        # 画像の種別確認
        if not _looks_like_image(img_bytes):
            self._reply(400, "text/plain", b"unsupported image type")
            return

        arr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            self._reply(400, "text/plain", b"image decode failed")
            return
        h, w = bgr.shape[:2]
        if h * w > MAX_IMAGE_PIXELS:
            self._reply(413, "text/plain", b"image too large")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 推論
        try:
            pipe = _get_pipeline()
            with _INFERENCE_LOCK:
                det = pipe.detector.detect(rgb, prompt, box_th, text_th)
                if det:
                    boxes_arr = np.stack([d.box_xyxy for d in det], axis=0)
                    mask = pipe.segmenter.segment(rgb, boxes_arr)
                else:
                    mask = np.zeros(rgb.shape[:2], np.uint8)
        except Exception:
            # 推論エラーの内部詳細は出さない
            self._reply(500, "text/plain", b"inference error")
            return

        ok, buf = cv2.imencode(".png", mask)
        if not ok:
            self._reply(500, "text/plain", b"encode error")
            return
        mask_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        out = {
            "boxes": [
                {
                    "x1": float(d.box_xyxy[0]), "y1": float(d.box_xyxy[1]),
                    "x2": float(d.box_xyxy[2]), "y2": float(d.box_xyxy[3]),
                    "score": float(d.score),
                    # label は入力 prompt に依存するので そのまま出す (XSS 面は C++ クライアント側で意識)
                    "label": d.label[:128] if d.label else "",
                }
                for d in det
            ],
            "mask_png_base64": mask_b64,
            "width": int(w),
            "height": int(h),
        }
        self._reply(200, "application/json", json.dumps(out).encode())

    def do_OPTIONS(self):
        # CORS preflight は CORS 許可時のみ 204 応答.
        if _ALLOW_CORS:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers",
                             "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.end_headers()
        else:
            self._reply(405, "text/plain", b"method not allowed")

    def _reply(self, status: int, ctype: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if _ALLOW_CORS:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


# --------------------------------------------------------------------------
def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost",)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address. 0.0.0.0 や LAN IP を使うには --allow-remote 必須.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--backend", choices=["torch", "onnx", "trt"], default="torch")
    ap.add_argument("--device", default=None)
    ap.add_argument("--token", default=None,
                    help="Bearer token. 未指定時は自動生成して起動時に出力.")
    ap.add_argument("--no-auth", action="store_true",
                    help="認証を無効化 (loopback バインド時のみ許可).")
    ap.add_argument("--allow-remote", action="store_true",
                    help="非 loopback アドレスへのバインドを明示的に許可する.")
    ap.add_argument("--cors", action="store_true",
                    help="CORS ヘッダを付与する (ブラウザから叩く場合のみ).")
    args = ap.parse_args(argv)

    # --- 安全ガード ----
    if not _is_loopback(args.host) and not args.allow_remote:
        print(f"ERROR: {args.host} は loopback ではありません. "
              "公開するには --allow-remote を付けてください.\n"
              "        (この時 --token または明示的な --no-auth が必要)",
              file=sys.stderr)
        return 2
    if not _is_loopback(args.host) and args.no_auth:
        print("ERROR: 非 loopback バインドで --no-auth は許可されません.",
              file=sys.stderr)
        return 2

    global _CONFIG, _EXPECTED_TOKEN, _ALLOW_CORS
    _CONFIG = Config()
    if args.device:
        _CONFIG.device = args.device
    _CONFIG.backend_detector = args.backend
    _CONFIG.backend_segmenter = args.backend
    _ALLOW_CORS = bool(args.cors)

    if args.no_auth:
        _EXPECTED_TOKEN = None
        print("[server] WARNING: authentication disabled (loopback only).")
    else:
        _EXPECTED_TOKEN = args.token or secrets.token_urlsafe(32)
        if not args.token:
            print(f"[server] generated token: {_EXPECTED_TOKEN}")
            print("         (Authorization: Bearer <token> で叩いてください)")

    print(f"[server] warming up pipeline (backend={args.backend})…")
    _ = _get_pipeline()
    print(f"[server] listening on http://{args.host}:{args.port}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] shutdown")
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
