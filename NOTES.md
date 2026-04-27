# Notes: VRAM 節約・モデル切り替え・トラブルシュート

## VRAM が足りない / OOM が出る

SAM 2.1 は 4つのサイズがあります。既定は Hiera-Large (VRAM 6-8GB 相当) ですが、
VRAM が足りないなら **Tiny** に落とすだけで動くことが多いです。

| バリアント | checkpoint                       | config                       | おおよそのVRAM |
|-----------|----------------------------------|------------------------------|---------------|
| Tiny      | `sam2.1_hiera_tiny.pt`           | `sam2.1_hiera_t.yaml`        | ~2 GB         |
| Small     | `sam2.1_hiera_small.pt`          | `sam2.1_hiera_s.yaml`        | ~3 GB         |
| Base+     | `sam2.1_hiera_base_plus.pt`      | `sam2.1_hiera_b+.yaml`       | ~4 GB         |
| Large     | `sam2.1_hiera_large.pt` (既定)    | `sam2.1_hiera_l.yaml`        | ~7 GB         |

### 切り替え方

`automasker/config.py` の `Config` を変えるか、起動時に上書きします。

```python
from automasker.config import Config
cfg = Config()
cfg.sam2_cfg  = "configs/sam2.1/sam2.1_hiera_t.yaml"
cfg.sam2_ckpt = "checkpoints/sam2.1_hiera_tiny.pt"
```

チェックポイントのURLは Meta のバケットから:

```
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

`scripts/download_checkpoints.py` の `FILES` 辞書に追記すれば自動DL可。

---

## プロンプトのコツ

Grounding DINO は **ピリオド区切り** を強く好みます。カンマや "and" はNG。

```
OK:  "person . tripod . camera operator . drone"
NG:  "person, tripod, and camera"
```

抽象語 (`"object"`, `"thing"`) は精度が落ちます。なるべく具体的な名詞で。
`"selfie stick"`, `"reflective glass window"`, `"construction cone"` のような
複合語はそのまま通じます。

---

## 時間的整合性 (VOS) がずれる

SAM2 の video predictor は **連番JPEGディレクトリ** が前提です。
入力が任意の命名の画像シーケンスだと失敗することがあります。

回避策:
- `cli.py` で `--video` を使うときはまず動画ファイルにエンコードし直す、または
- `video_to_frames()` を通して `000000.jpg`, `000001.jpg` ... に展開する。

1フレーム目で対象が画面外 / 隠れている場合は検出が取れず全体が失敗するので、
最初に対象がよく見えているフレームを指定する (`--stride` や `--max-frames` で調整).

---

## 境界のにじみが気になる

`--dilate` を 5-10 にして少し大きめに消すのが実用的。
最終出力前に GrabCut で締めたい場合は `refine.grabcut_refine()` を
`pipeline.py` の `run_single` の末尾で呼ぶように1行追加すれば有効化できます
(重いのでバッチ時のみ推奨).

---

## 3DGS 側での使い方 (参考)

gaussian-splatting / 2DGS / Scaffold-GS 系の `train.py` は、`scene/masks/<stem>.png`
を自動で読み込み、白=損失計算対象 / 黒=無視 の形で扱う実装が多いです。

本ツールは既定で **対象=0 (消す), 背景=255 (学習に使う)** なので、
COLMAP シーンに `masks/` を置くだけでそのまま train.py が動きます。

「対象だけを残したい」ケース (例: 物体単体の再構成) は `--invert` を付けてください。

## プロンプト言語について

Grounding DINO は英語コーパスで学習されているため、**プロンプトは英語** を推奨します。
日本語・中国語などのCJK言語はトークナイザの粒度と学習分布の違いから検出精度が大きく落ちます。

```bash
# 推奨
--prompt "person . tripod . chair"

# 非推奨 (動作はするが精度が出ない可能性)
--prompt "人物 . 三脚 . 椅子"
```

小文字 + ピリオド区切りが本家推奨フォーマットです。

## 長時間動画処理の注意

SAM2 の video predictor は全フレームのマスクをメモリ上に保持する設計のため、
長い動画ではRAMが爆発的に増えます。目安:

- 1080p × 1000 frames ≈ 2 GB RAM
- 4K × 1000 frames ≈ 8 GB RAM

対策:
- `--stride 2` または `--stride 4` でフレームを間引く
- `--max-frames 500` で長さを制限
- 長編はシーン分割してバッチごとに実行

実装上の将来課題として、ストリーミング処理 (1フレームずつ書き出して解放) を
予定しています。現状は v0.3-beta の制限事項です。
