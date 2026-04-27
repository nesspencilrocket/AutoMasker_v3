# AutoMasker (OSS版) v0.3-beta

![CI](https://github.com/YOUR-USERNAME/automasker/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

テキストプロンプトから画像・動画・**360°パノラマ** のマスクを自動生成し、
COLMAP/3DGS学習互換の出力を吐くツール。**Grounding DINO + SAM 2** をコアに、
**ONNX Runtime / TensorRT** で 2-4倍高速化、**mask-lift (Shen et al. ECCV 2024 のアルゴリズム)** で 2D マスクを
3D Gaussian Splatting シーンへリフト、**C++ / Dear ImGui** の低レイテンシ
ビューワも同梱します。

## 小言
作っている方を現実で見かけたのでClaudeとClaude Codeで作成しただけです。

> 📖 **セキュリティ**: [SECURITY.md](SECURITY.md) を必読してください.
> 特に `automasker.server` はデフォルトで localhost のみにバインドされ、
> ネットワーク公開には明示的な許可とトークンが必要です.
> 
> 🤝 **コントリビュート**: [CONTRIBUTING.md](CONTRIBUTING.md) を参照.

## 精度と制限事項 (重要)

本ツールは実用的なマスク自動生成を提供しますが、**すべてのコンポーネントが
論文と数値的に等価というわけではありません**. 特に 3DGS 連携部は独自実装
の近似を含みます. 以下を必ずご確認ください.

### マスク生成 (Grounding DINO + SAM 2)

- **torch バックエンド** (既定): `groundingdino-py` / 公式 SAM2 を直接呼ぶため、
  論文および公式実装と同じ精度が期待できる.
- **ONNX / TensorRT バックエンド**: 前処理・後処理・トークナイズを自前で
  再実装しているため、torch 版と極めて近いが 完全一致ではない. 特に Grounding
  DINO の text_threshold 周りは本家と挙動が微妙に異なる可能性があり、
  プロンプト設計は実データで検証することを推奨.
- **SAM2 の export 仕様差**: export した decoder が `orig_im_size` を受け
  取るかどうかで出力解像度が変わる. v0.3-beta で自動分岐を実装したが、
  自作の export を使う場合は出力解像度を一度目視確認すること.

### 360° (Equirectangular) パッチ推論

- **tangent 24視野** で全球被覆率 99.97%, 境界での IoU 誤差は
  テストで roundtrip IoU = 1.000 を確認.
- ただし、**極付近 (pitch ≈ ±90°)** は視野境界が密集するため、
  pitch 段数を 3 より増やすと精度向上する場合がある.
- **非 2:1 アスペクトの ERP** (完全円周画像でない魚眼等) は未対応.

### mask-lift (2D → 3D Gaussian ラベル)

**⚠️ この機能は「論文のアルゴリズムを独自に再実装した近似」です**.
公式 FlashSplat 実装 (INRIA 研究ライセンス) のコードは一切使っておらず、
数式だけを見て書き起こしているため、結果が論文と同一になる保証はありません.

| バックエンド | 精度特性 | 適した用途 |
|------------|---------|-----------|
| `approx` (既定) | 等方 circular splat の近似. anisotropic 情報を捨てる. | 物体が疎に配置されたシーン (単体オブジェクト抽出) |
| `gsplat` | 本物の anisotropic 3DGS rasterizer を使うが、**バッチ境界をまたぐ occlusion は失われる**. batch_size を全ガウス数に合わせられれば論文と等価. | 密なシーン. ただし batch_size 調整 (VRAM許す限り大) が必要. |

いずれのバックエンドでも論文のオリジナル実装の数値再現はしません.
実運用では **`--bg-bias λ` を 0.1 ~ 0.5 の範囲で実データに合わせて調整**
することを強く推奨します.

---



### マスク生成

- ✅ テキストプロンプト (`"person . tripod"`) で Open-Vocabulary 検出
- ✅ **時間的整合性 (VOS)**: 1 フレーム目のマスクを動画全体へ伝搬
- ✅ **360° Equirectangular 対応**: tangent patch / cubemap で歪み回避
- ✅ マスク反転 (対象を消す / 対象だけ残す)
- ✅ COLMAP 互換出力 (`masks/<name>.png`)
- ✅ バッチ処理、リアルタイム閾値調整、モルフォロジー整形

### パフォーマンス

- ✅ プラガブルバックエンド: `--backend {torch, onnx, trt}`
- ✅ SAM2 / Grounding DINO の ONNX 分割エクスポート
- ✅ TensorRT FP16 Engine (`trtexec` or Python API)
- ✅ ベンチマークスクリプト (median speedup 表)

### 3DGS 連携 (2D→3D mask lifting)

- ✅ 2D マスク集合 → 各 Gaussian のラベルを閉形式で解く
- ✅ 2 つの W 計算バックエンド:
  - **approx** — 等方 circular splat 近似 (依存なし, 軽量)
  - **gsplat** — nerfstudio-project/gsplat で anisotropic 厳密計算
- ✅ COLMAP sparse + 学習済み `.ply` + masks/ → 対象/非対象の `.ply`

### GUI / フロントエンド

- ✅ **PySide6 GUI** (`python -m automasker.gui`)
- ✅ **HTTP 推論サーバ** (`python -m automasker.server`) — ローカル JSON API
- ✅ **C++ / Dear ImGui フロントエンド** — 低レイテンシ OpenGL ビューワ

## ディレクトリ

```
automasker/
├── automasker/
│   ├── config.py / pipeline.py / refine.py / io_utils.py
│   ├── detector.py / segmenter.py           # torch 実装
│   ├── cli.py / gui.py / server.py
│   ├── backends/                            # ONNX / TRT
│   │   ├── sam2_onnx.py / sam2_trt.py
│   │   └── gdino_onnx.py / gdino_trt.py
│   ├── pano/                                # 360° Equirectangular
│   │   ├── projection.py                    # ERP ⇄ perspective
│   │   └── pipeline.py                      # パッチ推論 + ERP合成
│   └── mask_lift/                          # 2D→3D リフト (Shen et al. 2024 のアルゴリズム独自実装)
│       ├── lift.py                          # 閉形式解法 (approx)
│       ├── lift_gsplat.py                   # gsplat 厳密計算
│       ├── ply_io.py / camera_io.py
│       └── lift_cli.py
├── export/
│   ├── sam2_onnx.py / gdino_onnx.py
│   ├── build_trt.py / benchmark.py
├── cpp_frontend/                            # C++ ImGui ビューワ
│   ├── src/main.cpp
│   └── CMakeLists.txt
├── scripts/ / tests/ / NOTES.md
├── README.md
└── requirements.txt
```

## セットアップ

### 基本 (Torch)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
pip install -e third_party/sam2
pip install groundingdino-py
python scripts/download_checkpoints.py
```

### 高速化 (ONNX → TRT)

```bash
pip install onnxruntime-gpu onnxsim transformers
python -m export.sam2_onnx --cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --ckpt checkpoints/sam2.1_hiera_large.pt --output-dir checkpoints --simplify
python -m export.gdino_onnx --cfg checkpoints/GroundingDINO_SwinT_OGC.py \
    --ckpt checkpoints/groundingdino_swint_ogc.pth --output checkpoints/gdino.onnx

pip install tensorrt pycuda
python -m export.build_trt --target sam2  --fp16
python -m export.build_trt --target gdino --fp16
```

### mask-lift (gsplat 厳密版を使う場合)

```bash
pip install plyfile gsplat
```

### C++ フロントエンド

```bash
cd cpp_frontend
git clone https://github.com/ocornut/imgui external/imgui
wget https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h -P external/
mkdir -p external/stb && \
    wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -P external/stb/
# GLFW は apt/brew でインストール (Linux: libglfw3-dev / macOS: brew install glfw)
cmake -B build -S . && cmake --build build -j
```

## 使い方

### マスク生成 (CLI)

```bash
# 基本
python -m automasker.cli -i scene/images -o scene/masks --prompt "person . tripod"

# 高速化
python -m automasker.cli -i scene/images -o scene/masks --prompt "person" --backend trt

# 動画 (VOS)
python -m automasker.cli -i video.mp4 -o out/masks --prompt "person" --video

# 360° パノラマ (tangent 24 パッチ)
python -m automasker.cli -i panorama.jpg -o out/masks --prompt "person" \
    --pano --pano-mode tangent --pano-n-yaw 8 --pano-n-pitch 3 --pano-fov 90

# 360° cubemap モード (6面)
python -m automasker.cli -i pano/ -o out/masks --prompt "person" \
    --pano --pano-mode cubemap --pano-patch-size 1024
```

### GUI / サーバ

```bash
# PySide6 GUI (デスクトップ)
python -m automasker.gui

# HTTP 推論サーバ (C++ frontend や外部ツールから叩く)
python -m automasker.server --host 127.0.0.1 --port 8765 --backend trt

# C++ / ImGui ビューワ (別ターミナルで)
./cpp_frontend/build/automasker_frontend path/to/image.jpg
```

### mask-lift (3DGS リフト)

```bash
# 軽量版 (外部依存なし, 等方 circular splat 近似)
python -m automasker.mask_lift.lift_cli \
    --scene scene --ply scene/.../point_cloud.ply --masks scene/masks \
    --output-dir scene/mask_lift_out --mode both --bg-bias 0.2

# 厳密版 (gsplat anisotropic)
python -m automasker.mask_lift.lift_cli --backend gsplat --batch-size 2048 \
    --scene scene --ply scene/.../point_cloud.ply --masks scene/masks \
    --output-dir scene/mask_lift_out --mode both
```

### ベンチマーク


## 3DGS へのつなぎ込み

AutoMasker 既定出力は `対象=0 (消す), 背景=255 (学習に使う)` のため、
gaussian-splatting 系の `train.py` はそのまま `masks/` を読み込みます。

3DGS 学習後に mask-lift で 3D ラベルを決定すれば:
- **物体抽出**: `point_cloud_target.ply` → 単体 viewer で表示
- **物体除去**: `point_cloud_without_target.ply` → inpainting の初期値

## テスト

```bash
python tests/test_smoke.py        # 10 件: I/O, refinement, path sanitize
python tests/test_mask_lift.py   #  4 件: projection, solver
python tests/test_pano.py         #  4 件: ERP ⇄ perspective, roundtrip
# 合計 18 件全て pass
```

## ロードマップ

- [x] コア機能: Grounding DINO + SAM2 + COLMAP 出力
- [x] PySide6 GUI + CLI
- [x] ONNX / TensorRT バックエンド (2-4倍)
- [x] 2D→3D mask lifting 連携 (閉形式ラベル解法)
- [x] 360° (Equirectangular) 画像対応
- [x] gsplat ベース anisotropic W 計算
- [x] C++ / Dear ImGui フロントエンド + HTTP 推論サーバ

## v0.3-beta での既知の制約

- `mask_lift/` は **論文のアルゴリズム独自実装** であり、公式 FlashSplat
  実装との数値同一性は保証しない (詳細は 「精度と制限事項」 セクション参照).
- ONNX/TensorRT バックエンドには非ML部分のテストカバレッジがない
  (CI で ML 依存を展開しないため).
- `automasker.server` のエンドポイントには未テストコード経路がある.
  ローカル開発用途以外では使用を推奨しない.

実データで回してのフィードバック・issue 報告を歓迎します.
