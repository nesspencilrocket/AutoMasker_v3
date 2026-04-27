# Contributing to AutoMasker

まず本プロジェクトに興味を持っていただきありがとうございます。

## 開発環境のセットアップ

```bash
git clone <repo-url>
cd automasker

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 開発ツール
pip install bandit pip-audit
```

本体テスト (ML 抜きの純ロジック) は torch/SAM2/GDINO なしで走ります:

```bash
pip install numpy 'opencv-python-headless>=4.9' plyfile
python tests/test_smoke.py      # 7 件
python tests/test_mask_lift.py # 4 件
python tests/test_pano.py       # 4 件
```

フル機能を動かすには torch + SAM2 + Grounding DINO のインストールが必要です。
詳細は `README.md` を参照してください。

## ブランチとコミット

- `main` は常に緑 (CI 通過) を保ちます
- 機能追加は feature ブランチで、修正は fix ブランチで
- コミットメッセージは英語/日本語どちらでも OK、件名は 50 文字以内推奨

## プルリクエストの前に

以下をローカルで通してください:

```bash
# 1) 全テスト
python tests/test_smoke.py && python tests/test_mask_lift.py && python tests/test_pano.py

# 2) 静的セキュリティスキャン (medium+ 以上はゼロが必須)
bandit -r automasker/ export/ scripts/ --severity-level medium

# 3) 依存の既知 CVE チェック
pip-audit -r requirements.txt

# 4) 構文確認
python -c "import ast; from pathlib import Path; [ast.parse(p.read_text()) for p in Path('.').rglob('*.py') if '__pycache__' not in str(p)]"
```

CI (`.github/workflows/ci.yml`) が PR ごとに同じチェックを走らせます。

## 新しいテストの追加

- `tests/test_*.py` の単体スクリプト形式で十分です (pytest は必須ではない)
- ML 依存を引かないよう、合成データや numpy だけで検証する設計を推奨します
- `test_mask_lift.py` がその良い例です (2クラスタの Gaussians 合成データ)

## セキュリティに関わる変更

以下のいずれかに触れる場合は PR 説明に明記してください:

- `automasker/server.py` (HTTP エンドポイント, 認証, 入力検証)
- `scripts/download_checkpoints.py` (ダウンロード, ハッシュ検証)
- `torch.load` の `weights_only` パラメータ
- `subprocess` の実行引数
- 新しいネットワーク通信の追加

脆弱性の報告は `SECURITY.md` の手順に従い、public issue ではなく
GitHub Security Advisory を使ってください。

## コードスタイル

- Python は 4 スペースインデント、PEP 8 準拠 (blackで揃えるのが楽)
- docstring は日本語 OK、モジュール冒頭に何をするかを 1-3 行で
- type hints は歓迎 (必須ではない)
- 公開 API (`automasker/` 以下の import 可能なもの) を変えたら README を更新

## ライセンス

貢献されたコードは本プロジェクトの MIT License の下で配布されます。
