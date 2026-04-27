# 変更内容

<!-- 何を変えたか、なぜ変えたかを簡潔に -->

## チェックリスト

- [ ] 既存テストが pass する (`python tests/test_smoke.py`, `test_mask_lift.py`, `test_pano.py`)
- [ ] 新機能には対応するテストを追加した
- [ ] `bandit -r automasker/ export/ scripts/ --severity-level medium` を通過する
- [ ] README / NOTES / SECURITY を必要に応じて更新した
- [ ] モデル重みやユーザーデータをコミットしていない (`git status` 確認)

## 関連 issue

<!-- Fixes #N / Refs #N -->
