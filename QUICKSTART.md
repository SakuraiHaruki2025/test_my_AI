# クイックスタートガイド

このガイドでは、プロジェクトをすぐに始める方法を説明します。

## セットアップ（5分）

```bash
# 1. リポジトリのクローン
git clone https://github.com/SakuraiHaruki2025/test_my_AI.git
cd test_my_AI

# 2. 依存関係のインストール
pip install -r requirements.txt

# 3. 動作確認
python examples/demo.py
```

## 敬語AIを使う（学習不要）

敬語AIは学習済みモデルを使用するため、すぐに試せます。

```bash
# デモ実行（簡易版）
python src/models/model_manager.py \
  --model-type keigo \
  --prompt "今日はいい天気ですね"
```

もしくは直接推論スクリプトを使用：

```bash
python src/inference/inference.py \
  --model rinna/japanese-gpt-neox-3.6b \
  --prompt "今日はいい天気ですね" \
  --load-in-8bit
```

**注意**: 初回実行時はモデルのダウンロードに時間がかかります（約4GB）。

## 関西弁AIを作る（学習が必要）

### ステップ1: データを確認

サンプルデータが既に用意されています：

```bash
cat examples/kansai_data.json
```

### ステップ2: 学習を実行

```bash
python src/training/train.py \
  --base-model rinna/japanese-gpt-neox-3.6b \
  --data-path examples/kansai_data.json \
  --output-dir ./models/kansai_finetuned \
  --epochs 3 \
  --batch-size 4 \
  --use-qlora
```

**所要時間**: RTX 4070 SUPERで約30-60分

### ステップ3: 学習したモデルで推論

```bash
python src/inference/inference.py \
  --model ./models/kansai_finetuned/final_model \
  --prompt "今日はいい天気ですね" \
  --load-in-8bit
```

## 形容詞＋やねAIを作る（学習が必要）

### ステップ1: データを確認

```bash
cat examples/adjective_data.json
```

### ステップ2: 学習を実行

```bash
python src/training/train.py \
  --base-model rinna/japanese-gpt-neox-3.6b \
  --data-path examples/adjective_data.json \
  --output-dir ./models/adjective_finetuned \
  --epochs 5 \
  --batch-size 4 \
  --use-qlora
```

**所要時間**: RTX 4070 SUPERで約40-80分

### ステップ3: 学習したモデルで推論

```bash
python src/inference/inference.py \
  --model ./models/adjective_finetuned/final_model \
  --prompt "この花を見て" \
  --load-in-8bit
```

## よくある質問

### Q: メモリが足りないエラーが出ます

A: 以下の対策を試してください：

1. バッチサイズを小さくする: `--batch-size 2` または `--batch-size 1`
2. 4bit量子化を使う: `--load-in-4bit`
3. より小さいモデルを使う: `rinna/japanese-gpt-1b`

### Q: 学習にどのくらい時間がかかりますか？

A: 環境によりますが、目安：
- RTX 4070 SUPER: 30-60分（3エポック）
- RTX 3060: 1-2時間
- CPU: 推奨しません（非常に遅い）

### Q: もっとデータを追加したい

A: `examples/`のJSONファイルを編集するか、新しいファイルを作成してください：

```json
[
  {
    "text": "質問: あなたの質問\n答え: AIの応答"
  }
]
```

### Q: 他のベースモデルを使いたい

A: `--base-model`パラメータで変更できます：

```bash
# Open-CALMを使う場合
python src/training/train.py \
  --base-model cyberagent/open-calm-7b \
  ...
```

### Q: データセットを分割したい

A: データユーティリティを使用：

```bash
python src/data/data_utils.py split \
  --input examples/kansai_data.json \
  --output ./data/kansai \
  --train-ratio 0.8
```

## 次のステップ

1. **独自データの作成**: 自分のデータセットを作って学習してみる
2. **ハイパーパラメータ調整**: 学習率やエポック数を変えて実験
3. **モデルの比較**: 異なるベースモデルで性能を比較
4. **カスタムパターン**: 新しい話し方のパターンを作成

詳細は`README.md`を参照してください。
