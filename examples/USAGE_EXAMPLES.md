# 使用例集

このファイルには、各種スクリプトの具体的な使用例をまとめています。

## 推論スクリプト (inference.py)

### 基本的な使い方

```bash
# デフォルト設定で実行
python src/inference/inference.py \
  --model rinna/japanese-gpt-neox-3.6b \
  --prompt "こんにちは"
```

### メモリ節約モード

```bash
# 8bit量子化（推奨）
python src/inference/inference.py \
  --model rinna/japanese-gpt-neox-3.6b \
  --prompt "今日の天気はどうですか？" \
  --load-in-8bit

# 4bit量子化（最も省メモリ）
python src/inference/inference.py \
  --model rinna/japanese-gpt-neox-3.6b \
  --prompt "今日の天気はどうですか？" \
  --load-in-4bit
```

## 学習スクリプト (train.py)

### 基本的な学習

```bash
# 関西弁AIの学習
python src/training/train.py \
  --base-model rinna/japanese-gpt-neox-3.6b \
  --data-path examples/kansai_data.json \
  --output-dir ./models/kansai_model \
  --epochs 3
```

### 学習パラメータの調整

```bash
# より長い学習
python src/training/train.py \
  --base-model rinna/japanese-gpt-neox-3.6b \
  --data-path examples/kansai_data.json \
  --output-dir ./models/kansai_model_long \
  --epochs 10 \
  --learning-rate 1e-4
```

## モデル管理スクリプト (model_manager.py)

### コマンドラインから使用

```bash
# 敬語AIを使用
python src/models/model_manager.py \
  --model-type keigo \
  --prompt "今日はいい天気ですね"
```

### Pythonスクリプトから使用

```python
from src.models.model_manager import ModelManager

# マネージャーの初期化
manager = ModelManager(config_dir="./configs")

# 敬語AIの使用
manager.load_model("keigo")
result = manager.generate("keigo", "こんにちは")
print(result)
```

## データユーティリティ (data_utils.py)

### データセットの検証

```bash
# 関西弁データの検証
python src/data/data_utils.py validate \
  --input examples/kansai_data.json
```

### データセットの分割

```bash
# 80/20で分割
python src/data/data_utils.py split \
  --input examples/kansai_data.json \
  --output ./data/kansai_split \
  --train-ratio 0.8
```

## 実践的なワークフロー例

### 新しい方言AIを作る

```bash
# 1. データの準備（data/osaka_dialect.json）
# 2. データの検証
python src/data/data_utils.py validate \
  --input data/osaka_dialect.json

# 3. 学習
python src/training/train.py \
  --base-model rinna/japanese-gpt-neox-3.6b \
  --data-path data/osaka_dialect.json \
  --output-dir ./models/osaka_dialect \
  --epochs 5

# 4. 推論
python src/inference/inference.py \
  --model ./models/osaka_dialect/final_model \
  --prompt "こんにちは" \
  --load-in-8bit
```

詳細はREADME.mdやTUTORIAL.mdを参照してください。
