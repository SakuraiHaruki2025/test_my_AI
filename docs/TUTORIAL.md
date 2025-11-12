# チュートリアル

## はじめに

このチュートリアルでは、3種類の日本語AIモデルの作成と使用方法を学びます。
プログラミングの基礎知識があれば、順を追って進めることができます。

## 必要な知識

- Pythonの基本
- コマンドラインの基本操作
- 機械学習の基礎（推奨）

## セットアップ

### 1. 環境準備

```bash
# Python 3.8以上が必要
python --version

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# リポジトリのクローン
git clone https://github.com/SakuraiHaruki2025/test_my_AI.git
cd test_my_AI

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. GPUの確認

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## レッスン1: 敬語AIを使う

敬語AIは学習済みなので、すぐに使えます。

### 基本的な使い方

```bash
python src/inference/inference.py \
  --model rinna/japanese-gpt-neox-3.6b \
  --prompt "こんにちは、今日はいい天気ですね" \
  --load-in-8bit \
  --max-tokens 50
```

### パラメータの説明

- `--model`: 使用するモデル
- `--prompt`: 入力テキスト
- `--load-in-8bit`: 8bit量子化（メモリ節約）
- `--max-tokens`: 生成する最大トークン数

### 練習問題

異なるプロンプトで試してみましょう：

```bash
# 質問してみる
python src/inference/inference.py \
  --model rinna/japanese-gpt-neox-3.6b \
  --prompt "日本の首都はどこですか？" \
  --load-in-8bit

# 物語を始める
python src/inference/inference.py \
  --model rinna/japanese-gpt-neox-3.6b \
  --prompt "昔々あるところに" \
  --load-in-8bit \
  --max-tokens 100
```

## レッスン2: 関西弁AIを作る

### ステップ1: データの理解

サンプルデータを見てみましょう：

```bash
cat examples/kansai_data.json
```

データ形式：
```json
{
  "text": "質問: 標準語\n答え: 関西弁"
}
```

### ステップ2: データの拡張（オプション）

もっとデータを追加したい場合：

```python
# examples/my_kansai_data.json
[
  {
    "text": "質問: おはようございます\n答え: おはようさん！今日もええ天気やな"
  },
  {
    "text": "質問: お疲れ様です\n答え: お疲れさん！ほんまお疲れやったな"
  }
]
```

### ステップ3: 学習の実行

```bash
python src/training/train.py \
  --base-model rinna/japanese-gpt-neox-3.6b \
  --data-path examples/kansai_data.json \
  --output-dir ./models/my_kansai_model \
  --epochs 3 \
  --batch-size 4 \
  --use-qlora
```

学習中は以下の情報が表示されます：
- 損失（loss）: 小さくなるほど学習が進んでいる
- エポック: 全データを何回学習したか
- ステップ: 現在の学習進捗

### ステップ4: 学習したモデルで推論

```bash
python src/inference/inference.py \
  --model ./models/my_kansai_model/final_model \
  --prompt "今日はいい天気ですね" \
  --load-in-8bit
```

### 練習問題

1. 自分で関西弁データを5個追加してみよう
2. エポック数を5に増やして学習してみよう
3. 異なるプロンプトで関西弁応答を確認しよう

## レッスン3: 形容詞＋やねAIを作る

このAIは特殊なパターンで応答します。

### ステップ1: パターンの理解

```bash
cat examples/adjective_data.json
```

パターン：
- 入力: 「この花を見て」
- 出力: 「きれいやね」

### ステップ2: 学習

```bash
python src/training/train.py \
  --base-model rinna/japanese-gpt-neox-3.6b \
  --data-path examples/adjective_data.json \
  --output-dir ./models/my_adjective_model \
  --epochs 5 \
  --batch-size 4 \
  --use-qlora \
  --learning-rate 3e-4
```

### ステップ3: 推論

```bash
python src/inference/inference.py \
  --model ./models/my_adjective_model/final_model \
  --prompt "この景色を見て" \
  --load-in-8bit \
  --max-tokens 20
```

### 練習問題

新しい形容詞パターンを追加：
- 「嬉しい」「悲しい」「楽しい」などの感情形容詞
- 「大きい」「小さい」などの状態形容詞

## レッスン4: モデル管理ツールを使う

### ModelManagerの使用

```python
# test_model.py
from src.models.model_manager import ModelManager

# マネージャーの初期化
manager = ModelManager(config_dir="./configs")

# モデルのロード
manager.load_model("keigo")

# テキスト生成
result = manager.generate("keigo", "こんにちは")
print(result)
```

### データユーティリティの使用

```bash
# データセットの検証
python src/data/data_utils.py validate \
  --input examples/kansai_data.json

# データセットの分割
python src/data/data_utils.py split \
  --input examples/kansai_data.json \
  --output ./data/kansai_split \
  --train-ratio 0.8
```

## トラブルシューティング

### メモリ不足エラー

```bash
# 4bit量子化を使用
python src/inference/inference.py \
  --model ... \
  --load-in-4bit

# バッチサイズを小さく
python src/training/train.py \
  --batch-size 1 \
  ...
```

### 学習が遅い

```bash
# 勾配累積を増やす
python src/training/train.py \
  --gradient-accumulation-steps 8 \
  ...
```

### モデルのダウンロードエラー

Hugging Faceのキャッシュをクリア：
```bash
rm -rf ~/.cache/huggingface/
```

## 発展的なトピック

### 1. カスタムベースモデル

```bash
python src/training/train.py \
  --base-model cyberagent/open-calm-7b \
  ...
```

### 2. ハイパーパラメータ調整

```bash
python src/training/train.py \
  --lora-r 16 \
  --learning-rate 1e-4 \
  --epochs 10 \
  ...
```

### 3. 評価とテスト

学習したモデルの品質を確認：
- 複数のプロンプトで試す
- 期待される出力と比較
- 一貫性をチェック

## まとめ

このチュートリアルで学んだこと：
1. 学習済みモデルの使用方法
2. カスタムデータでのファインチューニング
3. 特殊なパターンのAI作成
4. メモリ最適化技術

次のステップ：
- 自分の用途に合わせたデータセット作成
- 異なるモデルサイズの実験
- 応用アプリケーションの開発

詳細はREADME.mdとARCHITECTURE.mdを参照してください。
