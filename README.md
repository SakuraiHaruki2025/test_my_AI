# test_my_AI - 日本語LLM学習プロジェクト

最近の言語モデルの知見に追いつきたいね。

## 概要

このプロジェクトは、日本語の言語モデルを学習・使用するための教育的なコードベースです。
3種類の異なる話し方をするAIを作成します：

1. **敬語AI** - 丁寧な敬語で話す学習済みモデル
2. **関西弁AI** - 関西弁で話すカスタム学習モデル
3. **形容詞＋やねAI** - 形容詞＋語尾（やね）のパターンで話すカスタム学習モデル

## 特徴

- 📱 **ポータブルGPU対応**: 低スペックGPUでも動作する量子化技術を使用
- 🚀 **効率的な学習**: QLoRAを使用したメモリ効率の良いファインチューニング
- 📚 **教育的**: 分かりやすいコメント付きコード
- 🆕 **最新技術**: 2024年時点の最新LLM技術を採用

## システム要件

### 推論用（ポータブルGPU）
- GPU: 4GB以上のVRAM
- RAM: 8GB以上
- Python: 3.8以上

### 学習用（推奨）
- GPU: NVIDIA GeForce RTX 4070 SUPER または同等
- VRAM: 12GB以上
- RAM: 16GB以上
- Python: 3.8以上

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/SakuraiHaruki2025/test_my_AI.git
cd test_my_AI

# 依存関係のインストール
pip install -r requirements.txt
```

## プロジェクト構造

```
test_my_AI/
├── src/
│   ├── inference/           # 推論用スクリプト
│   │   └── inference.py     # 量子化推論エンジン
│   ├── training/            # 学習用スクリプト
│   │   └── train.py         # QLoRAファインチューニング
│   └── models/              # モデル管理
│       └── model_manager.py # 統合インターフェース
├── configs/                 # モデル設定ファイル
│   ├── keigo_model.json     # 敬語AIの設定
│   ├── kansai_model.json    # 関西弁AIの設定
│   └── adjective_model.json # 形容詞＋やねAIの設定
├── examples/                # サンプルデータ
│   ├── kansai_data.json     # 関西弁学習データ
│   └── adjective_data.json  # 形容詞パターン学習データ
└── requirements.txt         # 依存パッケージ
```

## 使い方

### 1. 敬語AI（学習済みモデル）

敬語AIは学習済みのモデルをそのまま使用します。

```bash
# 推論の実行
python src/inference/inference.py \
  --model rinna/japanese-gpt-neox-3.6b \
  --prompt "今日はいい天気ですね" \
  --load-in-8bit
```

モデルマネージャーを使う場合：

```bash
python src/models/model_manager.py \
  --model-type keigo \
  --prompt "今日はいい天気ですね"
```

### 2. 関西弁AI（要学習）

関西弁AIはカスタムデータで学習が必要です。

```bash
# 学習の実行
python src/training/train.py \
  --base-model rinna/japanese-gpt-neox-3.6b \
  --data-path examples/kansai_data.json \
  --output-dir ./models/kansai_finetuned \
  --epochs 3 \
  --use-qlora

# 学習後の推論
python src/inference/inference.py \
  --model ./models/kansai_finetuned/final_model \
  --prompt "今日はいい天気ですね" \
  --load-in-8bit
```

### 3. 形容詞＋やねAI（要学習）

形容詞＋やねパターンで話すAIも学習が必要です。

```bash
# 学習の実行
python src/training/train.py \
  --base-model rinna/japanese-gpt-neox-3.6b \
  --data-path examples/adjective_data.json \
  --output-dir ./models/adjective_finetuned \
  --epochs 5 \
  --use-qlora

# 学習後の推論
python src/inference/inference.py \
  --model ./models/adjective_finetuned/final_model \
  --prompt "この花を見て" \
  --load-in-8bit
```

## 技術詳細

### 量子化技術

推論時のメモリを節約するために、以下の量子化技術を使用：

- **8bit量子化**: メモリ使用量を約50%削減
- **4bit量子化**: メモリ使用量を約75%削減（QLoRA）

### LoRA/QLoRA

学習時は以下の技術で効率化：

- **LoRA (Low-Rank Adaptation)**: パラメータの一部だけを学習
- **QLoRA**: 4bit量子化とLoRAの組み合わせ
- メモリ使用量を大幅に削減しながら高品質な学習が可能

### 推奨ベースモデル

日本語に特化したモデルを推奨：

1. `rinna/japanese-gpt-neox-3.6b` - 3.6Bパラメータ、高品質
2. `cyberagent/open-calm-7b` - 7Bパラメータ、より高性能
3. `rinna/japanese-gpt-1b` - 1Bパラメータ、軽量

## カスタマイズ

### 独自データセットの作成

学習データは以下の形式で作成：

```json
[
  {
    "text": "質問: あなたの入力\n答え: AIの応答"
  },
  ...
]
```

### ハイパーパラメータの調整

`configs/`ディレクトリの設定ファイルで調整可能：

- `lora_r`: LoRAのランク（8-64が一般的）
- `learning_rate`: 学習率（2e-4が標準）
- `batch_size`: バッチサイズ
- `num_epochs`: エポック数

## トラブルシューティング

### メモリ不足エラー

```bash
# より小さいバッチサイズを使用
python src/training/train.py ... --batch-size 2

# 4bit量子化を使用
python src/inference/inference.py ... --load-in-4bit
```

### CUDAエラー

```bash
# PyTorchとCUDAの互換性を確認
python -c "import torch; print(torch.cuda.is_available())"
```

## 参考資料

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [QLoRA論文](https://arxiv.org/abs/2305.14314)

## ライセンス

MITライセンス

## 貢献

プルリクエストを歓迎します！

## 作者

SakuraiHaruki2025
