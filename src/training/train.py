"""
学習スクリプト - RTX 4070 SUPER用
このスクリプトはLoRA/QLoRAを使用して効率的にファインチューニングを行います。
最新のLLM学習手法を採用しています。
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, Dataset
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List
import os


class LLMTrainer:
    """
    LLMファインチューニングクラス
    LoRA/QLoRAを使用してメモリ効率的に学習します。
    """
    
    def __init__(
        self,
        base_model: str,
        output_dir: str,
        use_qlora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None
    ):
        """
        Args:
            base_model: ベースモデル名
            output_dir: 出力ディレクトリ
            use_qlora: QLoRA（4bit量子化 + LoRA）を使用するか
            lora_r: LoRAのランク（低いほど省メモリ）
            lora_alpha: LoRAのスケーリングファクター
            lora_dropout: LoRAのドロップアウト率
            target_modules: LoRAを適用するモジュール名
        """
        self.base_model = base_model
        self.output_dir = output_dir
        self.use_qlora = use_qlora
        
        print(f"ベースモデル: {base_model}")
        print(f"QLoRA使用: {use_qlora}")
        print(f"出力先: {output_dir}")
        
        # 出力ディレクトリ作成
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 量子化設定（QLoRAの場合）
        quantization_config = None
        if use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # トークナイザーのロード
        print("トークナイザーをロード中...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        
        # パディングトークンの設定（必要な場合）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # モデルのロード
        print("モデルをロード中...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if not use_qlora else None
        )
        
        # QLoRAの場合は追加の準備
        if use_qlora:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA設定
        # デフォルトのターゲットモジュール（日本語LLMで一般的なもの）
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # LoRAの適用
        print("LoRAを適用中...")
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()  # 学習可能なパラメータ数を表示
        
        print("モデルの準備が完了しました！")
    
    def prepare_dataset(
        self,
        data_path: str,
        max_length: int = 512,
        prompt_template: Optional[str] = None
    ) -> Dataset:
        """
        データセットを準備する
        
        Args:
            data_path: データファイルのパス（JSON, JSONL, TXT）
            max_length: トークンの最大長
            prompt_template: プロンプトテンプレート（例: "### 入力:\n{text}\n### 出力:\n"）
            
        Returns:
            準備されたデータセット
        """
        print(f"データセットをロード中: {data_path}")
        
        # ファイル形式に応じてロード
        file_ext = Path(data_path).suffix.lower()
        
        if file_ext == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_ext == '.jsonl':
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        elif file_ext == '.txt':
            with open(data_path, 'r', encoding='utf-8') as f:
                texts = f.read().split('\n\n')  # 空行で分割
                data = [{"text": text.strip()} for text in texts if text.strip()]
        else:
            raise ValueError(f"サポートされていないファイル形式: {file_ext}")
        
        # データセット作成
        dataset = Dataset.from_list(data)
        
        # トークン化関数
        def tokenize_function(examples):
            texts = examples["text"]
            
            # プロンプトテンプレートの適用
            if prompt_template:
                texts = [prompt_template.format(text=text) for text in texts]
            
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
        
        # トークン化
        print("データセットをトークン化中...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        print(f"データセットサイズ: {len(tokenized_dataset)} サンプル")
        return tokenized_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500
    ):
        """
        モデルを学習する
        
        Args:
            train_dataset: 訓練データセット
            eval_dataset: 評価データセット
            num_epochs: エポック数
            batch_size: バッチサイズ
            gradient_accumulation_steps: 勾配累積ステップ数
            learning_rate: 学習率
            warmup_steps: ウォームアップステップ数
            logging_steps: ログ出力間隔
            save_steps: モデル保存間隔
            eval_steps: 評価実行間隔
        """
        print("学習設定:")
        print(f"  エポック数: {num_epochs}")
        print(f"  バッチサイズ: {batch_size}")
        print(f"  勾配累積: {gradient_accumulation_steps}")
        print(f"  実効バッチサイズ: {batch_size * gradient_accumulation_steps}")
        print(f"  学習率: {learning_rate}")
        
        # 学習設定
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=3,  # 最新の3チェックポイントのみ保存
            fp16=True,  # 混合精度学習
            optim="paged_adamw_32bit" if self.use_qlora else "adamw_torch",
            lr_scheduler_type="cosine",
            report_to="none",  # wandbなどを使う場合は変更
            load_best_model_at_end=True if eval_dataset else False,
            gradient_checkpointing=True,  # メモリ節約
        )
        
        # データコレーター
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # 因果的言語モデリング
        )
        
        # トレーナー
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        # 学習開始
        print("\n" + "="*50)
        print("学習を開始します...")
        print("="*50 + "\n")
        
        trainer.train()
        
        # 最終モデルの保存
        print("\n最終モデルを保存中...")
        trainer.save_model(os.path.join(self.output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_model"))
        
        print(f"学習完了！モデルは {self.output_dir} に保存されました。")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="LLM学習スクリプト")
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="ベースモデル名またはパス"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="学習データのパス（JSON, JSONL, TXT）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/finetuned",
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="エポック数"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="バッチサイズ"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="学習率"
    )
    parser.add_argument(
        "--use-qlora",
        action="store_true",
        default=True,
        help="QLoRAを使用（デフォルト: True）"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRAのランク"
    )
    
    args = parser.parse_args()
    
    # トレーナー初期化
    trainer = LLMTrainer(
        base_model=args.base_model,
        output_dir=args.output_dir,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r
    )
    
    # データセット準備
    dataset = trainer.prepare_dataset(args.data_path)
    
    # 学習実行
    trainer.train(
        train_dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
