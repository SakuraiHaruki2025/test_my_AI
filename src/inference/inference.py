"""
推論スクリプト - ポータブルGPU用に最適化
このスクリプトは性能の低いGPUでもLLMを実行できるように、
量子化とメモリ最適化を使用しています。
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
from typing import Optional
import json


class LLMInference:
    """
    LLM推論クラス
    量子化を使ってメモリを節約しながら推論を行います。
    """
    
    def __init__(
        self, 
        model_name: str,
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        device_map: str = "auto"
    ):
        """
        Args:
            model_name: 使用するモデル名（HuggingFace or ローカルパス）
            load_in_8bit: 8bit量子化を使用するか
            load_in_4bit: 4bit量子化を使用するか（より省メモリ）
            device_map: デバイスマッピング（"auto"で自動配置）
        """
        self.model_name = model_name
        
        print(f"モデルをロード中: {model_name}")
        print(f"量子化設定: 8bit={load_in_8bit}, 4bit={load_in_4bit}")
        
        # 量子化設定
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # ネストされた量子化でさらにメモリ節約
                bnb_4bit_quant_type="nf4"  # 正規化されたfloat4
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        
        # トークナイザーのロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # モデルのロード
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if not (load_in_4bit or load_in_8bit) else None
        )
        
        self.model.eval()  # 評価モード
        print("モデルのロードが完了しました！")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1
    ) -> str:
        """
        テキストを生成する
        
        Args:
            prompt: 入力プロンプト
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度（高いほど多様性が増す）
            top_p: nucleus sampling のパラメータ
            top_k: top-k sampling のパラメータ
            repetition_penalty: 繰り返しペナルティ
            
        Returns:
            生成されたテキスト
        """
        # トークン化
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # デコード
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # プロンプト部分を除いて返す
        return generated_text[len(prompt):].strip()


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="LLM推論スクリプト")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="モデル名またはパス"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="こんにちは、",
        help="入力プロンプト"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="生成する最大トークン数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="サンプリング温度"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="4bit量子化でロード（最も省メモリ）"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="8bit量子化でロード"
    )
    
    args = parser.parse_args()
    
    # デフォルトで8bit量子化を使用
    if not args.load_in_4bit and not args.load_in_8bit:
        args.load_in_8bit = True
    
    # 推論器の初期化
    inferencer = LLMInference(
        model_name=args.model,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    # テキスト生成
    print("\n" + "="*50)
    print(f"プロンプト: {args.prompt}")
    print("="*50)
    
    generated = inferencer.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    print(f"\n生成結果:\n{generated}\n")


if __name__ == "__main__":
    main()
