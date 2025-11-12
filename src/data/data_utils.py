"""
データ準備ユーティリティ

学習データの作成・変換・検証を行うツール
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import random


class DatasetBuilder:
    """
    データセット作成クラス
    """
    
    @staticmethod
    def create_conversation_dataset(
        pairs: List[tuple],
        output_path: str,
        format: str = "json"
    ):
        """
        会話ペアからデータセットを作成
        
        Args:
            pairs: (質問, 回答)のタプルのリスト
            output_path: 出力ファイルパス
            format: 出力形式 ("json" or "jsonl")
        """
        dataset = []
        for question, answer in pairs:
            dataset.append({
                "text": f"質問: {question}\n答え: {answer}"
            })
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
        elif format == "jsonl":
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"サポートされていない形式: {format}")
        
        print(f"データセットを作成しました: {output_path}")
        print(f"サンプル数: {len(dataset)}")
    
    @staticmethod
    def create_pattern_dataset(
        patterns: List[tuple],
        output_path: str,
        format: str = "json"
    ):
        """
        パターンデータセットを作成（形容詞＋やね用）
        
        Args:
            patterns: (入力, 出力)のタプルのリスト
            output_path: 出力ファイルパス
            format: 出力形式
        """
        dataset = []
        for input_text, output_text in patterns:
            dataset.append({
                "text": f"入力: {input_text}\n出力: {output_text}"
            })
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
        elif format == "jsonl":
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"パターンデータセットを作成しました: {output_path}")
        print(f"サンプル数: {len(dataset)}")
    
    @staticmethod
    def split_dataset(
        input_path: str,
        train_ratio: float = 0.8,
        output_dir: str = "./data"
    ):
        """
        データセットを訓練用と検証用に分割
        
        Args:
            input_path: 入力データセットのパス
            train_ratio: 訓練データの割合（0.0-1.0）
            output_dir: 出力ディレクトリ
        """
        # データの読み込み
        with open(input_path, 'r', encoding='utf-8') as f:
            if input_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        
        # シャッフル
        random.shuffle(data)
        
        # 分割
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # 出力ディレクトリ作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存
        train_file = output_path / "train.json"
        val_file = output_path / "val.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        print(f"データセットを分割しました:")
        print(f"  訓練データ: {train_file} ({len(train_data)} サンプル)")
        print(f"  検証データ: {val_file} ({len(val_data)} サンプル)")
    
    @staticmethod
    def validate_dataset(data_path: str) -> Dict[str, Any]:
        """
        データセットを検証
        
        Args:
            data_path: データセットのパス
            
        Returns:
            検証結果の辞書
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        
        # 統計情報
        total_samples = len(data)
        text_lengths = [len(item.get('text', '')) for item in data]
        avg_length = sum(text_lengths) / total_samples if total_samples > 0 else 0
        
        # 検証
        issues = []
        for i, item in enumerate(data):
            if 'text' not in item:
                issues.append(f"サンプル {i}: 'text'フィールドがありません")
            elif not item['text'].strip():
                issues.append(f"サンプル {i}: テキストが空です")
        
        result = {
            "total_samples": total_samples,
            "average_text_length": avg_length,
            "min_text_length": min(text_lengths) if text_lengths else 0,
            "max_text_length": max(text_lengths) if text_lengths else 0,
            "issues": issues,
            "valid": len(issues) == 0
        }
        
        print(f"データセット検証結果: {data_path}")
        print(f"  総サンプル数: {result['total_samples']}")
        print(f"  平均テキスト長: {result['average_text_length']:.1f}")
        print(f"  最小テキスト長: {result['min_text_length']}")
        print(f"  最大テキスト長: {result['max_text_length']}")
        print(f"  検証: {'合格' if result['valid'] else '問題あり'}")
        
        if not result['valid']:
            print("\n問題:")
            for issue in issues:
                print(f"  - {issue}")
        
        return result


def main():
    """デモンストレーション"""
    import argparse
    
    parser = argparse.ArgumentParser(description="データセットユーティリティ")
    parser.add_argument(
        "action",
        choices=["validate", "split"],
        help="実行するアクション"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="入力ファイルパス"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="出力ディレクトリ（splitの場合）"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="訓練データの割合（splitの場合）"
    )
    
    args = parser.parse_args()
    
    builder = DatasetBuilder()
    
    if args.action == "validate":
        builder.validate_dataset(args.input)
    elif args.action == "split":
        builder.split_dataset(args.input, args.train_ratio, args.output)


if __name__ == "__main__":
    main()
