"""
デモスクリプト - 3つのAIモデルの使い方を示す例

このスクリプトは3種類のAIモデルの基本的な使い方を示します。
実際に実行するには、モデルのダウンロードと学習が必要です。
"""

import sys
import os

# パスの設定
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.model_manager import ModelManager


def demo_keigo_model(manager: ModelManager):
    """敬語AIのデモ"""
    print("\n" + "="*60)
    print("1. 敬語AI - 丁寧な敬語で話します")
    print("="*60)
    
    try:
        # モデル情報の表示
        info = manager.get_model_info("keigo")
        print(f"モデル名: {info['name']}")
        print(f"説明: {info['description']}")
        print(f"ベースモデル: {info['base_model']}")
        
        # モデルのロード（実際には時間がかかります）
        print("\nモデルをロード中...")
        print("（注意: 初回は数分かかる場合があります）")
        # manager.load_model("keigo")
        
        # 推論例
        print("\n推論例:")
        test_inputs = [
            "今日はいい天気ですね",
            "このプロジェクトについて教えてください",
            "ありがとうございます"
        ]
        
        for input_text in test_inputs:
            print(f"\n入力: {input_text}")
            # result = manager.generate("keigo", input_text)
            # print(f"出力: {result}")
            print("（モデルロード後に生成されます）")
            
    except Exception as e:
        print(f"エラー: {e}")
        print("敬語モデルは学習済みモデルを使用します。")
        print("コマンド: python src/inference/inference.py --model rinna/japanese-gpt-neox-3.6b --prompt '今日はいい天気ですね' --load-in-8bit")


def demo_kansai_model(manager: ModelManager):
    """関西弁AIのデモ"""
    print("\n" + "="*60)
    print("2. 関西弁AI - 関西弁で話します")
    print("="*60)
    
    info = manager.get_model_info("kansai")
    print(f"モデル名: {info['name']}")
    print(f"説明: {info['description']}")
    
    print("\nこのモデルは学習が必要です。")
    print("学習手順:")
    print("1. 学習データの準備:")
    print("   examples/kansai_data.json を参照")
    print("2. 学習の実行:")
    print("   python src/training/train.py \\")
    print("     --base-model rinna/japanese-gpt-neox-3.6b \\")
    print("     --data-path examples/kansai_data.json \\")
    print("     --output-dir ./models/kansai_finetuned \\")
    print("     --epochs 3 --use-qlora")
    print("3. 推論の実行:")
    print("   python src/inference/inference.py \\")
    print("     --model ./models/kansai_finetuned/final_model \\")
    print("     --prompt '今日はいい天気ですね' --load-in-8bit")
    
    print("\n学習データの例:")
    print("  入力: 今日はいい天気ですね")
    print("  期待される出力: ほんまにええ天気やな。気持ちええわ。")


def demo_adjective_model(manager: ModelManager):
    """形容詞＋やねAIのデモ"""
    print("\n" + "="*60)
    print("3. 形容詞＋やねAI - 形容詞＋やねで話します")
    print("="*60)
    
    info = manager.get_model_info("adjective")
    print(f"モデル名: {info['name']}")
    print(f"説明: {info['description']}")
    
    print("\nこのモデルも学習が必要です。")
    print("学習手順:")
    print("1. 学習データの準備:")
    print("   examples/adjective_data.json を参照")
    print("2. 学習の実行:")
    print("   python src/training/train.py \\")
    print("     --base-model rinna/japanese-gpt-neox-3.6b \\")
    print("     --data-path examples/adjective_data.json \\")
    print("     --output-dir ./models/adjective_finetuned \\")
    print("     --epochs 5 --use-qlora")
    print("3. 推論の実行:")
    print("   python src/inference/inference.py \\")
    print("     --model ./models/adjective_finetuned/final_model \\")
    print("     --prompt 'この花を見て' --load-in-8bit")
    
    print("\n学習データの例:")
    print("  入力: この花を見て")
    print("  期待される出力: きれいやね")
    print("  入力: あの人の笑顔")
    print("  期待される出力: うれしそうやね")


def main():
    """メイン関数"""
    print("="*60)
    print("日本語LLM学習プロジェクト - デモスクリプト")
    print("="*60)
    print("\nこのスクリプトは3種類のAIモデルの使い方を示します。")
    print("実際にモデルを実行するには、各モデルのロードと学習が必要です。")
    
    # モデルマネージャーの初期化
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    manager = ModelManager(config_dir=config_dir)
    
    # 利用可能なモデルの表示
    print("\n利用可能なモデル:")
    for model_type, model_name in manager.list_available_models().items():
        print(f"  - {model_type}: {model_name}")
    
    # 各モデルのデモ
    demo_keigo_model(manager)
    demo_kansai_model(manager)
    demo_adjective_model(manager)
    
    print("\n" + "="*60)
    print("デモ終了")
    print("="*60)
    print("\n次のステップ:")
    print("1. requirements.txt から依存関係をインストール")
    print("2. 敬語AIを試す（学習不要）")
    print("3. 関西弁AIと形容詞＋やねAIのデータを準備して学習")
    print("\n詳細はREADME.mdを参照してください。")


if __name__ == "__main__":
    main()
