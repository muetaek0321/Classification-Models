from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# 定数
RESUTL_PATH = Path("./results/")
COLORS = ["r", "g", "b", "c", "m", "y"]


def main():
    # モデルごとの最新の学習結果ログをまとめる辞書
    summary_dict = {}
    
    for result_dir in RESUTL_PATH.iterdir():
        # ディレクトリ以外はスキップ
        if not result_dir.is_dir():
            continue
        
        # ディレクトリ名からモデル名のみを取得
        model_name = result_dir.name.split("_")[0]
        
        # 学習時のログの読み込み
        csv_path = result_dir.joinpath("training_log.csv")
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        
        # モデル名をキーとして保存
        summary_dict[model_name] = df
    
    # 各モデルの学習曲線をまとめてプロット
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    for i, (model_name, df) in enumerate(summary_dict.items()):
        # Train Loss
        ax[0, 0].set_title("Train Loss")
        ax[0, 0].plot(df["epoch"], df["train_loss"], c=COLORS[i], label=model_name)
        ax[0, 0].set_xlabel('Epoch')
        ax[0, 0].set_ylabel('Loss')
        ax[0, 0].legend()
        
        # Validation Loss
        ax[0, 1].set_title("Validation Loss")
        ax[0, 1].plot(df["epoch"], df["val_loss"], c=COLORS[i], label=model_name)
        ax[0, 1].set_xlabel('Epoch')
        ax[0, 1].set_ylabel('Loss')
        ax[0, 1].legend()
        
        # Train Accuracy
        ax[1, 0].set_title("Train Accuracy")
        ax[1, 0].plot(df["epoch"], df["train_accuracy"], c=COLORS[i], label=model_name)
        ax[1, 0].set_xlabel('Epoch')
        ax[1, 0].set_ylabel('Accuracy')
        ax[1, 0].legend()
        
        # Validation Accuracy
        ax[1, 1].set_title("Validation Accuracy")
        ax[1, 1].plot(df["epoch"], df["val_accuracy"], c=COLORS[i], label=model_name)
        ax[1, 1].set_xlabel('Epoch')
        ax[1, 1].set_ylabel('Accuracy')
        ax[1, 1].legend()
        
    plt.tight_layout()
    plt.savefig(RESUTL_PATH.joinpath("learning_curve.png"))
    
    plt.close()
        
        

if __name__ == "__main__":
    main()
