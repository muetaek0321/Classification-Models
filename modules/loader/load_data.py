from pathlib import Path

import pandas as pd


def make_pathlist(
    dataset_path: Path,
    label_data_file: str = "train.csv",
    train_ratio: int = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """データセットフォルダを読み込み
       
    Args:
        dataset_path (str): データセットフォルダのルートパス
        split_ratio (int): 分割割合
    
    Returns:
        pd.DataFrame: 訓練データのパスとラベル
        pd.DataFrame: 検証データのパスとラベル
    """
    # 画像とラベルの対応表を読み込み
    label_df = pd.read_csv(dataset_path.joinpath(label_data_file), encoding="utf-8")
    
    # 画像パスとラベルの対応を作成
    train_dict, val_dict = {"path": [], "label": []}, {"path": [], "label": []}
    for lbl in label_df["target"].unique().tolist():
        unique_label_df = label_df[label_df["target"]==lbl]
        num_train_datas = int(len(unique_label_df) * train_ratio)
        
        for i, img_name in enumerate(unique_label_df["id"]):
            if i < num_train_datas:
                train_dict["path"].append(str(dataset_path.joinpath("train_data", img_name)))
                train_dict["label"].append(lbl)
            else:
                val_dict["path"].append(str(dataset_path.joinpath("train_data", img_name)))
                val_dict["label"].append(lbl)
                
    # DataFrameに変換して返す
    return pd.DataFrame(train_dict), pd.DataFrame(val_dict)
    