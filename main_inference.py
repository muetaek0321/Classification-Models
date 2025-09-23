import os
os.environ["HF_HOME"] = "./.cache/huggingface"
from pathlib import Path

import numpy as np
import cv2
import torch
import toml
from tqdm import tqdm
from datasets import load_dataset

from modules.Inference import Inference
from modules.utils import fix_seeds
from modules.models import get_model_inference
from modules.loader import ClassificationDataset


# 定数
CONFIG_PATH = "./config/inference_config.toml"
    
def main():
    # 乱数の固定
    fix_seeds()
    
    # 設定ファイルの読み込み
    with open(CONFIG_PATH, mode="r", encoding="utf-8") as f:
        cfg = toml.load(f)
    
    ## 入出力パス
    result_path = Path(cfg["train_result_path"])
    
    #デバイスの設定
    gpu = cfg["gpu"]
    if torch.cuda.is_available() and (gpu >= 0):
        device = torch.device(f"cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        device = torch.device("cpu")
    print(f"使用デバイス {device}")
    
    # 学習時のconfigから必要なパラメータを取得
    with open(result_path.joinpath("train_config.toml"), mode="r", encoding="utf-8") as f:
        cfg_t = toml.load(f)
    model_name = cfg_t["model_name"]
    input_size = cfg_t["parameters"]["input_size"]
    
    # Datasetの読み込み
    # https://huggingface.co/datasets/Bingsu/Human_Action_Recognition
    dataset = load_dataset("Bingsu/Human_Action_Recognition")
    test_dataset = ClassificationDataset(dataset["test"], input_size, phase="test")
    classes = dataset["train"].features["labels"].names
    num_classes = dataset["train"].features["labels"].num_classes
    
    # 自作モデルを使用
    weight_path = list(result_path.glob("*best.pth"))[0]
    model = get_model_inference(model_name, num_classes, weight_path, device)
    
    # 推論クラスの定義
    infer = Inference(
        model=model,
        input_size=input_size,
        classes=classes,
        device=device,
        output_path=result_path
    )
    
    # 画像を1枚ずつ推論
    for i in tqdm(range(len(test_dataset)), desc="inference"):
        input_img, _ = test_dataset[i]
        ori_img = np.array(dataset["test"]["image"][i])
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        # 推論
        infer(input_img, ori_img)
        

if __name__ == "__main__":
    main()