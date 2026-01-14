import os
os.environ["TORCH_HOME"] = "pretrained"
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from .loader.augmentation import get_transform
from modules.utils import imwrite_jpn


class Inference:
    """推論を実行するクラス"""
    
    def __init__(
        self,
        model: nn.Module,
        input_size: list[int],
        classes: list[str],
        device: torch.device | str,
        output_path: str | Path
    ) -> None:
        """コンストラクタ
        """
        self.model = model
        self.input_size = input_size
        self.classes = classes
        self.device = device
        self.output_path = Path(output_path).joinpath("inference")
        
        # 学習の準備
        self.model.to(device)
        self.model.eval()
        
        # DataAugmentation
        self.transform = get_transform(self.input_size, "test")
        
        # 結果保存
        self.img_name_list = []
        self.pred_lbls = []
        
    def __call__(
        self,
        input_img: np.ndarray,
        ori_img: np.ndarray
    ) -> float:
        """1画像で推論の処理を実行
        """        
        input_img = input_img.to(self.device)
        input_img = input_img.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_img)
        
        # 予測ラベルを取得
        pred_lbl = output.argmax(dim=1, keepdim=True).detach().cpu().numpy()[0]
        self.pred_lbls.extend(pred_lbl)
        
        self.classification_image(ori_img, pred_lbl[0])
        
    def feature_extraction(
        self,
        input_img: np.ndarray,
    ) -> np.ndarray:
        """1画像で特徴量抽出の処理を実行
        """
        input_img = input_img.to(self.device)
        input_img = input_img.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model.forward_features(input_img)
        
        return output.detach().cpu().numpy()[0]
        
        
    def classification_image(
        self,
        img: np.ndarray,
        pred_lbl: int
    ) -> None:
        """分類画像を保存"""
        class_dir = self.output_path.joinpath(self.classes[pred_lbl])
        class_dir.mkdir(exist_ok=True, parents=True)
        
        if len(list(class_dir.iterdir())) >= 10:
            return
        
        img_name = f"{len(self.pred_lbls):06}.jpg"
        img_path = class_dir.joinpath(img_name)
        
        # 画像保存
        imwrite_jpn(img_path, img)
        
    # def output_result(
    #     self,
    #     result_file_name: str = "submission"
    # ) -> None:
    #     """判定結果をcsvファイルで出力
        
    #     Args:
    #         result_file_name (str): 出力ファイル名
    #     """
    #     # 結果をDF化
    #     df = pd.DataFrame({"img": self.img_name_list, "lbl": self.pred_lbls,
    #                        "img_number": [int(Path(img_name).stem) for img_name in self.img_name_list]})
    #     # 画像名でソート
    #     df_sorted = df.sort_values(by='img_number').loc[:, ["img", "lbl"]]
        
    #     # 出力
    #     df_sorted.to_csv(self.output_path.joinpath(f"{result_file_name}.csv"),
    #                      encoding='utf-8', index=False, header=False)
        
    
