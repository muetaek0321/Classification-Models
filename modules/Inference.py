import os
os.environ["TORCH_HOME"] = "pretrained"
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

from .loader.augmentation import get_transform
from .utils import imread_jpn


class Inference:
    """推論を実行するクラス"""
    
    def __init__(
        self,
        model: nn.Module,
        input_size: list[int],
        device: torch.device | str,
        output_path: str | Path
    ) -> None:
        """コンストラクタ
        """
        self.model = model
        self.input_size = input_size
        self.device = device
        self.output_path = Path(output_path)
        
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
        img_path: Path
    ) -> float:
        """1画像で推論の処理を実行
        """
        # 画像読み込み
        img = imread_jpn(img_path)
        # 判定画像のファイル名を保存しておく
        self.img_name_list.append(img_path.name)
        
        # 画像の前処理を適用
        input_img = self.transform(image=img)['image']
        input_img = input_img.to(self.device)
        input_img = input_img.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_img)
        
        # 予測ラベルを取得
        pred_lbl = output.argmax(dim=1, keepdim=True).detach().cpu().numpy()[0]
        self.pred_lbls.extend(pred_lbl)
        
    def output_result(
        self,
        result_file_name: str = "submission"
    ) -> None:
        """判定結果をcsvファイルで出力
        
        Args:
            result_file_name (str): 出力ファイル名
        """
        # 結果をDF化
        df = pd.DataFrame({"img": self.img_name_list, "lbl": self.pred_lbls,
                           "img_number": [int(Path(img_name).stem) for img_name in self.img_name_list]})
        # 画像名でソート
        df_sorted = df.sort_values(by='img_number').loc[:, ["img", "lbl"]]
        
        # 出力
        df_sorted.to_csv(self.output_path.joinpath(f"{result_file_name}.csv"),
                         encoding='utf-8', index=False, header=False)
        
    
