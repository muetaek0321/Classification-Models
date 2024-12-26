from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .augmentation import get_transform
from modules.utils import imread_jpn


__all__ = ["ClassificationDataset"]


class ClassificationDataset(Dataset):
    """画像分類用データセットクラス"""
    
    def __init__(
        self,
        img_path_list: list[Path],
        label_list: list[Path],
        input_size: list[int],
        phase: str
    ) -> None:
        """
        
        Args:
            img_path_list (list): 画像パスのリスト
            label_list (list): ラベルのリスト
            input_size (list): 入力画像サイズ
            phase (str): データセットの種類
        """
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.input_size = input_size
        self.phase = phase
        
        # DataAugmentationの準備
        self.transform = get_transform(self.input_size, self.phase)
        
    def __len__(
        self
    ) -> int:
        """Datasetの長さを返す"""
        return len(self.img_path_list)
    
    def __getitem__(
        self, 
        index: int
    ) -> tuple[np.ndarray, dict]:
        """index指定でデータを返す
        
        Args:
            index (int): Datasetのインデックス
            
        Returns:
            np.ndarray: 画像
            list: アノテーション
        """
        # 画像とアノテーションのパスを取得
        img_path, lbl = self.img_path_list[index], self.label_list[index]
        
        # 画像読み込み
        img = imread_jpn(img_path)
            
        # DataAugmentationの適用
        transformed = self.transform(image=img)
        img_trans = transformed['image']
        
        return img_trans, lbl
        
        
        

