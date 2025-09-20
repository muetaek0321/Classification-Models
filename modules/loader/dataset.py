import numpy as np
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset

from .augmentation import get_transform
from modules.utils import imread_jpn


__all__ = ["ClassificationDataset"]


class ClassificationDataset(Dataset):
    """画像分類用データセットクラス"""
    
    def __init__(
        self,
        dataset: HFDataset,
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
        self.dataset = dataset
        self.input_size = input_size
        self.phase = phase
        
        self.classes = dataset.features["labels"].names
        self.num_classes = dataset.features["labels"].num_classes
        
        # DataAugmentationの準備
        self.transform = get_transform(self.input_size, self.phase)
        
    def __len__(
        self
    ) -> int:
        """Datasetの長さを返す"""
        return len(self.dataset)
    
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
        # 画像とラベルを取得
        data = self.dataset[index]
        img, lbl = data["image"], data["labels"]
            
        # DataAugmentationの適用
        transformed = self.transform(image=np.array(img))
        img_trans = transformed['image']
        
        return img_trans, lbl
        
        
        

