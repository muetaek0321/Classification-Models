from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s


def get_model_train(
    model_name: str, 
    num_classes: int, 
    use_pretrained: bool,
    transfer: bool = False,
    finetune: bool = False,
) -> tuple[nn.Module, Any]:
    """画像分類モデルの準備
    
    Args:
        model_name (str): モデルの名前
        num_classes (int): 判定クラス数
        use_pretrained (bool): 事前学習モデルを使用するかの設定
        transfer (bool): 転移学習の設定
        finetune (bool): ファインチューニングの設定
        
    Returns:
        nn.Module: モデルアーキテクチャ
        
    """
    if model_name == "EfficientNetV2":
        # モデル読み込みと出力の設定変更
        model = model_efficientnet_v2(num_classes)
        # モデルパラメータの取得
        params = model.parameters()
        
    return model, params


def get_model_inference(
    model_name: str,
    num_classes: str,
    weight_path: str | Path,
    device: torch.device
) -> nn.Module:
    """画像分類モデルの準備
    
    Args:
        model_name (str): モデルの名前
        num_classes (int): 判定クラス数
        weight_path (str,Path): 学習済みモデルのパス
        device (torch.device): 使用デバイス
        
    Returns:
        nn.Module: モデルアーキテクチャ
    """
    if model_name == "EfficientNetV2":
        # モデル読み込みと出力の設定変更
        model = model_efficientnet_v2(num_classes)
        # 学習済みモデルの読み込み
        weight = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(weight)
        
    return model


def model_efficientnet_v2(
    num_classes: int,
) -> nn.Module:
    """EfficientNetV2
    
    Args:
        num_classes (int): 判定クラス数
    """
    model = efficientnet_v2_s(weights='DEFAULT')
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model