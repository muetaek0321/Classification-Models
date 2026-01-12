from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import timm


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
    # モデル読み込みと出力の設定変更
    if model_name == "EfficientNetV2":
        model = timm.create_model(
            'efficientnetv2_rw_m.agc_in1k', 
            pretrained=use_pretrained, num_classes=num_classes
        )
    if model_name == "SEResNeXt":
        model = timm.create_model(
            'timm/seresnext50_32x4d.racm_in1k', 
            pretrained=use_pretrained, num_classes=num_classes
        )
    elif model_name == "VisionTransformer":
        model = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=use_pretrained, num_classes=num_classes
        )
    elif model_name == "DeiT":
        model = timm.create_model(
            'deit_base_distilled_patch16_224', 
            pretrained=use_pretrained, num_classes=num_classes
        )
    
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
    # モデル読み込みと出力の設定変更
    if model_name == "EfficientNetV2":
        model = timm.create_model(
            'timm/efficientnetv2_rw_m.agc_in1k', 
            pretrained=False, num_classes=num_classes
        )
    elif model_name == "SEResNeXt":
        model = timm.create_model(
            'timm/seresnext50_32x4d.racm_in1k', 
            pretrained=False, num_classes=num_classes
        )
    elif model_name == "VisionTransformer":
        model = timm.create_model(
            'timm/vit_base_patch16_224.augreg2_in21k_ft_in1k', 
            pretrained=False, num_classes=num_classes
        )
    elif model_name == "DeiT":
        model = timm.create_model(
            'timm/deit_base_distilled_patch16_224.fb_in1k', 
            pretrained=False, num_classes=num_classes
        )
        
    # 学習済みモデルの読み込み
    weight = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(weight)
        
    return model
