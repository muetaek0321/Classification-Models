import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(
    input_size: list[int],
    phase: str
) -> A.Compose:
    """DataAugmentationのインスタンスを作成
    
    Args:
        dataset_type (str): データセット形式
        phase (str): 訓練、検証、テストの指定
        
    Returns:
        A.Compose: DataAugmentationのインスタンス
    """
    if phase == "train":
        aug_list = [
            A.Blur(blur_limit=3),
            A.GaussNoise(var_limit=(2, 10)),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.Resize(*input_size),
            A.Normalize(),
            ToTensorV2()
        ]
        
    elif phase == "val":
        aug_list = [
            A.Resize(*input_size),
            A.Normalize(),
            ToTensorV2()
        ]
        
    elif phase == "test":
        aug_list = [
            A.Resize(*input_size),
            A.Normalize(),
            ToTensorV2()
        ]
    
    return A.Compose(aug_list)
