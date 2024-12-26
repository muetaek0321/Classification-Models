import os
os.environ["TORCH_HOME"] = "pretrained"
from pathlib import Path
import shutil

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import AdamW
import toml
from schedulefree import RAdamScheduleFree

from modules.utils import fix_seeds, now_date_str, ProcessTimeManager
from modules.loader import make_pathlist, ClassificationDataset
from modules.models import get_model_train
from modules.trainer import Trainer


# 定数
CONFIG_PATH = "./config/train_config.toml"
    
def main():
    # 乱数の固定
    fix_seeds()
    
    # 設定ファイルの読み込み
    with open(CONFIG_PATH, mode="r", encoding="utf-8") as f:
        cfg = toml.load(f)
        
    ## モデル名称
    model_name = cfg["model_name"]
    use_pretrained = cfg["use_pretrained"]
    
    ## 入出力パス
    input_path = Path(cfg["input_path"])
    output_path = Path(cfg["output_path"]).joinpath(f"{model_name}_{now_date_str()}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    ## 各種パラメータ
    num_epoches = cfg["parameters"]["num_epoches"]
    batch_size = cfg["parameters"]["batch_size"]
    classes = cfg["parameters"]["classes"]
    num_classes = len(classes)
    input_size = cfg["parameters"]["input_size"]
    lr = cfg["optimizer"]["lr"]
    weight_decay = cfg["optimizer"]["weight_decay"]
    
    #デバイスの設定
    gpu = cfg["gpu"]
    if torch.cuda.is_available() and (gpu >= 0):
        device = torch.device(f"cuda:{gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        device = torch.device("cpu")
    print(f"使用デバイス {device}")
    
    # データのパスリストを作成
    train_df, val_df = make_pathlist(input_path)
    print(f"データ分割 train:val = {len(train_df)}:{len(val_df)}")
    # Datasetの作成
    train_dataset = ClassificationDataset(train_df["path"].tolist(), 
                                          train_df["label"].tolist(), 
                                          input_size, phase="train")
    val_dataset = ClassificationDataset(val_df["path"].tolist(), 
                                        val_df["label"].tolist(),
                                        input_size, phase="val")
    
    # DataLoaderの作成
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, 
                                  num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, 
                                num_workers=0, pin_memory=True)
    
    # モデルの定義
    model, params = get_model_train(model_name, num_classes, use_pretrained)

    # optimizerの定義
    # optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    optimizer = RAdamScheduleFree(params, lr=lr)
    
    # 損失関数の定義
    criterion = CrossEntropyLoss()
    
    # Trainerの定義
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        output_path=output_path
    )
    
    # configを保存
    shutil.copy2(CONFIG_PATH, output_path)
    
    # 学習ループを実行
    for i in range(num_epoches):
        epoch = i + 1
        
        # 訓練
        train_loss, train_acc = trainer.train(epoch)
        # 検証
        val_loss, val_acc = trainer.validation(epoch)
        
        # ログの標準出力
        print(f"Epoch:{epoch}")
        print(f"  train_loss:{train_loss:.4f}  val_loss:{val_loss:.4f}")
        print(f"  train_accuracy:{train_acc:.4f}  val_accuracy:{val_acc:.4f}")
        
        # 学習の進捗を出力
        trainer.output_learning_curve()
    
    # モデルとログの出力
    trainer.save_weight()
    trainer.output_log()


if __name__ == "__main__":
    with ProcessTimeManager(is_print=True) as pt:
        main()
    