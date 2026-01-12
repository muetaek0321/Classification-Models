import os
os.environ["TORCH_HOME"] = "./.cache/torch"
os.environ["HF_HOME"] = "./.cache/huggingface"
from pathlib import Path
import shutil

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import toml
from schedulefree import RAdamScheduleFree
from datasets import load_dataset

from modules.utils import fix_seeds, now_date_str, ProcessTimeManager, make_train_parser
from modules.loader import ClassificationDataset
from modules.models import get_model_train
from modules.trainer import Trainer


def main() -> None:
    with ProcessTimeManager(is_print=True) as pt:
        output_path, device = train()
    
    # 学習全体の実行時間とGPUメモリの使用量を記録
    with open(output_path.joinpath("process_log.txt"), mode="w", encoding="cp932") as f:
        f.write(f"学習全体の実行時間: {pt.proc_time:.1f}s\n")
        f.write(f"GPUメモリ使用量: {torch.cuda.max_memory_allocated(device)/1024**3:.3f}GB\n")

    
def train() -> tuple[Path, torch.device]:
    args = make_train_parser()
    
    # 乱数の固定
    fix_seeds()
    
    # 設定ファイルの読み込み
    with open(args.config, mode="r", encoding="utf-8") as f:
        cfg = toml.load(f)
        
    ## モデル名称
    model_name = cfg["model_name"]
    use_pretrained = cfg["use_pretrained"]
    
    ## 出力パス
    output_path = Path(cfg["output_path"]).joinpath(f"{model_name}_{now_date_str()}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    ## 各種パラメータ
    num_epoches = cfg["parameters"]["num_epoches"]
    batch_size = cfg["parameters"]["batch_size"]
    input_size = cfg["parameters"]["input_size"]
    lr = cfg["optimizer"]["lr"]
    
    #デバイスの設定
    gpu = cfg["gpu"]
    if torch.cuda.is_available() and (gpu >= 0):
        device = torch.device("cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        device = torch.device("cpu")
    print(f"使用デバイス {device}")
    
    # Datasetの読み込み
    # https://huggingface.co/datasets/SunnyAgarwal4274/Food_and_Vegetables
    dataset = load_dataset("imagefolder", data_dir="./dataset")
    
    # Datasetの作成
    train_dataset = ClassificationDataset(dataset["train"], input_size, phase="train")
    val_dataset = ClassificationDataset(dataset["validation"], input_size, phase="val")
    print(f"データ分割 train:val = {len(train_dataset)}:{len(val_dataset)}")
    
    # DataLoaderの作成
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, 
                                  num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, 
                                num_workers=0, pin_memory=True)
    
    # モデルの定義
    model, params = get_model_train(
        model_name=model_name, 
        num_classes=train_dataset.num_classes, 
        use_pretrained=use_pretrained
    )

    # optimizerの定義
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
        classes=train_dataset.classes,
        device=device,
        output_path=output_path
    )
    
    # configを保存
    shutil.copy2(args.config, output_path)
    
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
    trainer.output_log()
    
    return output_path, device


if __name__ == "__main__":
    main()
        