from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from schedulefree import RAdamScheduleFree


# エラー対処
matplotlib.use('Agg') 


class Trainer:
    """訓練を実行するクラス"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer | RAdamScheduleFree,
        criterion: nn.CrossEntropyLoss,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device | str,
        output_path: str | Path
    ) -> None:
        """コンストラクタ
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_path = Path(output_path)
        
        # 学習の準備
        self.model.to(device)
        self.criterion.to(device)
        self.best_model = None
        self.best_epoch = 0
        self.best_loss = np.inf
        
        # ログ保存の準備
        self.log = {"epoch": [], 
                    "train_loss": [], "val_loss": [],
                    "train_accuracy": [], "val_accuracy": []}
        
    def train(
        self,
        epoch: int
    ) -> float:
        """訓練のループを実行
        """
        self.model.train()
        self.optimizer.train()
        iter_train_loss = []
        pred_lbls, true_lbls = [], []
        
        for imgs, lbls in tqdm(self.train_dataloader, desc="train"):
            imgs, lbls = imgs.to(self.device), lbls.to(self.device)
            
            self.optimizer.zero_grad()

            output = self.model(imgs)
            
            loss = self.criterion(output, lbls)

            loss.backward()
            self.optimizer.step()
            
            iter_train_loss.append(loss.item())
            
            pred_lbls.extend(output.argmax(dim=1, keepdim=True).detach().cpu().numpy())
            true_lbls.extend(lbls.detach().cpu().numpy())
            
        # 1epochの平均lossを計算
        epoch_train_loss = np.mean(iter_train_loss)
        self.log["train_loss"].append(epoch_train_loss)
        epoch_train_acc = accuracy_score(true_lbls, pred_lbls)
        self.log["train_accuracy"].append(epoch_train_acc)
        self.log["epoch"].append(epoch)
            
        return epoch_train_loss, epoch_train_acc
    
    def validation(
        self,
        epoch: int
    ) -> float:
        """検証のループを実行
        """
        self.model.eval()
        self.optimizer.eval()
        iter_val_loss = []
        pred_lbls, true_lbls = [], []
        
        for imgs, lbls in tqdm(self.val_dataloader, desc="val"):
            imgs, lbls = imgs.to(self.device), lbls.to(self.device)

            with torch.no_grad():
                output = self.model(imgs)
                
            loss = self.criterion(output, lbls)
            
            iter_val_loss.append(loss.item())
            
            pred_lbls.extend(output.argmax(dim=1, keepdim=True).detach().cpu().numpy())
            true_lbls.extend(lbls.detach().cpu().numpy())
            
        # 1epochの平均lossを計算
        epoch_val_loss = np.mean(iter_val_loss)
        self.log["val_loss"].append(epoch_val_loss)
        epoch_val_acc = accuracy_score(true_lbls, pred_lbls)
        self.log["val_accuracy"].append(epoch_val_acc)
        
        # 最良のLossを判定
        if self.best_loss > epoch_val_loss:
            self.best_model = deepcopy(self.model)
            self.best_epoch = epoch
            self.best_loss = epoch_val_loss
            
        return epoch_val_loss, epoch_val_acc
        
    def save_weight(
        self
    ) -> None:
        """モデルの重みを保存
        """        
        # 最終epochのモデル
        epoch = self.log["epoch"][-1]
        model_name = f"{epoch}_latest.pth"
        torch.save(self.model.state_dict(), self.output_path.joinpath(model_name))
        print(f"model saved: {model_name}")
        
        # 最良のepochのモデル
        best_model_name = f"{self.best_epoch}_best.pth"
        torch.save(self.best_model.state_dict(), self.output_path.joinpath(best_model_name))
        print(f"best model saved: {best_model_name} (best loss: {self.best_loss})")  
    
    def output_learning_curve(
        self,
    ) -> None:
        """学習曲線の出力
        """
        epoch = len(self.log["epoch"]) # 現在までのエポック数を取得
        
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title(f"Loss (Epoch:{epoch})")
        ax1.plot(self.log["epoch"], self.log["train_loss"], c='red', label='train')
        ax1.plot(self.log["epoch"], self.log["val_loss"], c='blue', label='val')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title(f"Accuracy (Epoch:{epoch})")
        ax2.plot(self.log["epoch"], self.log["train_accuracy"], c='red', label='train')
        ax2.plot(self.log["epoch"], self.log["val_accuracy"], c='blue', label='val')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_path.joinpath("learning_curve.png"))
        
        plt.close()
        
    def output_log(
        self,
    ) -> None:
        """ログファイルの出力
        """
        # DataFrameに変換してcsvで出力
        log_df = pd.DataFrame(self.log)
        log_df.to_csv(self.output_path.joinpath("training_log.csv"),
                      encoding='utf-8-sig', index=False)
    
