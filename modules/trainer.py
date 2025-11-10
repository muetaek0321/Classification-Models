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
import seaborn as sns
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
        classes: list[str],
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
        self.classes = classes
        self.device = device
        self.output_path = Path(output_path)
        
        # 学習の準備
        self.model.to(device)
        self.criterion.to(device)
        self.best_loss = np.inf
        
        # ログ保存の準備
        self.log = {"epoch": [], 
                    "train_loss": [], "val_loss": [],
                    "train_accuracy": [], "val_accuracy": []}
        
        # 途中再開時の開始epochを保持
        self.resume_epoch = 0
        
    def resume_state(
        self,
        resume_epoch: int = 0
    ) -> None:
        """学習途中の状態に復元
        """
        self.resume_epoch = resume_epoch
        
        # 以前のログを読み込み
        log_path = self.output_path.joinpath("training_log.csv")
        log_df = pd.read_csv(log_path, encoding='utf-8-sig')
        self.log = log_df.to_dict(orient="list")
        # 最良のLossを復元
        self.best_loss = log_df["val_loss"].min()
        
        # 読み込んだoptimizerのstateをGPUに渡す
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        
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
        
        # 混同行列の出力
        self.output_confusion_matrix(true_lbls, pred_lbls)
        
        # 最良のLossを判定
        if self.best_loss > epoch_val_loss:
            self.best_loss = epoch_val_loss
            # 最良モデルの保存
            self.save_weight_best(epoch)
            
        # 最新モデルの保存
        self.save_weight_latest(epoch)
            
        return epoch_val_loss, epoch_val_acc
        
    def save_weight_best(
        self,
        epoch: int
    ) -> None:
        """モデルの重みを保存
        """         
        # 最良のepochのモデル
        best_model_name = "model_best.pth"
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, self.output_path.joinpath(best_model_name))
        print(f"best model saved: {best_model_name} (best loss: {self.best_loss})")  
        
    def save_weight_latest(
        self,
        epoch: int
    ) -> None:
        """モデルの重みを保存
        """        
        # 最終epochのモデル
        model_name = "model_latest.pth"
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, self.output_path.joinpath(model_name))
        print(f"model saved: {model_name}")
    
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
        if self.resume_epoch > 0:
            ax1.axvline(self.resume_epoch, 0, 1, color='gray', linestyle='dotted')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title(f"Accuracy (Epoch:{epoch})")
        ax2.plot(self.log["epoch"], self.log["train_accuracy"], c='red', label='train')
        ax2.plot(self.log["epoch"], self.log["val_accuracy"], c='blue', label='val')
        if self.resume_epoch > 0:
            ax2.axvline(self.resume_epoch, 0, 1, color='gray', linestyle='dotted')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_path.joinpath("learning_curve.png"))
        
        plt.close()
        
    def output_confusion_matrix(
        self,
        true: list,
        pred: list
    ) -> None:
        """混同行列の出力
        """
        cm = confusion_matrix(true, pred, labels=range(len(self.classes)))
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=cm, fmt="d")
        
        plt.tight_layout()
        plt.savefig(self.output_path.joinpath("confusion_matrix.png"))
        
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
    
