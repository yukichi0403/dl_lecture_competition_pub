import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import *
from src.models import *
from src.utils import *

from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

import shutil
import gc


def get_LR_scheduler(optimizer, config):
    if config.LR_scheduler is None:
        return None
    if config.LR_scheduler == "OneCycleLR":
        return OneCycleLR(optimizer, **config.LR_scheduler_params)
    elif config.LR_scheduler == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, **config.LR_scheduler_params)
    elif config.LR_scheduler == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, **config.LR_scheduler_params)
    elif config.LR_scheduler == "StepLR":
        return StepLR(optimizer, **config.LR_scheduler_params)
    elif config.LR_scheduler == "CosineAnnealingWarmRestarts":
        return get_cosine_schedule_with_warmup(optimizer, **config.LR_scheduler_params)
    else:
        raise ValueError(f"Invalid LR scheduler: {config.LR_scheduler}")

def get_labels(dataset):
    labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        labels.append(label)
    return np.array(labels)

@hydra.main(version_base=None, config_path="configs", config_name="config_effnet0b_holdout")
def run(args: DictConfig):
    print("Config loaded: ", args)  # 読み込まれた設定の内容を表示
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    # 例として、TrainとValidのデータセットを読み込み
    print(f"Now Loading Train/Valid Datasets")
    train_set = ThingsMEGDataset("train", args.data_dir)
    val_set = ThingsMEGDataset("val", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    print(f"Train shape: {len(train_set)}, Val shape: {len(val_set)}.")
    
    # ------------------
    #       Model
    # ------------------
    model = CustomModel(
        args.backbone, args.num_classes, True, args.aux_loss_ratio
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ------------------
    #     LR scheduler
    # ------------------
    lr_scheduler = get_LR_scheduler(optimizer, args)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    no_improve_epochs = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=args.num_classes, top_k=10
    ).to(args.device)
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)

            y_pred, y_pred_aux = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            if args.aux_loss_ratio is not None:
                aux_loss = F.cross_entropy(y_pred_aux, y)
                total_loss = loss + args.aux_loss_ratio * aux_loss
            else:
                total_loss = loss
            
            train_loss.append(total_loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y, subject_idxs = X.to(args.device), y.to(args.device), subject_idxs.to(args.device)
            
            with torch.no_grad():
                if args.aux_loss_ratio is not None:
                    y_pred, y_pred_aux = model(X)
                else:
                    y_pred = model(X)
            
            if args.aux_loss_ratio is not None:
                total_val_loss = F.cross_entropy(y_pred, y) + args.aux_loss_ratio * F.cross_entropy(y_pred_aux, y)
                val_loss.append(total_val_loss.item())
            else:
                total_val_loss = F.cross_entropy(y_pred, y)
                val_loss.append(total_val_loss.item())
            
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
    
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
    
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, f"model_best.pt"))
            max_val_acc = np.mean(val_acc)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs > args.early_stopping_rounds:
                cprint("Early stopping.", "cyan")
                break

    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------        
    print("Now Loading Test Datasets")
    test_set = ThingsMEGDataset("test", args.data_dir)
    
    print(f"Test size: {len(test_set)}")
        
    test_loader = DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = CustomModel(
        args.backbone,
        args.num_classes,
        True,
        args.aux_loss_ratio
    ).to(args.device)
    model.load_state_dict(torch.load(os.path.join(logdir, f"model_best.pt"), map_location=args.device))
    
    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        with torch.no_grad():
            if args.aux_loss_ratio is not None:
                y_pred, _ = model(X.to(args.device))
            else:
                y_pred = model(X.to(args.device))
            preds.append(y_pred.detach().cpu())
    
    preds = torch.cat(preds, dim=0).numpy()

    np.save(os.path.join(logdir, "submission.npy"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

    # Google Driveにファイルをコピー
    drive_dir = os.path.join(args.data_dir, f"{args.expname}_{args.ver}")
    os.makedirs(drive_dir, exist_ok=True)

    submission_path = os.path.join(logdir, "submission.npy")
    if os.path.exists(submission_path):
        shutil.copy(submission_path, drive_dir)
        print(f'Submission file saved to Google Drive: {drive_dir}')



if __name__ == "__main__":
    run()
