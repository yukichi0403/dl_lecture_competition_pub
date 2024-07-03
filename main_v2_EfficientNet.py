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

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed

from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
import shutil


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
        return CosineAnnealingWarmRestarts(optimizer, **config.LR_scheduler_params)
    else:
        raise ValueError(f"Invalid LR scheduler: {config.LR_scheduler}")


@hydra.main(version_base=None, config_path="configs", config_name="config_colab")
def run(args: DictConfig):
    print("Config loaded: ", args)  # 読み込まれた設定の内容を表示
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    # 例として、TrainとValidのデータセットを読み込み
    print(f"Now Loading Train/Valid Datasets")
    train_set = ThingsMEGDataset("train", args.data_dir)
    valid_set = ThingsMEGDataset("valid", args.data_dir)
    # TrainとValidを結合
    combined_dataset = ConcatDataset([train_set, valid_set])
    X = []; y = []
    for data in combined_dataset:
        X.append(data[0])
        y.append(data[1])
    X = torch.stack(X)
    y = np.array(y)
    
    skf = StratifiedKFold(n_splits=args.num_splits, shuffle=True, random_state=args.seed)
        
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"Now training for fold {fold+1}/{args.num_splits}")
        # トレーニングとテストのデータセットを作成
        train_subsampler = Subset(combined_dataset, train_index)
        val_subsampler = Subset(combined_dataset, val_index)
        
        train_loader = DataLoader(train_subsampler,  shuffle=True, **loader_args)
        val_loader = DataLoader(val_subsampler,  shuffle=False, **loader_args)
        

        # ------------------
        #       Model
        # ------------------
        model = BasicConvClassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)

        # ------------------
        #     Optimizer
        # ------------------
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
            task="multiclass", num_classes=train_set.num_classes, top_k=10
        ).to(args.device)
      
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            
            train_loss, train_acc, val_loss, val_acc = [], [], [], []
            
            model.train()
            for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
                X, y = X.to(args.device), y.to(args.device)

                y_pred = model(X)
                
                loss = F.cross_entropy(y_pred, y)
                train_loss.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                acc = accuracy(y_pred, y)
                train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, f"model_best_{fold}.pt"))
            max_val_acc = np.mean(val_acc)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs > args.early_stopping_epochs:
                cprint("Early stopping.", "cyan")
                break

    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------    
    print("Now Loading Test Datasets")
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    folds_preds = []
    for fold in range(args.num_splits):
        model = BasicConvClassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)
        model.load_state_dict(torch.load(os.path.join(logdir, f"model_best_{fold}.pt"), map_location=args.device))
        preds = [] 
        model.eval()
        for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
            preds.append(model(X.to(args.device)).detach().cpu())
            preds = torch.cat(preds, dim=0).numpy()
        
        folds_preds.append(preds)
    
    final_preds = np.mean(folds_preds, axis=0)
    np.save(os.path.join(logdir, "submission"), final_preds)
    cprint(f"Submission {final_preds.shape} saved at {logdir}", "cyan")

    # Google Driveにファイルをコピー
    drive_dir = os.path.join(args.data_dir, f"{args.expname}_{args.ver}", )
    os.makedirs(drive_dir, exist_ok=True)

    if os.path.exists(logdir):
        shutil.copy(logdir, drive_dir)
        print(f'Model saved to Google Drive: {drive_dir}')

    if os.path.exists(logdir):
        shutil.copy(logdir, drive_dir)
        print(f'Submission file saved to Google Drive: {drive_dir}')



if __name__ == "__main__":
    run()
