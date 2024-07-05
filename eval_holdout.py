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

import shutil
import gc
from torch.utils.data import DataLoader, ConcatDataset, Subset


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config_colab")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    
    # ------------------
    #    Dataloader
    # ------------------    
    test_set = ThingsMEGDataset("test", args.data_dir)
    preds = np.zeros((len(test_set), args.num_classes))

    test_loader = DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = CustomModel(
        args.backbone,
        args.num_classes,
        True,
        args.aux_loss_ratio
    ).to(args.device)
    model.load_state_dict(torch.load(os.path.join(savedir, f"model_best.pt"), map_location=args.device))
    
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

    np.save(os.path.join(savedir, "submission.npy"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")

    # Google Driveにファイルをコピー
    drive_dir = os.path.join(args.data_dir, f"{args.expname}_{args.ver}")
    os.makedirs(drive_dir, exist_ok=True)

    submission_path = os.path.join(savedir, "submission.npy")
    if os.path.exists(submission_path):
        shutil.copy(submission_path, drive_dir)
        print(f'Submission file saved to Google Drive: {drive_dir}')

if __name__ == "__main__":
    run()