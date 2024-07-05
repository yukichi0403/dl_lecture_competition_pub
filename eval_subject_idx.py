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
    subject_idxs_test = torch.load(os.path.join(args.data_dir, "test_subject_idxs.pt"))
    unique_subjects = torch.unique(subject_idxs_test)
    preds = np.zeros((len(test_set), args.num_classes))

    for subject in unique_subjects:
        test_indices = (subject_idxs_test == subject).nonzero(as_tuple=True)[0]
        print(f"Subject {subject}. Test size: {len(test_indices)}")
        
        test_subset = Subset(test_set, test_indices)
        test_loader = DataLoader(
            test_subset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
        )

        model = CustomModel(
            args.backbone,
            args.num_classes
        ).to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, f"model_best_{subject}.pt"), map_location=args.device))
        
        model.eval()
        subject_preds = [] 
        for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
            subject_preds.append(model(X.to(args.device)).detach().cpu())
        
        subject_preds = torch.cat(subject_preds, dim=0).numpy()
        preds[test_indices, :] = subject_preds

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