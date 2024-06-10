"""Utilities functions"""

import os
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch import nn
import numpy as np
from scipy.linalg import sqrtm



def save_checkpoint(
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    early_stop_counter: int,
    save_path: os.PathLike,
):
    """
    Save the checkpoint of the model during training.

    Args:
        model (nn.Module): The model to be saved.
        epoch (int): The current epoch number.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        best_val_loss (int): The best validation loss achieved so far.
        early_stop_counter (int): The counter for early stopping.
        save_path (os.PathLike): The path to save the checkpoint.

    Returns:
        None
    """
    model_name = str(type(model).__name__)
    checkpoint_path = Path(save_path) / "models" / str(type(model).__name__)

    if not checkpoint_path.exists():
        os.makedirs(checkpoint_path)
    checkpoint = checkpoint_path / f"{model_name}_epoch_{epoch+1}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "early_stop_counter": early_stop_counter,
        },
        checkpoint,
    )


def load_model(model: nn.Module, model_path: os.PathLike):
    """Loads into `model` the dictionary found at `model_path`"""
    model_path = Path(model_path)
    model.load_state_dict(torch.load(Path(model_path)))


def load_checkpoint(
    model, checkpoint_path
) -> Tuple[nn.Module, int, torch.optim.Optimizer, int]:
    """
    Loads a checkpoint for a given model.

    Args:
        model (nn.Module): The model to load the checkpoint for.
        save_path (str): The path where the checkpoint is saved.

    Returns:
        tuple: A tuple containing the loaded model, epoch number, optimizer, and early stop counter.
    """

    epoch = int(checkpoint_path.split("_epoch_")[1].split(".pth")[0])
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    early_stop_counter = checkpoint["early_stop_counter"]
    print(f"Checkpoint loaded: {checkpoint}, starting from epoch: {epoch+1}")
    return model, epoch, optimizer, early_stop_counter


def get_model_path(model_root: os.PathLike) -> Optional[os.PathLike]:
    """Returns the model path if exists."""
    model_root = Path(model_root)
    for file in model_root.glob("*.pth"):
        if "epoch" not in file.stem:
            return file
    return None


def get_latest_checkpoint_path(model_root: os.PathLike) -> Optional[os.PathLike]:
    """Returns the latest checkpoint path if exists."""
    model_root = Path(model_root)
    filtered = list(filter(lambda file: "epoch" in file.stem, model_root.glob("*.pth")))
    if not filtered:
        return None

    return sorted(
        filtered,
        key=lambda file: int(file.stem.split("_epoch_")[1].split(".pth")[0]),
    )[-1]


def get_latest_epoch(save_path: os.PathLike) -> int:
    """
    Get the latest epoch number from the checkpoint directory.

    Args:
        save_path (os.PathLike): The path to the checkpoint directory.

    Returns:
        int: The latest epoch number.
    """
    checkpoint_path = Path(save_path) / "models"
    checkpoints_epoch = []
    for c in checkpoint_path.glob("*.pth"):
        try:
            checkpoints_epoch.append(int(c.stem.split("_")[-1]))
        except ValueError:
            print(f"Skipping non-epoch file: {c.name}")

    if checkpoints_epoch:
        latest_epoch = max(checkpoints_epoch)
        print("Latest epoch found: ", latest_epoch)
        return latest_epoch
    else:
        print("No checkpoints found")
        return 0


def save_model(model: nn.Module, save_path: os.PathLike):
    """Saves the model state dict.

    Args:
        model (nn.Module): The model to be saved.
        save_path (os.PathLike): The path to save the model.

    """
    model_name = str(type(model).__name__)
    model_path = Path(save_path) / "models" / model_name / f"{model_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")



def calculate_mean_and_cov(embeddings):
    if embeddings.dim() > 2:
        embeddings = embeddings.view(-1, embeddings.size(-1))
    mu = torch.mean(embeddings, axis=0)
    sigma = torch.cov(embeddings.t())
    return mu, sigma



def frechet_distance(mu1, sigma1, mu2, sigma2):
    mu1 = mu1.to('cpu')
    sigma1 = sigma1.to('cpu')
    mu2 = mu2.to('cpu')
    sigma2 = sigma2.to('cpu')
    
    diff = mu1 - mu2
    covmean = sqrtm((sigma1.numpy() @ sigma2.numpy()))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    covmean = torch.from_numpy(covmean)
    
    term1 = torch.dot(diff, diff)
    term2 = torch.trace(sigma1 + sigma2 - 2 * covmean)
    return term1 + term2
