"""Utilities functions"""

import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn


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
    checkpoint_path = Path(save_path) / "models"

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


def load_model(model: nn.Module, model_path:os.PathLike):
    if model_path.exists():
        print("Model found")
        return torch.load(Path(model_path))
    
    else: 
        print("Model not found")
        return None



def load_checkpoint(
    model, epoch, save_path, latest=True
) -> Tuple[nn.Module, int, torch.optim.Optimizer, int]:
    """
    Loads a checkpoint for a given model.

    Args:
        model (nn.Module): The model to load the checkpoint for.
        epoch (int): The epoch number of the checkpoint.
        save_path (str): The path where the checkpoint is saved.

    Returns:
        tuple: A tuple containing the loaded model, epoch number, optimizer, and early stop counter.
    """

    if latest:
        epoch = get_latest_epoch(save_path) 

    model_name = str(type(model).__name__)
    checkpoint = Path(save_path) / "models" / f"{model_name}_epoch_{epoch+1}.pth"

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    early_stop_counter = checkpoint["early_stop_counter"]
    # print(
    #     f"Checkpoint loaded: {checkpoint}, starting from epoch: {checkpoint['epoch']+1}"
    # )
    print(f"Checkpoint loaded: {checkpoint}, starting from epoch: {epoch+1}")
    return model, epoch, optimizer, early_stop_counter


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
    model_path = Path(save_path) / "models" / f"{str(type(model).__name__)}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")
