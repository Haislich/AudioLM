"""Train module"""

import os
from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from tqdm.auto import tqdm

from audiolm.acoustic_modelling.custom_encodec import CustomEncodecModel
from audiolm.costants import DEVICE
from audiolm.data_preparation import AudioDataLoader
from audiolm.semantic_acoustic_modeling.W2VHuBERT_Quantizier import W2VHuBERT_Quantizier
from audiolm.transformer.AbsoluteTransformer import TransformerDecoderOnly


class Trainer(ABC):
    """
    Trainer class for training a Transformer model.


    """

    @abstractmethod
    # pylint: disable =too-many-arguments
    def __init__(
        self,
        semantic_encoder: Optional[W2VHuBERT_Quantizier] = None,
        semantic_transformer: Optional[TransformerDecoderOnly] = None,
        acoustic_enc_dec: Optional[CustomEncodecModel] = None,
        acoustic_transformer: Optional[TransformerDecoderOnly] = None,
        train_dataloader: Optional[AudioDataLoader] = None,
        val_dataloader: Optional[AudioDataLoader] = None,
        test_dataloader: Optional[AudioDataLoader] = None,
        loss: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        intervals: Optional[int] = None,
        save_path: Optional[Path] = None,
        early_stop_counter: Optional[int] = None,
        early_stopping_range: Optional[int] = None,
        epochs: Optional[int] = None,
    ):
        self.semantic_encoder = semantic_encoder
        self.semantic_transformer = semantic_transformer
        self.acoustic_enc_dec = acoustic_enc_dec
        self.acoustic_transformer = acoustic_transformer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.intervals = intervals
        self.epochs = epochs
        self.save_path = save_path
        self.best_val_loss = float("inf")
        self.early_stopping_range = early_stopping_range
        self.early_stop_counter = early_stop_counter
        self.loss = loss
        if save_path is not None and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    @abstractmethod
    def _loss_generator(self, batch):
        pass

    def _train(self):
        """
        Train the Transformer model.
        """
        for epoch in self.epochs:
            epoch_loss = 0
            for batch in tqdm(
                self.train_dataloader,
                desc=f"Epoch: {epoch+1}/{self.epochs}",
                total=ceil(
                    len(self.train_dataloader) / self.train_dataloader.batch_size
                ),
            ):
                batch = batch.to(DEVICE)
                output, target = self._loss_generator(batch)
                loss = self.loss(target, output)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()

                self.loss.backward()

                self.optimizer.step()
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch: {epoch+1}/{self.epochs} | Loss: {avg_epoch_loss: .4f}")

            # self.save_checkpoint(epoch)

    # def evaluate(self):
    #     """
    #     Evaluate the Transformer model on the validation data.

    #     Returns:
    #         float: The average validation loss.
    #     """
    #     self.model().eval()
    #     validation_loss = 0
    #     with torch.no_grad:
    #         for batch in enumerate(
    #             tqdm(
    #                 self.val_dataloader,
    #                 total=ceil(
    #                     len(self.val_dataloader) / self.val_dataloader.batch_size
    #                 ),
    #             )
    #         ):
    #             batch = batch.to(self.device)
    #             semantic_token_batch = self.quantizier.forward(batch)

    #             # teacher forcing, shift the input by one position in order to predict the next token
    #             input = semantic_token_batch[:, :-1].to(self.device)
    #             target = semantic_token_batch[:, 1:].to(self.device)

    #             # generate causal mask to prevent the model from attending to future tokens
    #             causal_mask = self.model.generate_causal_mask(seq_len=input.size(1)).to(
    #                 self.device
    #             )

    #             # forward pass
    #             output = self.model(input, tgt_mask=causal_mask)
    #             output = output.reshape(-1, output.size(-1))
    #             target = target.reshape(-1)

    #             loss = self.loss_function(output, target)
    #             validation_loss += loss.item()

    #     avg_val_loss = validation_loss / len(self.val_dataloader)
    #     print(f"Validation Loss: {avg_val_loss: .4f}")

    #     return avg_val_loss

    # def test(self):
    #     """
    #     Test the Transformer model on the test data.

    #     Args:
    #         test_dataloader (AudioDataLoader): The dataloader for the test data.
    #     """
    #     self.model.eval()
    #     test_loss = 0
    #     with torch.no_grad():
    #         for batch in enumerate(
    #             tqdm(
    #                 self.test_dataloader,
    #                 total=ceil(
    #                     len(self.test_dataloader) / self.test_dataloader.batch_size
    #                 ),
    #             )
    #         ):
    #             batch = batch.to(self.device)
    #             semantic_token_batch = self.quantizier.forward(batch)

    #             # teacher forcing, shift the input by one position in order to predict the next token
    #             input = semantic_token_batch[:, :-1].to(self.device)
    #             target = semantic_token_batch[:, 1:].to(self.device)

    #             # generate causal mask to prevent the model from attending to future tokens
    #             causal_mask = self.model.generate_causal_mask(seq_len=input.size(1)).to(
    #                 self.device
    #             )

    #             output = self.model(input, tgt_mask=causal_mask)
    #             output = output.reshape(-1, output.size(-1))
    #             target = target.reshape(-1)

    #             loss = self.loss_function(output, target)
    #             test_loss += loss.item()

    #     avg_test_loss = test_loss / len(self.test_dataloader)
    #     print(f"Test Loss: {avg_test_loss: .4f}")

    #     return avg_test_loss

    # def save_checkpoint(self, epoch):
    #     """
    #     Save a checkpoint of the model and optimizer.

    #     Args:
    #         epoch (int): The current epoch.
    #     """
    #     checkpoint_path = os.path.join(self.save_path, f"model_epoch_{epoch+1}.pth")
    #     torch.save(
    #         {
    #             "epoch": epoch,
    #             "model_state_dict": self.model.state_dict(),
    #             "optimizer_state_dict": self.optimizer.state_dict(),
    #             "scheduler_state_dict": (
    #                 self.scheduler.state_dict() if self.scheduler else None
    #             ),
    #             "best_val_loss": self.best_val_loss,
    #             "early_stop_counter": self.early_stop_counter,
    #         },
    #         checkpoint_path,
    #     )
    #     print(f"Checkpoint saved: {checkpoint_path}")

    # def load_checkpoint(self, checkpoint_path):
    #     """
    #     Load a checkpoint of the model and optimizer.

    #     Args:
    #         checkpoint_path (str): The path to the checkpoint file.

    #     Returns:
    #         int: The epoch from which training will resume.
    #     """
    #     checkpoint = torch.load(checkpoint_path)
    #     self.model.load_state_dict(checkpoint["model_state_dict"])
    #     self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     if self.scheduler and checkpoint["scheduler_state_dict"]:
    #         self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    #     self.best_val_loss = checkpoint["best_val_loss"]
    #     self.early_stop_counter = checkpoint["early_stop_counter"]
    #     print(
    #         f"Checkpoint loaded: {checkpoint_path}, starting from epoch: {checkpoint['epoch']+1}"
    #     )
    #     return checkpoint["epoch"] + 1

    # def save_model(self, path):
    #     torch.save(self.model.state_dict(), path)
    #     print(f"Model saved: {path}")


class SemanticTrainer(Trainer):
    def __init__(
        self,
        semantic_encoder: W2VHuBERT_Quantizier,
        semantic_transformer: TransformerDecoderOnly,
        train_dataloader: AudioDataLoader,
        val_dataloader: AudioDataLoader,
        test_dataloader: AudioDataLoader,
        loss: torch.Module,
        optimizer: torch.optim.Optimizer,
        intervals: int,
        save_path: Path,
        early_stop_counter: int,
        early_stopping_range: int,
        epochs: int,
    ):
        super().__init__(
            semantic_encoder,
            semantic_transformer,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            loss,
            optimizer,
            intervals,
            save_path,
            early_stop_counter,
            early_stopping_range,
            epochs,
        )

    @Trainer._train
    def train(self):
        self._train()
        return

    def _loss_generator(self, batch):
        semantic_encode = self.semantic_encoder(batch)
        return self.semantic_transformer.fit(semantic_encode)
