"""Train module"""

import os
from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from tqdm.auto import tqdm

from audiolm.custom_encodec import CustomEncodecModel
from audiolm.costants import DEVICE, DEBUG
from audiolm.data_preparation import AudioDataLoader
from audiolm.w2v_hubert import W2VHuBERT_Quantizier
from audiolm.absolute_transformer import TransformerDecoderOnly

# TODO Fix docstrings.


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
        loss: Optional[nn.Module] = None,
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

    # region: Abstract methods, this methods must be redefined accordingly.
    @abstractmethod
    def loss_generator(self, batch):
        """Generate loss"""

    @abstractmethod
    def train(self):
        """
        Train the Transformer model.
        """

    @abstractmethod
    def validate(self):
        """Test the model on the validation dataset."""

    @abstractmethod
    def test(self):
        """Test the model on the test dataset."""

    @abstractmethod
    def save_checkpoint(self, epoch):
        """
        Save a checkpoint of the model and optimizer.

        Args:
            epoch (int): The current epoch.
        """

    @abstractmethod
    def save_model(self):
        """Save the current model."""

    @abstractmethod
    def load_checkpoint(self, epoch):
        """
        Load a checkpoint of the model and optimizer.

        Args:
            checkpoint_path (str): The path to the checkpoint file.

        Returns:
            int: The epoch from which training will resume.
        """

    # endregion

    # region: Private methods.
    def _train(self, model: nn.Module):
        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in tqdm(
                self.train_dataloader,
                desc=f"Epoch: {epoch+1}/{self.epochs}",
                total=ceil(
                    len(self.train_dataloader) / self.train_dataloader.batch_size
                ),
            ):
                batch = batch.to(DEVICE)
                print(f"Batch moved to device: {DEVICE}" if DEBUG else "")

                # output, target = self.loss_generator(batch)
                # print(
                #     f"Generate output: {output.shape} and target:{target.shape}"
                #     if DEBUG
                #     else ""
                # )
                # loss = self.loss(output, target)
                loss = self.loss_generator(batch)
                print(f"Loss: {loss.item()}" if DEBUG else "")

                epoch_loss += loss.item()

                self.optimizer.zero_grad()

                loss.backward()
                print("Loss backward" if DEBUG else "")

                self.optimizer.step()
                print("Optimizer step" if DEBUG else "")

            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(
                f"Epoch: {epoch+1}/{self.epochs} | Loss: {avg_epoch_loss: .4f}"
                if DEBUG
                else ""
            )

            self.save_checkpoint(epoch)
            print(f"Saving loss: {loss}" if DEBUG else "")
            validation_loss = self._validate(model)

            if validation_loss < self.best_val_loss:
                self.best_val_loss = validation_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.early_stopping_range:
                print(f"Early Stopping at epoch: {epoch+1}")
                break

    def _validate(self, model):

        model().eval()
        validation_loss = 0
        with torch.no_grad:
            for batch in tqdm(
                self.val_dataloader,
                total=ceil(len(self.val_dataloader) / self.val_dataloader.batch_size),
            ):
                batch = batch.to(DEVICE)
                print("Moved batch to device." if DEBUG else "")

                output, target = self.loss_generator(batch)
                print(
                    f"Generate output: {output.shape} and target:{target.shape}"
                    if DEBUG
                    else ""
                )

                loss = self.loss(output, target)
                print(f"Loss: {loss.item()}" if DEBUG else "")

                validation_loss += loss.item()

        avg_val_loss = validation_loss / len(self.val_dataloader)
        print(f"Validation Loss: {avg_val_loss: .4f}" if DEBUG else "")

        return avg_val_loss

    def _test(self, model):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in enumerate(
                tqdm(
                    self.test_dataloader,
                    total=ceil(
                        len(self.test_dataloader) / self.test_dataloader.batch_size
                    ),
                )
            ):
                batch = batch.to(DEVICE)
                output, target = self.loss_generator(batch)

                loss = self.loss(output, target)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(self.test_dataloader)
        print(f"Test Loss: {avg_test_loss: .4f}")

        return avg_test_loss

    def _save_checkpoint(self, epoch, model):

        model_name = type(model).__name__
        print(f"Model Name {model_name}" if DEBUG else "")

        checkpoint_path = os.path.join(
            self.save_path, f"{model_name}_epoch_{epoch+1}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "early_stop_counter": self.early_stop_counter,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, model, epoch):

        model_name = type(model).__name__
        print(f"Model Name {model_name}" if DEBUG else "")
        checkpoint_path = os.path.join(
            self.save_path, f"{model_name}_epoch_{epoch+1}.pth"
        )
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.early_stop_counter = checkpoint["early_stop_counter"]
        print(
            f"Checkpoint loaded: {checkpoint_path}, starting from epoch: {checkpoint['epoch']+1}"
        )
        return model, checkpoint["epoch"] + 1

    def _save_model(self, model):
        model_path = self.save_path / type(model).__name__
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

    # endregion


# pylint: disable =too-many-arguments
class SemanticTrainer(Trainer):
    """Trainer class derived from `Trainer`."""

    def __init__(
        self,
        semantic_encoder: W2VHuBERT_Quantizier,
        semantic_transformer: TransformerDecoderOnly,
        train_dataloader: AudioDataLoader,
        val_dataloader: AudioDataLoader,
        test_dataloader: AudioDataLoader,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        intervals: int,
        save_path: Path,
        early_stop_counter: int,
        early_stopping_range: int,
        epochs: int,
    ):
        """
        Takes as input `semantic_encoder` and `semantic_transformer`.
        They determine the `semantic_modelling`.

        `semantic_encoder` must be trained ahead of time, this trainer only
        trains `semantic_transformer`.

        Args
        ----
            `semantic_encoder` (W2VHuBERT_Quantizier)

            `semantic_transformer` (TransformerDecoderOnly)

            `train_dataloader` (AudioDataLoader)

            `val_dataloader` (AudioDataLoader)

            `test_dataloader` (AudioDataLoader)

            `loss` (nn.Module)

            `optimizer` (torch.optim.Optimizer)

            `intervals` (int)

            `save_path` (Path)

            `early_stop_counter` (int)

            `early_stopping_range` (int)

            `epochs` (int)
        """
        super().__init__(
            semantic_encoder=semantic_encoder,
            semantic_transformer=semantic_transformer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            loss=loss,
            optimizer=optimizer,
            intervals=intervals,
            save_path=save_path,
            early_stop_counter=early_stop_counter,
            early_stopping_range=early_stopping_range,
            epochs=epochs,
        )

    def loss_generator(self, batch):
        semantic_encode = self.semantic_encoder(batch)
        print(f"semantic_encode shape: {semantic_encode.shape}" if DEBUG else "")
        output, target = self.semantic_transformer.fit(semantic_encode)
        print(
            f"Generate output: {output.shape} and target:{target.shape}"
            if DEBUG
            else ""
        )
        loss = self.loss(output, target)
        return loss

    def train(self):
        return self._train(self.semantic_transformer)

    def validate(self):
        return self._validate(self.semantic_transformer)

    def test(self):
        return self._test(self.semantic_transformer)

    def save_checkpoint(self, epoch):
        return self._save_checkpoint(self.semantic_transformer, epoch)

    def load_checkpoint(self, epoch):
        return self._load_checkpoint(self.semantic_transformer, epoch)

    def save_model(self):
        return self._save_model(self.semantic_transformer)


# pylint: disable =too-many-arguments
class AcousticTrainer(Trainer):
    """Trainer class derived from `Trainer`."""

    def __init__(
        self,
        semantic_encoder: W2VHuBERT_Quantizier,
        semantic_transformer: TransformerDecoderOnly,
        acoustic_enc_dec: CustomEncodecModel,
        acoustic_transformer: TransformerDecoderOnly,
        train_dataloader: AudioDataLoader,
        val_dataloader: AudioDataLoader,
        test_dataloader: AudioDataLoader,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        intervals: int,
        save_path: Path,
        early_stop_counter: int,
        early_stopping_range: int,
        epochs: int,
    ):
        """
        Takes as input `semantic_encoder` and `semantic_transformer`.
        They determine the `semantic_modelling`.

        `semantic_encoder` must be trained ahead of time, this trainer only
        trains `semantic_transformer`.

        Args
        ----
            `semantic_encoder` (W2VHuBERT_Quantizier)

            `semantic_transformer` (TransformerDecoderOnly)

            `train_dataloader` (AudioDataLoader)

            `val_dataloader` (AudioDataLoader)

            `test_dataloader` (AudioDataLoader)

            `loss` (nn.Module)

            `optimizer` (torch.optim.Optimizer)

            `intervals` (int)

            `save_path` (Path)

            `early_stop_counter` (int)

            `early_stopping_range` (int)

            `epochs` (int)
        """
        super().__init__(
            semantic_encoder=semantic_encoder,
            semantic_transformer=semantic_transformer,
            acoustic_enc_dec=acoustic_enc_dec,
            acoustic_transformer=acoustic_transformer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            loss=loss,
            optimizer=optimizer,
            intervals=intervals,
            save_path=save_path,
            early_stop_counter=early_stop_counter,
            early_stopping_range=early_stopping_range,
            epochs=epochs,
        )

    def loss_generator(self, batch):
        # TODO: Finish, but wait for valerio.
        semantic_encode = self.semantic_encoder(batch)
        semantic_token = self.semantic_transformer.generate(semantic_encode, 3)

        coarse_acoustic_tokens, _, _ = self.acoustic_enc_dec.encode(batch)
        print(f"shape coarse_acoustic: {coarse_acoustic_tokens.shape}")
        print(f"shape semantic_token: {semantic_token.shape}")

        conditioning = torch.cat((semantic_token, coarse_acoustic_tokens), dim=1)

        output, target = self.acoustic_transformer.fit(conditioning)
        print(
            f"Generate output: {output.shape} and target:{target.shape}"
            if DEBUG
            else ""
        )
        loss = self.loss(output, target)
        return loss

    def train(self):
        return self._train(self.acoustic_transformer)

    def validate(self):
        return self._validate(self.acoustic_transformer)

    def test(self):
        return self._test(self.acoustic_transformer)

    def save_checkpoint(self, epoch):
        return self._save_checkpoint(self.acoustic_transformer, epoch)

    def load_checkpoint(self, epoch):
        return self._load_checkpoint(self.acoustic_transformer, epoch)

    def save_model(self):
        return self._save_model(self.acoustic_transformer)
