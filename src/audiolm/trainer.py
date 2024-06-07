"""Definition for trainer classes"""

import os
from math import ceil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import torch
from torch import nn

from audiolm.encodec import Encodec
from audiolm.constants import DEVICE
from audiolm.data_preparation import AudioDataLoader
from audiolm.w2v_hubert import W2VHuBert
from audiolm.utils import save_checkpoint, save_model
from audiolm.absolute_transformer import (
    SemanticTransformer,
    CoarseAcousticTransformer,
    FineAcousticTransformer,
)


class Trainer(ABC):
    """
    Trainer class for training a Transformer model.
    """

    @abstractmethod
    # pylint: disable =too-many-arguments
    def __init__(
        self,
        semantic_encoder: Optional[W2VHuBert] = None,
        semantic_transformer: Optional[SemanticTransformer] = None,
        acoustic_encoder_decoder: Optional[Encodec] = None,
        coarse_acoustic_transformer: Optional[CoarseAcousticTransformer] = None,
        fine_acoustic_transformer: Optional[FineAcousticTransformer] = None,
        train_dataloader: Optional[AudioDataLoader] = None,
        val_dataloader: Optional[AudioDataLoader] = None,
        test_dataloader: Optional[AudioDataLoader] = None,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        intervals: Optional[int] = None,
        save_path: Optional[os.PathLike] = None,
        early_stop_counter: Optional[int] = None,
        early_stopping_range: Optional[int] = None,
        epochs: Optional[int] = None,
    ):
        self.semantic_encoder = semantic_encoder
        self.semantic_transformer = semantic_transformer
        self.acoustic_encoder_decoder = acoustic_encoder_decoder
        self.coarse_acoustic_transformer = coarse_acoustic_transformer
        self.fine_acoustic_transformer = fine_acoustic_transformer
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
    def test(self):
        """Test the model on the test dataset."""

    # endregion

    # region: Private methods.
    def _train_step(self, model: nn.Module) -> float:
        model.train()
        train_loss = 0

        for batch in tqdm(
            self.train_dataloader,
            total=ceil(len(self.train_dataloader) / self.train_dataloader.batch_size),
        ):
            batch = batch.to(DEVICE)
            loss = self.loss_generator(batch)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch.detach().cpu() # Free up memory
        train_loss /= len(self.train_dataloader)
        return train_loss

    def _validation_step(self, model: nn.Module) -> float:
        model.eval()
        validation_loss = 0
        with torch.inference_mode():
            for batch in self.val_dataloader:
                batch = batch.to(DEVICE)
                loss = self.loss_generator(batch)
                validation_loss += loss.item()
                batch.detach().cpu() # Free up memory
        validation_loss /= len(self.val_dataloader)

        return validation_loss

    def _train(self, model: nn.Module):
        writer = SummaryWriter(
            Path(self.save_path) / "runs" / str(type(model).__name__)
        )
        for epoch in tqdm(range(self.epochs), total=self.epochs, desc="Training"):
            train_loss = self._train_step(model)
            validation_loss = self._validation_step(model)
            print("SAVING CHECKPOINT...")
            save_checkpoint(
                model, epoch, self.optimizer, self.early_stop_counter, self.save_path
            )
            print("SAVING RUN FOR TENSORBOARD...")
            writer.add_scalars(
                main_tag=f"Loss_{str(type(model).__name__)}",
                tag_scalar_dict={
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                },
                global_step=epoch,
            )

            if validation_loss < self.best_val_loss:
                self.best_val_loss = validation_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.early_stopping_range:
                print(f"Early stopping training at epoch: {epoch+1}")
                break
        
        model.detach().cpu() # Free up memory
        writer.flush()
        writer.close()
        save_model(model, self.save_path)

    def _test(self, model):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing"):
                batch = batch.to(DEVICE)
                loss = self.loss_generator(batch)
                test_loss += loss.item()
                batch.detach().cpu() # Free up memory
        test_loss /= len(self.test_dataloader)
        print(f"Test Loss: {test_loss: .4f}")

        return test_loss

    # endregion


class SemanticTrainer(Trainer):
    """Trainer class derived from `Trainer`."""

    def __init__(
        self,
        semantic_encoder: W2VHuBert,
        semantic_transformer: SemanticTransformer,
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
            `semantic_encoder` (W2VHuBert)

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
        self.semantic_encoder = semantic_encoder
        self.semantic_transformer = semantic_transformer

    def loss_generator(self, batch):
        semantic_encode = self.semantic_encoder(batch)

        output, target = self.semantic_transformer.fit(semantic_encode)

        loss = self.loss(output, target)
        return loss

    def train(self):
        self.semantic_encoder.to(DEVICE)
        self.semantic_transformer.to(DEVICE)
        
        self._train(self.semantic_transformer)
        if DEVICE=="cuda":
            self.semantic_encoder.to('cpu')
            self.semantic_transformer.to('cpu')


    def test(self):
        return self._test(self.semantic_transformer)


class CoarseAcousticTrainer(Trainer):
    """Trainer class derived from `Trainer`."""

    def __init__(
        self,
        semantic_encoder: W2VHuBert,
        semantic_transformer: SemanticTransformer,
        acoustic_encoder_decoder: Encodec,
        coarse_acoustic_transformer: CoarseAcousticTransformer,
        train_dataloader: AudioDataLoader,
        val_dataloader: AudioDataLoader,
        test_dataloader: AudioDataLoader,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        intervals: int,
        save_path: Path,
        early_stop_counter: int,
        early_stopping_range: int,
        generate_audio_len: int,
        epochs: int,
    ):
        """
        Takes as input `semantic_encoder` and `semantic_transformer`.
        They determine the `semantic_modelling`.

        `semantic_encoder` must be trained ahead of time, this trainer only
        trains `semantic_transformer`.

        Args
        ----
            `semantic_encoder` (W2VHuBert)

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
            acoustic_encoder_decoder=acoustic_encoder_decoder,
            coarse_acoustic_transformer=coarse_acoustic_transformer,
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
        self.semantic_encoder = semantic_encoder
        self.semantic_transformer = semantic_transformer
        self.acoustic_encoder_decoder = acoustic_encoder_decoder
        self.coarse_acoustic_transformer = coarse_acoustic_transformer
        self.generate_audio_len = generate_audio_len

    def loss_generator(self, batch):

        semantic_encode = self.semantic_encoder(batch)
        semantic_token = self.semantic_transformer.generate(
            semantic_encode, self.generate_audio_len * 50
        )

        coarse_acoustic_tokens, _, _ = self.acoustic_encoder_decoder.encode(batch)
        conditioning = torch.cat((semantic_token, coarse_acoustic_tokens), dim=1)

        output, target = self.coarse_acoustic_transformer.fit(conditioning)
        loss = self.loss(output, target)
        return loss

    def train(self):
        self.semantic_encoder.to(DEVICE)
        self.semantic_transformer.to(DEVICE)
        self.acoustic_encoder_decoder.to(DEVICE)
        self.coarse_acoustic_transformer.to(DEVICE)

        self._train(self.coarse_acoustic_transformer)
        
        if DEVICE=="cuda":
            self.semantic_encoder.to('cpu')
            self.semantic_transformer.to('cpu')
            self.acoustic_encoder_decoder.to('cpu')
            self.coarse_acoustic_transformer.to('cpu')

    def test(self):
        return self._test(self.coarse_acoustic_transformer)
