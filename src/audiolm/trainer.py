"""Train module"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from audiolm.custom_encodec import CustomEncodecModel
from audiolm.constants import DEVICE, DEBUG
from audiolm.data_preparation import AudioDataLoader
from audiolm.w2v_hubert import W2VHuBERT_Quantizier
from audiolm.absolute_transformer import (
    SemanticTransformer,
    CoarseAcousticTransformer,
    FineAcousticTransformer,
)
from audiolm.utils import save_checkpoint, save_model


class Trainer(ABC):
    """
    Trainer class for training a Transformer model.
    """

    @abstractmethod
    # pylint: disable =too-many-arguments
    def __init__(
        self,
        semantic_encoder: Optional[W2VHuBERT_Quantizier] = None,
        semantic_transformer: Optional[SemanticTransformer] = None,
        acoustic_encoder_decoder: Optional[CustomEncodecModel] = None,
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
        for batch in self.train_dataloader:
            batch = batch.to(DEVICE)
            loss = self.loss_generator(batch)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
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

        validation_loss /= len(self.val_dataloader)

        return validation_loss

    def _train(self, model: nn.Module):
        writer = SummaryWriter(Path(self.save_path) / "runs")
        for epoch in tqdm(range(self.epochs), desc="Training"):
            train_loss = self._train_step(model)
            validation_loss = self._validation_step(model)
            save_checkpoint(
                model, epoch, self.optimizer, self.early_stop_counter, self.save_path
            )
            writer.add_scalars(
                main_tag="Loss",
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
        writer.flush()
        writer.close()
        save_model(model, self.save_path)

    def _test(self, model):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing"):
                batch = batch.to(DEVICE)
                output, target = self.loss_generator(batch)

                loss = self.loss(output, target)
                test_loss += loss.item()

        test_loss /= len(self.test_dataloader)
        print(f"Test Loss: {test_loss: .4f}")

        return test_loss

    # endregion


# pylint: disable =too-many-arguments
class SemanticTrainer(Trainer):
    """Trainer class derived from `Trainer`."""

    def __init__(
        self,
        semantic_encoder: W2VHuBERT_Quantizier,
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

    def test(self):
        return self._test(self.semantic_transformer)


# pylint: disable =too-many-arguments
class CoarseAcousticTrainer(Trainer):
    """Trainer class derived from `Trainer`."""

    def __init__(
        self,
        semantic_encoder: W2VHuBERT_Quantizier,
        semantic_transformer: SemanticTransformer,
        acoustic_encoder_decoder: CustomEncodecModel,
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

    def loss_generator(self, batch):
        semantic_encode = self.semantic_encoder(batch)
        semantic_token = self.semantic_transformer.generate(semantic_encode, 3)

        coarse_acoustic_tokens, _, _ = self.acoustic_encoder_decoder.encode(batch)
        print(f"shape coarse_acoustic: {coarse_acoustic_tokens.shape}")
        print(f"shape semantic_token: {semantic_token.shape}")

        conditioning = torch.cat((semantic_token, coarse_acoustic_tokens), dim=1)

        output, target = self.coarse_acoustic_transformer.fit(conditioning)
        print(
            f"Generate output: {output.shape} and target:{target.shape}"
            if DEBUG
            else ""
        )
        loss = self.loss(output, target)
        return loss

    def train(self):
        return self._train(self.coarse_acoustic_transformer)

    def test(self):
        return self._test(self.coarse_acoustic_transformer)


class FineAcousticTrainer(Trainer):
    """Trainer class derived from `Trainer`."""

    def __init__(
        self,
        semantic_encoder: W2VHuBERT_Quantizier,
        semantic_transformer: SemanticTransformer,
        acoustic_encoder_decoder: CustomEncodecModel,
        coarse_acoustic_transformer: CoarseAcousticTransformer,
        fine_acoustic_transformer: FineAcousticTransformer,
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
            acoustic_encoder_decoder=acoustic_encoder_decoder,
            coarse_acoustic_transformer=coarse_acoustic_transformer,
            fine_acoustic_transformer=fine_acoustic_transformer,
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
        semantic_token = self.semantic_transformer.generate(semantic_encode, 3)

        coarse_acoustic_tokens, fine_acoustic_tokens, _ = (
            self.acoustic_encoder_decoder.encode(batch)
        )
        coarse_conditioning = torch.cat((semantic_token, coarse_acoustic_tokens), dim=1)
        coarse_tokens = self.coarse_acoustic_transformer.generate(
            coarse_conditioning, 3
        )

        output, target = self.fine_acoustic_transformer(
            torch.cat((coarse_tokens, fine_acoustic_tokens), dim=1)
        )
        loss = self.loss(output, target)
        return loss

    def train(self):
        return self._train(self.fine_acoustic_transformer)

    def test(self):
        return self._test(self.fine_acoustic_transformer)
