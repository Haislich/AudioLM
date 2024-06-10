import os
from pathlib import Path

from math import floor

import torch
from torch import nn

from audiolm.absolute_transformer import (
    SemanticTransformer,
    CoarseAcousticTransformer,
    FineAcousticTransformer,
)
from audiolm.encodec import Encodec
from audiolm.w2v_hubert import W2VHuBert
from audiolm.trainer import SemanticTrainer, CoarseAcousticTrainer
from audiolm.data_preparation import AudioDataLoader


class AudioLM:
    def __init__(
        self,
        semantic_encoder: W2VHuBert,
        semantic_transformer: SemanticTransformer,
        acoustic_encoder_decoder: Encodec,
        coarse_acoustic_transformer: CoarseAcousticTransformer,
        fine_acoustic_transformer: FineAcousticTransformer,
        # https://stackoverflow.com/a/53797072
        *,
        audio_len=1,
        # We set Q' = 4 such that we predict the flattened tokens corresponding
        # to the coarse 4 layers in the second stage.
        n_coarse_quantizers=4,
        # Not specified, but num quantizers must be a power of 2
        # so this is the most reasonable combination.
    ) -> None:
        super().__init__()
        self.semantic_encoder = semantic_encoder
        for param in self.semantic_encoder.model.parameters():
            param.requires_grad = False
        self.acoustic_encoder_decoder = acoustic_encoder_decoder
        for param in self.acoustic_encoder_decoder.model.parameters():
            param.requires_grad = False
        self.semantic_transformer = semantic_transformer
        self.coarse_acoustic_transformer = coarse_acoustic_transformer
        self.fine_acoustic_transformer = fine_acoustic_transformer
        self.audio_len = audio_len
        self.n_coarse_quantizers = n_coarse_quantizers

    def generate(self, x: torch.Tensor, audio_len: int = 3):

        semantic_encode = self.semantic_encoder(x)
        prompt_token, semantic_token = self.semantic_transformer.generate(
            semantic_encode, audio_len * 50
        )

        concat = torch.cat((prompt_token, semantic_token), dim=1)

        coarse_acoustic_tokens, fine_acoustic_tokens, audio_scales = (
            self.acoustic_encoder_decoder.encode(x)
        )

        coarse_conditioning = torch.cat((concat, coarse_acoustic_tokens), dim=1)
        _, coarse_tokens = self.coarse_acoustic_transformer.generate(
            coarse_conditioning, audio_len * (75 * self.n_coarse_quantizers)
        )
        coarse_tokens = coarse_tokens.unsqueeze(0).unsqueeze(0)

        new_len = (
            floor(coarse_tokens.shape[3] / self.n_coarse_quantizers)
            * self.n_coarse_quantizers
        )
        coarse_tokens = coarse_tokens[:, :, :, :new_len]
        coarse_tokens = coarse_tokens.reshape(
            1, 1, 4, new_len // self.n_coarse_quantizers
        )

        num_token = floor(coarse_acoustic_tokens.size(-1) / self.n_coarse_quantizers)
        coarse_acoustic_tokens = coarse_acoustic_tokens.reshape(
            (self.n_coarse_quantizers, num_token)
        )
        coarse_acoustic_tokens = coarse_acoustic_tokens.unsqueeze(0).unsqueeze(0)

        output_concat = torch.cat((coarse_acoustic_tokens, coarse_tokens), dim=-1)
        output = self.acoustic_encoder_decoder.decode(output_concat, [None])

        return output["audio_values"]

    @staticmethod
    def from_pretrained(
        models_path: os.PathLike,
        semantic_encoder: W2VHuBert,
        acoustic_encoder_decoder: Encodec,
        semantic_transformer: SemanticTransformer,
        coarse_acoustic_transformer: CoarseAcousticTransformer,
    ):

        state_dict = torch.load(
            models_path
            / "models"
            / "SemanticTransformer"
            / f"{str(type(semantic_transformer).__name__)}.pth"
        )
        semantic_transformer.load_state_dict(state_dict)

        state_dict = torch.load(
            models_path
            / "models"
            / "CoarseAcousticTransformer"
            / f"{str(type(coarse_acoustic_transformer).__name__)}.pth"
        )
        coarse_acoustic_transformer.load_state_dict(state_dict)
        return AudioLM(
            semantic_encoder=semantic_encoder,
            semantic_transformer=semantic_transformer,
            acoustic_encoder_decoder=acoustic_encoder_decoder,
            coarse_acoustic_transformer=coarse_acoustic_transformer,
            fine_acoustic_transformer=None,
        )

    @staticmethod
    def train(
        train_dataloader: AudioDataLoader,
        val_dataloader: AudioDataLoader,
        models_path: os.PathLike,
        fine_acoustic_modelling: bool = False,
        intervals: int = 10,
        early_stop_counter: int = 10,
        early_stopping_range: int = 10,
        epochs: int = 1,
    ):
        w2v_hubert = W2VHuBert()
        encodec = Encodec()

        semantic_transformer = SemanticTransformer()
        if not (
            Path(models_path)
            / "models"
            / f"{str(type(semantic_transformer).__name__)}.pth"
        ).exists():

            semantic_loss = nn.CrossEntropyLoss()
            semantic_optimizer = torch.optim.Adam(
                semantic_transformer.parameters(), lr=0.001
            )

            semantic_trainer = SemanticTrainer(
                semantic_encoder=w2v_hubert,
                semantic_transformer=semantic_transformer,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=None,
                loss=semantic_loss,
                optimizer=semantic_optimizer,
                intervals=intervals,
                save_path=models_path,
                early_stop_counter=early_stop_counter,
                early_stopping_range=early_stopping_range,
                epochs=epochs,
            )
            semantic_trainer.train()
        else:
            state_dict = torch.load(
                models_path
                / "models"
                / f"{str(type(semantic_transformer).__name__)}.pth"
            )
            semantic_transformer.load_state_dict(state_dict)
        coarse_acoustic_transformer = CoarseAcousticTransformer()
        if not (
            Path(models_path)
            / "models"
            / f"{str(type(coarse_acoustic_transformer).__name__)}.pth"
        ).exists():

            coarse_loss = nn.CrossEntropyLoss()
            coarse_optimizer = torch.optim.Adam(
                coarse_acoustic_transformer.parameters(), lr=0.001
            )

            coarse_acoustic_trainer = CoarseAcousticTrainer(
                semantic_encoder=w2v_hubert,
                semantic_transformer=semantic_transformer,
                acoustic_encoder_decoder=encodec,
                coarse_acoustic_transformer=coarse_acoustic_transformer,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=None,
                loss=coarse_loss,
                optimizer=coarse_optimizer,
                intervals=intervals,
                save_path=models_path,
                early_stop_counter=early_stop_counter,
                early_stopping_range=early_stopping_range,
                epochs=epochs,
            )
            coarse_acoustic_trainer.train()

        else:
            state_dict = torch.load(
                models_path
                / "models"
                / f"{str(type(coarse_acoustic_transformer).__name__)}.pth"
            )
            coarse_acoustic_transformer.load_state_dict(state_dict)

    def test(
        self,
        test_dataloader: AudioDataLoader,
        loss: nn.Module = nn.CrossEntropyLoss(),
        intervals=10,
        early_stop_counter=10,
        early_stopping_range=10,
        epochs=1,
    ):

        semantic_trainer = SemanticTrainer(
            semantic_encoder=self.semantic_encoder,
            semantic_transformer=self.semantic_transformer,
            train_dataloader=None,
            val_dataloader=None,
            test_dataloader=test_dataloader,
            loss=loss,
            optimizer=None,
            intervals=intervals,
            save_path=None,
            early_stop_counter=early_stop_counter,
            early_stopping_range=early_stopping_range,
            epochs=epochs,
        )
        semantic_trainer.test()

        coarse_acustic_trainer = CoarseAcousticTrainer(
            semantic_encoder=self.semantic_encoder,
            semantic_transformer=self.semantic_transformer,
            acoustic_encoder_decoder=self.acoustic_encoder_decoder,
            coarse_acoustic_transformer=self.coarse_acoustic_transformer,
            train_dataloader=None,
            val_dataloader=None,
            test_dataloader=test_dataloader,
            loss=loss,
            optimizer=None,
            intervals=intervals,
            save_path=None,
            early_stop_counter=early_stop_counter,
            early_stopping_range=early_stopping_range,
            epochs=epochs,
        )
        coarse_acustic_trainer.test()
