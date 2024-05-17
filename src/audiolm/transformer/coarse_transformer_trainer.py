"""This module contains a trainer class"""

# https://xylambda.github.io/blog/python/pytorch/machine-learning/2021/01/04/pytorch_trainer.html
import torch
from torch import nn
import tqdm
from math import ceil
import os

from audiolm.transformer.AbsoluteTransformer import TransformerDecoderOnly
from audiolm.acoustic_modelling.custom_encodec import CustomEncodecModel
from audiolm.semantic_acoustic_modeling.W2VHuBERT_Quantizier import W2VHuBERT_Quantizier
from audiolm.data_preparation import AudioDataLoader
from audiolm.costants import DEVICE


class CoarseTransformerTrainer(nn.Module):
    """
    Trainer class for training a Transformer model.

    """

    def __init__(
        self,
        coarse_acoustic_transformer: TransformerDecoderOnly,
        semantic_transformer: TransformerDecoderOnly,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module,
        early_stopping_range: int = 5,
        epochs: int = 10,
    ):
        super().__init__()
        self.semantic_encoder = W2VHuBERT_Quantizier()
        self.semantic_transformer = semantic_transformer

        self.acoustic_encoder = CustomEncodecModel()
        self.coarse_acoustic_transformer = coarse_acoustic_transformer

        self.optimizer = optimizer
        self.epochs = epochs
        self.best_val_loss = float("inf")
        self.early_stopping_range = early_stopping_range
        self.early_stop_counter = 0
        self.loss_function = nn.CrossEntropyLoss()

    def fit(self, __train_dataloader: AudioDataLoader):
        """
        Train the Transformer model.
        """
        return NotImplementedError

    def train_epoch(self, dataloader):
        # self.train()
        train_loss, train_acc = 0, 0
        for batch in dataloader:
            semantic_encode = self.semantic_encoder(batch)
            print(semantic_encode.shape)
            break


if __name__ == "__main__":
    import os
    from audiolm.data_preparation import AudioDataLoader

    dataloader = AudioDataLoader(
        os.getcwd() + "\\data\\datasets\\", 2, max_length_audio=3
    )
    semantic_transformer = TransformerDecoderOnly(
        500,  # Valore massimo del quantizzatore
        768,  # Numero arbitratio
    )
    coarse_acoustic_transformer = acoustic_transfomer = TransformerDecoderOnly(
        1024,  # Valore massimo del quantizzatore
        1024,  # Numero arbitratio
    )
    optimizer = torch.optim.Adam(coarse_acoustic_transformer.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    CoarseTransformerTrainer(
        coarse_acoustic_transformer=coarse_acoustic_transformer,
        semantic_transformer=semantic_transformer,
        optimizer=optimizer,
        loss=loss,
    ).train_epoch(dataloader)
