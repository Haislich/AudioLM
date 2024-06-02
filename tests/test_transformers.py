"""Unittests for trainer class."""

import os
import unittest
import unittest.test
import warnings
from pathlib import Path

import torch
from torch import nn

from audiolm.data_preparation import AudioDataLoader
from audiolm.w2v_hubert import W2VHuBert
from audiolm.absolute_transformer import (
    SemanticTransformer,
    CoarseAcousticTransformer,
    FineAcousticTransformer,
)
from audiolm.trainer import SemanticTrainer, CoarseAcousticTrainer, FineAcousticTrainer
from audiolm.encodec import Encodec

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.set_warn_always(False)

DATA_PATH = Path(os.getcwd()) / Path("..") / Path("data") / Path("datasets")
MODEL_PATH = Path(os.getcwd()) / Path("..") / Path("data")
INTERVALS = 10
EARLY_STOP_COUNTER = 0
EARLY_STOPPING_RANGE = 5
EPOCHS = 10


class TestTransformer(unittest.TestCase):
    semantic_encoder = W2VHuBert()
    acoustic_encoder_decoder = Encodec()
    train_dataloader = AudioDataLoader(DATA_PATH / "train", batch_size=4, max_elems=10)
    val_dataloader = AudioDataLoader(DATA_PATH / "val", batch_size=4, max_elems=5)
    test_dataloader = AudioDataLoader(DATA_PATH / "test", batch_size=4, max_elems=2)

    def test_generate(self):

        semantic_transformer = SemanticTransformer()
        state_dict = torch.load(
            MODEL_PATH / "models" / f"{str(type(semantic_transformer).__name__)}.pth"
        )
        semantic_transformer.load_state_dict(state_dict)
        coarse_acoustic_transformer = CoarseAcousticTransformer(num_heads=16, layers=12)
        coarse_loss = nn.CrossEntropyLoss()
        coarse_optimizer = torch.optim.Adam(
            coarse_acoustic_transformer.parameters(), lr=0.001
        )
        coarse_acoustic_trainer = CoarseAcousticTrainer(
            semantic_encoder=self.semantic_encoder,
            semantic_transformer=semantic_transformer,
            acoustic_encoder_decoder=self.acoustic_encoder_decoder,
            coarse_acoustic_transformer=coarse_acoustic_transformer,
            train_dataloader=self.train_dataloader,
            val_dataloader=sefl.val_dataloader,
            test_dataloader=self.test_dataloader,
            loss=coarse_loss,
            optimizer=coarse_optimizer,
            intervals=INTERVALS,
            save_path=MODEL_PATH,
            early_stop_counter=EARLY_STOP_COUNTER,
            early_stopping_range=EARLY_STOPPING_RANGE,
            epochs=EPOCHS,
        )
        coarse_acoustic_trainer.train()


if __name__ == "__main__":
    unittest.main()
