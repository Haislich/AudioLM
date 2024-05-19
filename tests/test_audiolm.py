"""Unittests for AudioLM model."""

import os
import unittest
import unittest.test
import warnings
from pathlib import Path

import torch

from audiolm.data_preparation import AudioDataLoader
from audiolm.model import AudioLM

from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.set_warn_always(False)

DATA_PATH = Path(os.getcwd() + "\\data\\datasets")
MODELS_PATH = Path(os.getcwd() + "\\data\\")


class TestAudioLM(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        dataloader = AudioDataLoader(DATA_PATH)
        total_size = len(dataloader)
        split_lengths = [
            int(0.7 * total_size),
            int(0.2 * total_size),
            total_size - int(0.7 * total_size) - int(0.2 * total_size),
        ]
        train, val, test = random_split(dataloader, split_lengths)
        self.train_dataloader = DataLoader(train, batch_size=2, shuffle=True)

        self.val_dataloader = DataLoader(val, batch_size=2, shuffle=True)
        self.test_dataloader = DataLoader(test, batch_size=2, shuffle=True)

    def test_from_pretrained(self):
        AudioLM.from_pretrained(MODELS_PATH)

    def test_model_train(self):
        train_dataloader, val_dataloader = None, None
        AudioLM.train(
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            models_path=MODELS_PATH,
        )

    def test_model_test(self):
        audiolm = AudioLM.from_pretrained(MODELS_PATH)
        audiolm.test(self.test_dataloader)
