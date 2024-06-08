"""Unittests for AudioLM model."""

import os
import unittest
import unittest.test
import warnings
from pathlib import Path

import torch
import torchaudio

from audiolm.data_preparation import AudioDataLoader
from audiolm.model import AudioLM
from audiolm.constants import DEVICE


warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.set_warn_always(False)

DATA_PATH = Path(os.getcwd() + "\\data\\datasets")
MODELS_PATH = Path(os.getcwd() + "\\data\\")


class TestAudioLM(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        dataloader = AudioDataLoader(DATA_PATH, max_length_audio=2)

        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            dataloader.split(0.4, 0.3, 0.3)
        )

    def _model_train(self):
        AudioLM.train(
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            models_path=MODELS_PATH,
        )

    def _model_test(self):
        audiolm = AudioLM.from_pretrained(MODELS_PATH)
        audiolm.test(self.test_dataloader)

    def test_generation(self):
        audiolm = AudioLM.from_pretrained(MODELS_PATH)
        elem = next(iter(self.train_dataloader))[0:1, :, :].to(DEVICE)
        test_generate = audiolm.generate(elem, audio_len=3).squeeze(0)
        if DEVICE=="cuda":
            test_generate = test_generate.cpu().detach()
        torch.cuda.empty_cache()
        torchaudio.save(
            DATA_PATH / "generated.flac", test_generate, 24000
        )


if __name__ == "__main__":
    unittest.main()
