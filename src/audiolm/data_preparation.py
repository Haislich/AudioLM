"""This module contain the definition of the dataset and dataloader."""

import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from semantic_acoustic_modeling.utils import (
    padding_audio,
    cut_audio,
)

# TODO: Necessaria la attention mask per il transformer??


class AudioDataset(Dataset):
    """
    Dataset class for audio data.

    Args:
        path_folder (str): Path to the folder containing the audio files.
        max_length_audio (int, optional): Maximum length of the audio in seconds.
        sample_frequency (int, optional): Sample frequency of the audio in Hz. Defaults to 16000.

    Returns:
        torch.Tensor: Audio data tensor.
    """

    def __init__(self, path_folder, max_length_audio=3, sample_frequency=16000):
        super().__init__()
        self.path_folder = Path(path_folder)
        assert (
            self.path_folder.exists()
        ), f'Watch out! "{str(self.path_folder)}" was not found.'
        self.max_length_audio = max_length_audio
        self.sample_frequency = sample_frequency
        self.max_len = max_length_audio * sample_frequency
        self.path_audios = self.__collate_audio()
        self.data = []
        self.preprocess_dataset()

    def __len__(self):
        return len(self.data)

    def __collate_audio(self):
        """Returns the file locations."""
        path_audios = []
        for dirpath, _, filenames in os.walk(self.path_folder):
            for filename in filenames:
                path_to_audio = os.path.join(dirpath, filename)
                if path_to_audio.endswith(".flac"):
                    path_audios.append(path_to_audio)

        return path_audios

    def preprocess_dataset(self):
        """
        Preprocesses the audio data.

        Returns:
            list: List of preprocessed audio tensors.
        """
        overlap = 0.01 * self.max_len
        for path in self.path_audios:
            # Load the audio file and its sample rate, resampling it if necessary
            audio, sr = torchaudio.load(path, channels_first=True)
            if sr != self.sample_frequency:
                audio = torchaudio.functional.resample(audio, sr, self.sample_frequency)

            samples = audio.size(1)
            start = 0
            while start + self.max_len < samples:
                self.data.append((path, start, start + self.max_len))
                start += (self.max_len - overlap)
            self.data.append((path, start, samples))


    def __getitem__(self, idx):
        path, start, end = self.data[idx]
        audio, sr = torchaudio.load(path, channels_first=True, frame_offset=start, num_frames=end - start)
        if sr != self.sample_frequency:
            audio = torchaudio.functional.resample(audio, sr, self.sample_frequency)
        
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        
        if audio.size(1) < self.max_len:
            audio = padding_audio(audio, self.max_len)
        
        return audio


class AudioDataLoader(DataLoader):
    """
    DataLoader for loading audio data.

    Args:
        data_path (str): The path to the audio data.
        batch_size (int, optional): The batch size. Defaults to 2.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        max_length_audio (int, optional): The maximum length of audio in seconds. Defaults to 20.
        sample_frequency (int, optional): The sample frequency of the audio. Defaults to 24000.

    Returns:
        DataLoader: The DataLoader object for loading audio data.
    """

    def __init__(
        self,
        data_path,
        batch_size=2,
        shuffle=False,
        max_length_audio=3,
        sample_frequency=16000,
    ):
        self.dataset = AudioDataset(
            data_path,
            max_length_audio=max_length_audio,
            sample_frequency=sample_frequency,
        )
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.__collate_fn)

    def __len__(self):
        return len(self.dataset)

    def __collate_fn(self, batch):
        audio = [elem for elem in batch]
        audio = torch.stack(audio)

        return audio




##just for test


data = "/Users/valerio/Desktop/ei"

a = AudioDataLoader(data_path=data, batch_size=5, shuffle=False, max_length_audio=3)
# dataloader = a.return_DataLoader()
print(a.__len__())
for batch in a:
    print(batch.shape)
