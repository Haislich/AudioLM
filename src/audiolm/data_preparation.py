"""This module contain the definition of the dataset and dataloader."""

import os
from pathlib import Path
import copy

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split


def padding_audio(audio, max_len):
    padding = max_len - audio.size(1)
    audio = F.pad(audio, (0, padding), "constant", 0)
    return audio


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

    def __init__(self, path_folder, max_length_audio=3, sample_frequency=16000, max_elems=100):
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
        self.max_elems = max_elems
        self.preprocess_dataset()

    def __len__(self):
        return len(self.data)

    def __collate_audio(self):
        """Returns the file locations."""
        cnt = 0
        path_audios = []
        for dirpath, _, filenames in os.walk(self.path_folder):
            for filename in filenames:
                if cnt >= self.max_elems:
                    return path_audios
                path_to_audio = os.path.join(dirpath, filename)
                if path_to_audio.endswith(".flac"):
                    cnt +=1
                    path_audios.append(path_to_audio)

        return path_audios

    def preprocess_dataset(self):
        """
        Preprocesses the audio data.

        Returns:
            list: List of preprocessed audio tensors.
        """
        overlap = 1 * self.sample_frequency
        for path in self.path_audios:
            # Load the audio file and its sample rate, resampling it if necessary
            audio, sr = torchaudio.load(path, channels_first=True)
            if sr != self.sample_frequency:
                audio = torchaudio.functional.resample(audio, sr, self.sample_frequency)

            samples = audio.size(1)
            start = 0
            while start + self.max_len < samples:
                self.data.append((path, start, start + self.max_len))
                start += self.max_len - overlap
            self.data.append((path, start, samples))

    def __getitem__(self, idx):
        path, start, end = self.data[idx]
        audio, sr = torchaudio.load(
            path,
            channels_first=True,
            frame_offset=int(start),
            num_frames=int(end - start),
        )
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
        max_elems=100,
        dataset=None,
    ):
        if dataset:
            self.dataset = dataset
        else:
            self.dataset = AudioDataset(
                data_path,
                max_length_audio=max_length_audio,
                sample_frequency=sample_frequency,
                max_elems=max_elems,
            )
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.__collate_fn,
        )

    def __len__(self):
        return len(self.dataset)  # Return the length of the data list.

    def __collate_fn(self, batch):
        audio = [elem for elem in batch]
        audio = torch.stack(audio)

        return audio

    def split(
        self, train_size: float = 0.5, val_size: float = 0.3, test_size: float = 0.2
    ):
        assert train_size + val_size + test_size, 1
        train_subset, val_subset, test_subset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        train = AudioDataLoader("", dataset=train_subset)
        val = AudioDataLoader("", dataset=val_subset)
        test = AudioDataLoader("", dataset=test_subset)

        return train, val, test
