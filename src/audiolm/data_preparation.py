import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path

import os 
import librosa

##TODO: Necessaria la attention mask per il transformer?? 


class AudioDataset(Dataset):
    def __init__(self, path_folder, max_length_audio=None, sample_frequency=24000):
        super().__init__()
        self.path_folder = Path(path_folder)
        assert self.path_folder.exists(), f'Watch out! "{str(self.path_folder)}" was not found.'
        self.data = self.__collate_audio__(self.path_folder)
        self.max_length_audio = max_length_audio
        self.sample_frequency = sample_frequency

    def __len__(self):
        return len(self.data)

    def __collate_audio__(self, path_folder):
        path_audios = []
        for dirpath, _, filenames in os.walk(path_folder):
            for filename in filenames: 
                path_to_audio = os.path.join(dirpath, filename)
                if path_to_audio.endswith(".flac"):
                    path_audios.append(path_to_audio)
        
        return path_audios

    def __getitem__(self, idx):
        data = self.data[idx]
        audio, sr = librosa.load(data, sr=self.sample_frequency)
        audio = torch.tensor(audio)

        #As said in the paper, we need to use one-only channel audio
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        
        #Padding audios or cutting them if they are too long
        if self.max_length_audio is not None:
            if audio.size(1) > self.max_length_audio:
                audio = audio[:self.max_length_audio]
            else:
                padding = self.max_length_audio - audio.size(1)
                audio = F.pad(audio, (0, padding), 'constant', 0)     

        audio = audio.squeeze(0) #We don't need anymore the channel dimension

        return audio

