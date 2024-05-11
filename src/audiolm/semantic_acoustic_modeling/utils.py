import torch
import torch.nn.functional as F


def padding_audio(audio, max_len):
    padding = max_len - audio.size(1)
    audio = F.pad(audio, (0, padding), "constant", 0)
    return audio


def cut_audio(audio, max_len):
    if audio.size(1) < max_len:
        return audio
    return audio[:, :max_len]


def cut_the_audios(audio, max_len):

    # Padding audios or cutting them if they are too long
    list_audio_slices = []
    while audio.size(1) > max_len:
        audio_1 = audio[:, :max_len]

        if audio_1.size(1) < max_len:
            audio_1 = padding_audio(audio_1, max_len)
        list_audio_slices.append(audio_1)

        audio = audio[:, max_len:]
    if audio.size(1) < max_len:
        audio = padding_audio(audio, max_len)
        list_audio_slices.append(audio)

    return list_audio_slices
