import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from time import sleep


def plot_waveform():
    plt.ion()  # Turn on interactive mode for real-time plotting

    fig, ax = plt.subplots()
    x = np.arange(0, CHUNKSIZE)  # Assuming CHUNKSIZE is the size of each audio chunk
    (line,) = ax.plot(x, np.random.rand(CHUNKSIZE))  # Initialize an empty plot
    ax.set_ylim(-10, 10)  # Assuming audio samples are in the range [-1, 1]
    ax.set_title("Real-time Audio Waveform")

    while True:
        data = sd.rec(frames=CHUNKSIZE, samplerate=RATE, channels=1, blocking=True)
        audio_data = data.flatten()
        line.set_ydata(audio_data)
        fig.canvas.draw()
        fig.canvas.flush_events()
        sleep(0.01)


if __name__ == "__main__":
    CHUNKSIZE = 1024  # Adjust this according to your requirements
    RATE = 44100  # Sample rate

    plot_waveform()
