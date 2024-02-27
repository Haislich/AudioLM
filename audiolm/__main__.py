from textual.app import App, ComposeResult
from textual.widgets import Footer, Button, Static
from textual.containers import Container
from textual_plotext import PlotextPlot
from textual.reactive import reactive
import sounddevice as sd

import numpy as np
import queue
from time import monotonic

BLOCKSIZE = 512
CHANNELS = 1
SAMPLERATE = 44100  # Sample rate
REFRESH_INTERVAL = 0.1  # Refresh interval in seconds
MAX_RECORDING = 3  # Maximum recording time according to the paper
DATALEN = int(MAX_RECORDING / REFRESH_INTERVAL)


class Audio(PlotextPlot):
    timer: reactive = reactive(0.0)
    __record: reactive = reactive(False)
    __playing: bool = False
    __audio_queue: queue.Queue = queue.Queue()

    def __init__(self) -> None:
        super().__init__()
        self.__stream = sd.InputStream(
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            callback=lambda indata, *_: self.__audio_queue.put_nowait(indata),
            channels=CHANNELS,
        )
        # self.__data = np.zeros((DATALEN))

    def on_mount(self) -> None:
        self.set_interval(REFRESH_INTERVAL, self.graph_audio)

    # https://python-sounddevice.readthedocs.io/en/0.3.14/api.html#sounddevice.InputStream
    # def __audio_callback(indata:np.ndarray,frames:int,time, status) -> None:

    def start_recording(self) -> None:
        self.__record = True
        self.__stream.start()

    def stop_recording(self) -> None:
        self.__record = False
        self.__stream.stop()

    def graph_audio(self) -> None:
        self.plt.clear_figure()
        self.plt.xfrequency(0)
        self.plt.yfrequency(0)
        self.plt.ylim(-4, 4)
        audiowave = np.zeros(BLOCKSIZE).tolist()
        if self.__record:
            data = sd.rec(
                frames=BLOCKSIZE, samplerate=SAMPLERATE, channels=1, blocking=True
            )
            audiowave = data.flatten()
        # ignore
        self.plt.plot(audiowave, marker="braille")
        self.refresh()


class Recorder(Static):
    def compose(self) -> ComposeResult:
        self.start_recording = Button(
            label="Start\nRecording", variant="success", id="start"
        )
        self.stop_recording = Button(
            label="Stop\nRecording", variant="error", id="stop"
        )
        self.restart = Button(label="Restart\nRecording", disabled=True, id="reset")
        self.generate_audio = Button(
            label="Generate\nAudio", disabled=True, id="gen_audio"
        )
        with Container(id="cont"):
            with Container(id="buttons"):
                yield self.start_recording
                yield self.stop_recording
                yield self.restart
                yield self.generate_audio
            yield Audio()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.query_one(Audio).start_recording()
        if event.button.id == "stop":
            self.query_one(Audio).stop_recording()


class AudioLMApp(App):
    """A Textual app to manage AudioLM."""

    CSS_PATH = "./css/app.tcss"

    BINDINGS = []

    def compose(self) -> ComposeResult:
        """Called to add widgets to the app."""

        yield Footer()
        yield Recorder()


if __name__ == "__main__":
    app = AudioLMApp()
    app.run()
