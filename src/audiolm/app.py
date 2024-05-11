"""This module defines the core application logic and its front-end.

The gui has two modes:

- Recording, toggled by the start function.
- Playing, once a 3 second clip is recorded, it can be played before being processed.
"""

from typing import Literal
from textual.app import App, ComposeResult
from textual.widgets import Footer, Button, Static
from textual.containers import Container
from textual_plotext import PlotextPlot
import sounddevice as sd  # type: ignore
import numpy as np

BLOCKSIZE: int = 1024
CHANNELS: int = 1
SAMPLERATE: int = 44100  # Sample rate
REFRESH_INTERVAL: float = 0.1  # Refresh interval in seconds
MAX_RECORDING: float = 3  # Maximum recording time according to the paper
MARKER = "braile"


class Audio(PlotextPlot):
    """Custom widget that inherits from textual_plotext.Plotext

    This widget is responsible for recording sound and playing it.
    """

    __modes = ["recorder", "player"]

    def __init__(self, mode: Literal["recorder", "player"]) -> None:
        """Costructor for the Audio Widget.

        When constructing the object, it creates a stream that gets ahold of the audio device.
        """
        super().__init__()
        self.__record = False
        # self.__play = False
        if mode not in self.__modes:
            raise ValueError(
                f"\nMode `{mode}` is not a valid mode.\nAvailable modes are {self.__modes}"
            )
        # self.__data = []
        if mode == "recorder":
            self.__stream = sd.InputStream(
                samplerate=SAMPLERATE,
                blocksize=BLOCKSIZE,
                channels=CHANNELS,
            )
            self.set_interval(REFRESH_INTERVAL, self.graph_recording)
        else:
            self.__stream = sd.OutputStream(
                samplerate=SAMPLERATE,
                blocksize=BLOCKSIZE,
                channels=CHANNELS,
            )
            self.set_interval(REFRESH_INTERVAL, self.graph_playing)

    def start_recording(self) -> None:
        """Start the input stream.

        The input stream gets enabled and the recording starts.

        """
        self.__stream.start()
        self.__record = True

    def stop_recording(self) -> None:
        """Stop the input stream.

        The input stream gets stopped and the recording stopped.
        This function is automatically run after 3 seconds.
        """
        self.__record = False
        self.__stream.stop()

    def graph_recording(self) -> None:
        """Real Time graphing of the incoming audio waveform."""
        self.__graph_setup()
        audiowave = np.zeros(BLOCKSIZE)
        if self.__record:
            data, _ = self.__stream.read(BLOCKSIZE)
            audiowave = data.flatten().tolist()

        self.plt.plot(audiowave, marker="braille")
        self.refresh()  # Explicitly needed because there is no reactive element.

    def start_playing(self) -> None:
        """TODO: Start the output stream.

        The output stream gets enabled and the audio starts playing.

        """
        raise NotImplementedError("The start playing function is a WIP")
        # self.__stream.start()
        # self.__play = True

    def stop_playing(self) -> None:
        """Stop the input stream.

        The input stream gets stopped and the recording stopped.
        This function is automatically run after 3 seconds.
        """
        raise NotImplementedError("The stop playing function is a WIP")
        # self.__record = False
        # self.__stream.stop()

    def graph_playing(self) -> None:
        """Real Time graphing of the incoming audio waveform."""
        self.__graph_setup()
        audiowave = np.zeros(BLOCKSIZE)
        # if self.__record:
        #     data, _ = self.__stream.read(BLOCKSIZE)
        #     audiowave = data.flatten().tolist()

        self.plt.plot(audiowave, marker="braille")
        self.refresh()  # Explicitly needed because there is no reactive element.

    # Private methods
    def __graph_setup(self):
        """This functions clears the graph from previous data and prepares for incoming."""
        self.plt.clear_figure()
        self.plt.xfrequency(0)
        self.plt.yfrequency(0)
        self.plt.ylim(-10, 10)
        self.plt.theme("dark")


class Recorder(Static):
    def compose(self) -> ComposeResult:

        with Container(id="recorder-cont"):
            with Container(id="buttons"):
                yield Button(label="Start\nRecording", variant="success", id="start")
                yield Button(label="Stop\nRecording", variant="error", id="stop")
                yield Button(label="Restart\nRecording", disabled=True, id="reset")
                yield Button(label="Generate\nAudio", disabled=True, id="gen_audio")
            yield Audio(mode="recorder")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.query_one(Audio).start_recording()
        if event.button.id == "stop":
            self.query_one(Audio).stop_recording()


class Player(Static):
    def compose(self) -> ComposeResult:
        with Container(id="player-cont"):
            with Container(id="buttons"):
                yield Button(label="Start\nPlaying", variant="success", id="start")
                yield Button(label="Stop\nPlaying", variant="error", id="stop")
            yield Audio(mode="player")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.query_one(Audio).start_recording()
        if event.button.id == "stop":
            self.query_one(Audio).stop_recording()


class AudioLMApp(App):
    """A Textual app to manage AudioLM."""

    CSS_PATH = "./../css/app.tcss"

    BINDINGS = []

    def compose(self) -> ComposeResult:
        """Called to add widgets to the app."""
        with Container(id="cont"):
            yield Recorder()
            yield Player()
        yield Footer()
