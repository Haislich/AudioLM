from textual.app import App, ComposeResult
from textual.widgets import Footer, Button, Static
from textual.containers import Container
from textual_plotext import PlotextPlot
from textual.reactive import reactive


class Audio(PlotextPlot):
    audiowave = reactive(0.0)
    __inc = 0

    def on_mount(self) -> None:
        self.audiowave = self.set_interval(1 / 10, self.render_graph)

    def increment(self) -> None:
        self.__inc += 1
        self.render_graph()

    def render_graph(self) -> None:
        self.plt.clear_figure()
        self.plt.xfrequency(0)
        self.plt.yfrequency(0)
        self.audiowave = self.plt.sin(self.__inc, 20, phase=self.__inc)
        self.plt.plot(self.audiowave)


class Recorder(Static):
    def compose(self) -> ComposeResult:
        self.start_recording = Button(
            label="Start\nRecording", variant="success", id="start"
        )

        self.stop_recording = Button(
            label="Stop\nRecording", variant="error", disabled=True, id="stop"
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
            print(event.button.id)
            self.query_one(Audio).increment()


class AudioLMApp(App):
    """A Textual app to manage stopwatches."""

    CSS_PATH = "./css/app.tcss"

    BINDINGS = []

    def compose(self) -> ComposeResult:
        """Called to add widgets to the app."""

        yield Footer()
        yield Recorder()


if __name__ == "__main__":
    app = AudioLMApp()
    app.run()
