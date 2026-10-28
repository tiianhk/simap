from abc import ABC, abstractmethod
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator
import os
import sys
import tempfile

import numpy as np
import pysfizz
import fluidsynth


@contextmanager
def suppress_stderr():
    stderr_fd = sys.stderr.fileno()
    original_stderr_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(original_stderr_fd, stderr_fd)
        os.close(original_stderr_fd)


@contextmanager
def capture_stderr():
    """
    Capture all output written to stderr, including C-level output (FD 2).
    Meant to be used in a single process (e.g., each process in multiprocessing).
    Returns a list containing captured stderr content as a string.
    """
    stderr_fd = sys.stderr.fileno()
    original_stderr_fd = os.dup(stderr_fd)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    temp_fd = os.open(temp_file.name, os.O_WRONLY | os.O_TRUNC)
    os.dup2(temp_fd, stderr_fd)
    os.close(temp_fd)
    captured = []
    try:
        yield captured
    finally:
        os.dup2(original_stderr_fd, stderr_fd)
        os.close(original_stderr_fd)
        with open(temp_file.name, "rb") as f:
            content = f.read()
            if content:
                content = content.decode("latin-1")
                captured.append(content)
        os.unlink(temp_file.name)


class SynthInterface(ABC):
    def __init__(self, backend_synth: object):
        self.backend_synth = backend_synth
        self.channel_num = self._get_channel_num()
        self.sample_rate = self._get_sample_rate()

    def _validate_args(self, channel: int, midi_note: int, velocity: int) -> None:
        """
        Validate the arguments of get_active_voices() and render_note().

        Args:
            channel (int): Channel number, in the range [0, self.channel_num - 1].
            midi_note (int): MIDI note number, in the range [0, 127].
            velocity (int): Velocity of the note, in the range [0, 127].
        """
        if not isinstance(channel, int):
            raise TypeError(f"channel must be int, got {type(channel).__name__}")
        if not isinstance(midi_note, int):
            raise TypeError(f"midi_note must be int, got {type(midi_note).__name__}")
        if not isinstance(velocity, int):
            raise TypeError(f"velocity must be int, got {type(velocity).__name__}")
        if channel < 0 or channel >= self.channel_num:
            if self.channel_num == 1:
                raise ValueError(f"channel must be 0, got {channel}")
            else:
                raise ValueError(
                    f"channel must be in the range [0, {self.channel_num - 1}], got {channel}"
                )
        if midi_note < 0 or midi_note > 127:
            raise ValueError(
                f"midi_note must be in the range [0, 127], got {midi_note}"
            )
        if velocity < 0 or velocity > 127:
            raise ValueError(f"velocity must be in the range [0, 127], got {velocity}")

    def get_text_label(self, channel: int) -> str:
        self._validate_args(channel, 0, 0)
        return self._get_text_label_impl(channel)

    def get_bank_preset(self, channel: int) -> tuple[int, int] | tuple[None, None]:
        """
        Get where a channel's instrument sits in the file it came from.

        Args:
            channel (int): see _validate_args()

        Returns:
            tuple: (bank, preset) as the file declares them, or (None, None) for sfz.
        """
        self._validate_args(channel, 0, 0)
        return self._get_bank_preset_impl(channel)

    def get_active_voices(self, channel: int, midi_note: int, velocity: int) -> int:
        """
        Get the number of active voices produced by a given MIDI note.

        Args:
            channel, midi_note, velocity: see _validate_args()

        Returns:
            int: Number of active voices produced by the note.
        """
        self._validate_args(channel, midi_note, velocity)
        return self._get_active_voices_impl(channel, midi_note, velocity)

    def render_note(
        self,
        channel: int,
        midi_note: int,
        velocity: int,
        note_duration: int | float,
        render_duration: int | float,
    ) -> np.ndarray:
        """
        Render a single MIDI note.

        Args:
            channel, midi_note, velocity: see _validate_args()
            note_duration (int or float): Duration the key is pressed, in seconds.
            render_duration (int or float): Total length of the rendered waveform, in seconds.

        Returns:
            np.ndarray: Rendered waveform with shape (channels, samples).
        """
        self._validate_args(channel, midi_note, velocity)
        if not isinstance(note_duration, (int, float)):
            raise TypeError(
                f"note_duration must be int or float, got {type(note_duration).__name__}"
            )
        if not isinstance(render_duration, (int, float)):
            raise TypeError(
                f"render_duration must be int or float, got {type(render_duration).__name__}"
            )
        if note_duration < 0:
            raise ValueError(f"note_duration must be non-negative, got {note_duration}")
        if render_duration < 0:
            raise ValueError(
                f"render_duration must be non-negative, got {render_duration}"
            )
        return self._render_note_impl(
            channel, midi_note, velocity, note_duration, render_duration
        )

    def get_layers(self, channel: int, midi_note: int) -> list:
        self._validate_args(channel, midi_note, 0)
        return self._get_layers_impl(channel, midi_note)

    @abstractmethod
    def _get_channel_num(self) -> int:
        pass

    @abstractmethod
    def _get_sample_rate(self) -> int:
        pass

    @abstractmethod
    def _get_text_label_impl(self, channel: int) -> str:
        pass

    @abstractmethod
    def _get_bank_preset_impl(self, channel: int) -> tuple[int, int] | tuple[None, None]:
        pass

    @abstractmethod
    def _get_active_voices_impl(self, channel: int, midi_note: int, velocity: int) -> int:
        pass

    @abstractmethod
    def _render_note_impl(
        self,
        channel: int,
        midi_note: int,
        velocity: int,
        note_duration: int | float,
        render_duration: int | float,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _get_layers_impl(self, channel: int, midi_note: int) -> list:
        pass

    @abstractmethod
    def delete(self) -> None:
        pass


def _sfz_sample_id(sfz_path, name):
    """What a region plays, as sample_reader keys it."""
    if not name:
        return None  # a region that names no sample, so there is no audio
    if name.startswith("*"):
        return name  # sfizz generates it, and the name says which waveform
    return (Path(sfz_path).parent / name).resolve()


class SfizzInterface(SynthInterface):
    def _get_channel_num(self) -> int:
        return 1

    def _get_sample_rate(self) -> int:
        return self.backend_synth.get_sample_rate()

    def _get_text_label_impl(self, channel: int) -> str:
        return Path(self.backend_synth.path).stem

    def _get_bank_preset_impl(self, channel: int) -> tuple[None, None]:
        return None, None  # sfz has no bank or preset

    def _get_active_voices_impl(self, channel: int, midi_note: int, velocity: int) -> int:
        before = self.backend_synth._synth.get_num_active_voices()
        self.backend_synth._synth.note_on(
            0, midi_note, velocity
        )  # 0 is delay (in samples), not channel index
        after = self.backend_synth._synth.get_num_active_voices()
        self.backend_synth._synth.all_sound_off()  # clear note
        return after - before

    def _render_note_impl(
        self,
        channel: int,
        midi_note: int,
        velocity: int,
        note_duration: int | float,
        render_duration: int | float,
    ) -> np.ndarray:
        self.backend_synth._synth.all_sound_off()  # avoid voice accumulation
        return self.backend_synth.render_note(
            midi_note, velocity, note_duration, render_duration
        )

    def _get_layers_impl(self, channel: int, midi_note: int) -> list:
        note_info = self.backend_synth.get_note_info(midi_note)
        layers = []
        for sample in note_info:
            layers.append(
                {
                    # the file this region plays, resolved, which is how
                    # sample_reader keys what it finds in the folder. sfizz
                    # names generated waveforms with a leading star and has
                    # nothing at all for a region that named no sample
                    "sample_id": _sfz_sample_id(self.backend_synth.path,
                                                sample["sample_id"]),
                    "pitch_shift_semitones": sample["pitch_shift_semitones"],
                    "lovel": sample["lovel"],
                    "hivel": sample["hivel"],
                    "loopmode": float("nan"),  # sfizz's loopmode is not reliable
                    "gens": None,
                    "mods": None,
                    "opcodes": {
                        k: v
                        for k, v in sample["opcodes"].items()
                        if k != "sample" and not k.endswith("_label")
                    } if sample["opcodes"] is not None else None,
                }
            )
        return layers

    def delete(self) -> None:
        pass


class FluidSynthInterface(SynthInterface):
    def _get_channel_num(self) -> int:
        max_channel = self.backend_synth.get_setting("synth.midi-channels")
        for ch in range(max_channel):
            sfid = self.backend_synth.program_info(ch)[0]
            if sfid == 0:
                return ch
        return max_channel

    def _get_sample_rate(self) -> int:
        return self.backend_synth.get_setting("synth.sample-rate")

    def _get_text_label_impl(self, channel: int) -> str:
        return self.backend_synth.channel_info(channel)[-1]

    def _get_bank_preset_impl(self, channel: int) -> tuple[int, int]:
        _, bank, preset = self.backend_synth.program_info(channel)
        return bank, preset

    def _get_active_voices_impl(self, channel: int, midi_note: int, velocity: int) -> int:
        before = self.backend_synth.get_active_voice_count()
        self.backend_synth.noteon(channel, midi_note, velocity)
        after = self.backend_synth.get_active_voice_count()
        self.backend_synth.all_sounds_off(-1)  # request: kill active voices
        _ = self.backend_synth.get_samples(
            64  # sample num of a single buffer (which is a hard-coded value in fluidsynth)
        )  #  writing a buffer deterministically executes voice killing
        return after - before

    def _render_note_impl(
        self,
        channel: int,
        midi_note: int,
        velocity: int,
        note_duration: int | float,
        render_duration: int | float,
    ) -> np.ndarray:
        self.backend_synth.all_sounds_off(-1)  # avoid voice accumulation
        _ = self.backend_synth.get_samples(64)
        note_sample_num = int(self.sample_rate * note_duration)
        render_sample_num = int(self.sample_rate * render_duration)
        samples = []
        self.backend_synth.noteon(channel, midi_note, velocity)
        samples = np.append(samples, self.backend_synth.get_samples(note_sample_num))
        self.backend_synth.noteoff(channel, midi_note)
        if render_sample_num > note_sample_num:
            samples = np.append(
                samples,
                self.backend_synth.get_samples(render_sample_num - note_sample_num),
            )
        samples = samples.reshape(-1, 2).T
        if render_sample_num < note_sample_num:
            samples = samples[:, :render_sample_num]
        samples = samples.astype(np.float32) / 32768.0
        return samples

    def _get_layers_impl(self, channel: int, midi_note: int) -> list:
        note_info = self.backend_synth.get_note_info(channel, midi_note)
        layers = []
        for sample in note_info:
            layers.append(
                {
                    # the span the sample header gives, which names the record
                    # in the file. Fluidsynth's own id is the address it loaded
                    # the sample to, so it says nothing once the file is closed
                    "sample_id": (sample["source_start"], sample["source_end"]),
                    "pitch_shift_semitones": sample["pitch_shift_semitones"],
                    "lovel": sample["vello"],
                    "hivel": sample["velhi"],
                    "loopmode": sample["loopmode"],
                    "gens": sample["gens"],
                    "mods": sample["mods"],
                    "opcodes": None,  # sfz only
                }
            )
        return layers

    def delete(self) -> None:
        self.backend_synth.delete()


def synth_factory(synth: object) -> SynthInterface:
    if isinstance(synth, pysfizz.Synth):
        return SfizzInterface(synth)
    elif isinstance(synth, fluidsynth.Synth):
        return FluidSynthInterface(synth)
    else:
        raise TypeError(f"{type(synth).__name__} is not supported")


class Loader(ABC):
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def load(self, path: Path) -> list[SynthInterface] | None:
        if not path.is_file():
            raise FileNotFoundError(f"{path} is not a valid file")
        synths = list(self._load_impl(path))
        if not synths:
            return None
        return [synth_factory(s) for s in synths]

    @abstractmethod
    def _load_impl(self, path: Path) -> Iterator[object]:
        pass


class SfizzLoader(Loader):
    def _load_impl(self, path: Path) -> Iterator[pysfizz.Synth]:
        synth = pysfizz.Synth(
            sample_rate=self.sample_rate
        )  # TODO: select an appropriate block_size
        if synth.load_sfz_file(path):
            yield synth


class FluidSynthLoader(Loader):
    FLUIDSYNTH_MAX_CHANNEL = 256
    FLUIDSYNTH_UNSET_PRESET = 128
    FLUIDSYNTH_MAX_PACKED_BANK = 16383
    FLUIDSYNTH_MAX_PACKED_PRESET = 255

    def _get_new_synth(self, path: Path) -> tuple[fluidsynth.Synth | None, int]:
        path = str(path)
        with suppress_stderr():
            synth = fluidsynth.Synth(
                samplerate=self.sample_rate,
                channels=FluidSynthLoader.FLUIDSYNTH_MAX_CHANNEL,
                gain=1.0,
                **{"synth.polyphony": 256},  # sufficient for simap v0.0.1
            )
        with capture_stderr() as stderr:
            sfid = synth.sfload(path)
        if stderr or sfid == -1:
            synth.delete()
            synth = None
        return synth, sfid

    def _load_impl(self, path: Path) -> Iterator[fluidsynth.Synth]:
        current_synth, current_sfid = self._get_new_synth(path)
        if current_synth is None:
            return  # invalid path, stop generating synths
        presets = list(
            dict.fromkeys(  # a soundfont can declare >1 presets at one (bank, preset) and only the first one is reachable by fluidsynth
                (bank, preset)
                for bank, preset, _ in current_synth.sfpresets(current_sfid)
            )
        )
        channel = 0
        for bank, preset in presets:
            if (
                preset == FluidSynthLoader.FLUIDSYNTH_UNSET_PRESET
                or not 0 <= bank <= FluidSynthLoader.FLUIDSYNTH_MAX_PACKED_BANK
                or not 0 <= preset <= FluidSynthLoader.FLUIDSYNTH_MAX_PACKED_PRESET
            ):
                print(
                    f"When loading {path}: Skipping this preset, fluidsynth "
                    f"cannot handle bank={bank}, preset={preset}"
                )
                continue
            if channel == FluidSynthLoader.FLUIDSYNTH_MAX_CHANNEL:
                yield current_synth  # 256 channels are full, yield the current synth
                current_synth, current_sfid = self._get_new_synth(path)
                channel = 0  # reset channel
            current_synth.program_select(channel, current_sfid, bank, preset)
            channel += 1
        if channel == 0:
            current_synth.delete()
            return  # no channel loaded
        yield current_synth


def load(path: Path | str, sample_rate: int) -> list[SynthInterface] | None:
    path = Path(path)
    if path.suffix.lower() == ".sfz":
        loader = SfizzLoader(sample_rate)
    elif path.suffix.lower() in [".sf2", ".sf3"]:
        loader = FluidSynthLoader(sample_rate)
    else:
        raise ValueError(f"unsupported file extension: {path.suffix}")
    return loader.load(path)
