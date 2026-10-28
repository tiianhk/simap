import ctypes
import json
import logging
import random
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data._utils.collate import (
    collate,
    default_collate_fn_map,
    collate_tensor_fn,
)

from simap.keymap import Keymap
from simap.memory_usage import load_or_measure
from simap.synth_interface import load
from simap.user_config import get_user_config

logger = logging.getLogger(__name__)

SPLITS = ("train", "valid", "test")


class BaseSimapDataset(Dataset):
    """
    PyTorch Dataset that samples instruments from SIMAP and yields audio (and optionally text)
    for model training.

        metadata[idx]
        |
        v
    +-------+     channel_id    --> ch
    |       |     keys_of(idx)   --> random pitch p1, p2
    | synth |     velocity      --> random v1, v2
    |       |     note_duration --> random d1, d2
    +-------+     release_dur   --> r
        |
        +---> render_note(ch, p1, v1, d1, d1 + r) --> view1 ─┐
        |                                                    ├─> {"audio": (view1, view2)}
        +---> render_note(ch, p2, v2, d2, d2 + r) --> view2 ─┘

    An item also carries what it was made from: metadata_idx, the pitch and
    note_dur actually drawn, whether that pitch sustains, and the text. Each of
    pitch, note_dur and if_sustained is a pair when there are two views, matching
    audio, so a view can always be traced to the note that produced it.

    Recommended modes:
     - get_two_audio_views=True, get_text_label=False -> audio contrastive pairs
     - get_two_audio_views=False, get_text_label=True -> audio-text contrastive pairs

    Splits
    ------
    split picks one of the three lists in split.json, which is a partition of the
    corpus: train by default, then valid and test, or None for the whole corpus.

    None and train draw a fresh pitch, velocity and duration every time an instrument
    comes round. valid and test seed the draw on the instrument's own index instead, 
    so an instrument is always asked the same notes.

    Two index spaces
    ----------------
    pos          where a sample sits in this dataset, 0 to len(self) - 1. It is what
                 the DataLoader asks for, and it means nothing outside this object:
                 a chunked dataset reassigns every position each chunk.
    metadata_idx which instrument that is, 0 to len(metadata) - 1. It is the identity
                 the whole project shares -- `simap render` takes it, keymap.parquet
                 and the embedding stores are keyed by it, the subset files list it.

    self.metadata_idx is the map between them, and it is a selection: every entry is
    a real corpus index and nothing is ever renumbered. Subsetting, dropping and
    chunking all narrow or rebuild it, which is why self.metadata is never filtered
    -- its positions are the identity, so they must not move. Anything else keyed on
    an instrument belongs alongside metadata, full length, rather than alongside the
    selection, so that rebuilding the selection cannot desynchronise it.

    Every item carries its metadata_idx, so a prediction can always be traced back
    to the instrument it came from.
    """

    def __init__(
        self,
        sample_rate,
        split: str | None = "train",
        get_two_audio_views: bool = False,
        get_text_label: bool = True,
        text: list[str] | None = None,
        drop_empty_text: bool = True,
        min_pitch: int = 0,
        max_pitch: int = 127,
        max_shift: float | None = 3.0,
        min_velocity: int = 20,
        max_velocity: int = 127,
        min_note_dur: float = 0.25,
        max_note_dur: float = 4.0,
        release_dur: float = 3.0,
    ):
        self.sample_rate = sample_rate
        if split is not None and split not in SPLITS:
            raise ValueError(f"split must be None or one of {SPLITS}, "
                             f"not {split!r}")
        self.split = split
        self.get_two_audio_views = get_two_audio_views
        self.get_text_label = get_text_label
        self.drop_empty_text = drop_empty_text
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.release_dur = release_dur

        self.log_min_note_dur = math.log(min_note_dur)
        self.log_max_note_dur = math.log(max_note_dur)

        config = get_user_config()
        if "simap_dir" not in config:
            raise ValueError(
                "The dataset directory is not configured. "
                "Run `simap set-dir dataset <path>` to set the dataset directory."
                # TODO: add the download command
            )
        self.simap_dir = Path(config["simap_dir"])
        with open(self.simap_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        # default uses the raw author labels
        self.text = [m["text_label"] for m in self.metadata] if text is None else text
        if len(self.text) != len(self.metadata):
            raise ValueError(
                f"text has {len(self.text):,} entries for {len(self.metadata):,} "
                f"instruments; it is indexed by metadata_idx, so it must cover "
                f"the whole corpus")

        # the split comes first, so what the filters below report is what this
        # split lost rather than what the corpus did. None is the whole corpus,
        # train and test together, which is what an unsplit run used to serve
        if self.split is None:
            self.metadata_idx = np.arange(len(self.metadata), dtype=np.int32)
        else:
            with open(self.simap_dir / "split.json", "r") as f:
                self.metadata_idx = np.array(json.load(f)[self.split], np.int32)

        # drop instruments with empty text labels
        if self.drop_empty_text:
            keep = [i for i in self.metadata_idx if self.text[i] != ""]
            if len(keep) != len(self.metadata_idx):
                logger.info("%s of %s instruments have no text and are dropped",
                            len(self.metadata_idx) - len(keep), len(self.metadata_idx))
                self.metadata_idx = np.array(keep, np.int32)

        # keymap data
        self.keymap = Keymap(self.simap_dir / "keymap.parquet")
        ok = self.keymap.max_abs_shift() <= (np.inf if max_shift is None else max_shift)
        ok[:, :min_pitch] = False
        ok[:, max_pitch + 1:] = False
        self._keys = np.nonzero(ok)[1].astype(np.uint8)
        self._key_offs = np.concatenate(([0], np.cumsum(ok.sum(1)))).astype(np.int64)
        self._sustain = self.keymap.is_sustained()[ok]

        playable = np.diff(self._key_offs)[self.metadata_idx] > 0
        if not playable.all():
            logger.info(
                "%s of %s instruments have no key in %s-%s with max_shift=%s "
                "and are dropped", int((~playable).sum()), len(self.metadata_idx),
                min_pitch, max_pitch, max_shift)
            self.metadata_idx = self.metadata_idx[playable]

    def keys_of(self, metadata_idx):
        """The keys this instrument may be asked for, as a view into one buffer."""
        return self._keys[self._key_offs[metadata_idx]:self._key_offs[metadata_idx + 1]]

    def sustain_of(self, metadata_idx):
        """Whether each of keys_of(metadata_idx) sustains, in the same order."""
        return self._sustain[self._key_offs[metadata_idx]:self._key_offs[metadata_idx + 1]]

    def __len__(self):
        return len(self.metadata_idx)

    def get_sampler_weights(self):
        """One weight per served instrument, in position order."""
        return np.diff(self._key_offs)[self.metadata_idx]

    def _render(self, synth, ch, pitch, velocity, note_dur):
        audio = torch.from_numpy(
            synth.render_note(ch, pitch, velocity, note_dur,
                              note_dur + self.release_dur))
        return audio

    def _get_synths(self, file):
        """The loaded soundfont, from wherever this subclass keeps it."""
        raise NotImplementedError

    def _clean_up_synths(self, synths):
        """Release what _get_synths returned, if it needs releasing."""
        raise NotImplementedError

    def __getitem__(self, pos):
        metadata_idx = self.metadata_idx[pos]
        metadata = self.metadata[metadata_idx]
        synths = self._get_synths(metadata["file"])
        synth = synths[metadata["synth_id"]]
        ch = metadata["channel_id"]
        keys, sustain = self.keys_of(metadata_idx), self.sustain_of(metadata_idx)
        # training draws fresh notes every epoch, which is the point of it. valid
        # and test draw from the instrument's own index instead, so an instrument
        # is always asked the same notes and a metric moves only when the model
        # does
        rng = (random.Random(int(metadata_idx))
               if self.split in ("valid", "test") else random)
        # the position is drawn rather than the key, so the pitch and whether it
        # sustains come from one draw. int() because the keys are uint8 and the
        # synth type-checks for a plain int rather than coercing
        j1, j2 = rng.choices(range(len(keys)), k=2)
        p1, p2 = int(keys[j1]), int(keys[j2])
        v1, v2 = rng.sample(
            range(self.min_velocity, self.max_velocity + 1), 2)
        d1 = math.exp(rng.uniform(self.log_min_note_dur, self.log_max_note_dur))
        d2 = math.exp(rng.uniform(self.log_min_note_dur, self.log_max_note_dur))
        view1 = self._render(synth, ch, p1, v1, d1)
        view2 = (self._render(synth, ch, p2, v2, d2)
                 if self.get_two_audio_views else None)

        def both(a, b):
            """Per view when there are two of them, otherwise just the one."""
            return (a, b) if self.get_two_audio_views else a

        out = {
            "metadata_idx": int(metadata_idx),
            "audio": both(view1, view2),
            "pitch": both(p1, p2),
            "note_dur": both(d1, d2),
            "if_sustained": both(bool(sustain[j1]), bool(sustain[j2])),
        }
        if self.get_text_label:
            out["text"] = self.text[metadata_idx]
        self._clean_up_synths(synths)
        return out


class ChunkedSimapDataset(BaseSimapDataset):
    def __init__(self, *args, ram_limit: float, **kwargs):
        super().__init__(*args, **kwargs)
        if ram_limit < 2.0:
            raise ValueError("ram_limit must be greater than 2.0")
        self.ram_limit = ram_limit
        self.libc = ctypes.CDLL("libc.so.6")       # linux only, for malloc_trim

        # what each soundfont costs to hold, so a chunk can be filled to the
        # budget. Cached beside the user config; measured here the first time,
        # which is a few minutes over the whole corpus and happens once
        self.memory_usage = load_or_measure()

        # the base selected across the whole corpus; from here the selection is
        # per chunk, and get_files is what remembers the whole of it
        self.files = self.get_files()
        self.filenames = [
            f for f in self.files.keys() if len(self.files[f]["metadata_idx"]) > 0
        ]
        self.total_size = sum(self.files[f]["size"] for f in self.filenames)

        # initialization for the first full pass
        self.file_pointer = len(self.filenames)
        self.loaded_synths = {}
        self.metadata_idx = np.empty(0, np.int32)      # this chunk's selection
        self.full_pass_idx = -1

    def get_files(self):
        """
        The selection, grouped by the soundfont a chunk has to load to reach it.

        Every file gets an entry, including ones holding nothing this dataset
        serves, because total_size has to account for the whole corpus; the empty
        ones are dropped from filenames just below.
        """
        files = {
            filename: {"size": self.memory_usage[filename], "metadata_idx": []}
            for filename in dict.fromkeys(m["file"] for m in self.metadata)
        }
        for i in self.metadata_idx:
            files[self.metadata[i]["file"]]["metadata_idx"].append(int(i))
        return files

    def update(self):
        if self.ram_limit >= self.total_size and len(self.loaded_synths) > 0:
            return
        self._unload_current_chunk()
        if self.file_pointer == len(self.filenames):
            self._start_new_iteration()
        self._load_next_chunk()

    def _start_new_iteration(self):
        random.shuffle(self.filenames)
        self.filesizes = [self.files[filename]["size"] for filename in self.filenames]
        self.full_pass_idx += 1
        self.chunk_idx = -1
        self.file_pointer = 0
        self.visited_size = 0.0

    def _load_next_chunk(self):
        assert len(self.loaded_synths) == 0 and len(self.metadata_idx) == 0
        self.chunk_idx += 1
        cumsum_size = 0
        loaded = []
        logger.info(
            "Full Pass %s, Chunk %s: Starting to load..",
            self.full_pass_idx,
            self.chunk_idx,
        )
        while self.file_pointer < len(self.filenames):
            if cumsum_size + self.filesizes[self.file_pointer] > self.ram_limit:
                break
            cumsum_size += self.filesizes[self.file_pointer]
            filename = self.filenames[self.file_pointer]
            filepath = self.simap_dir / filename
            synths = load(filepath, self.sample_rate)
            self.loaded_synths[filename] = synths
            loaded.extend(self.files[filename]["metadata_idx"])
            self.file_pointer += 1
        # one array per chunk rather than a growing list: the selection is read
        # once per sample and never appended to after the chunk is built
        self.metadata_idx = np.array(loaded, np.int32)
        self.visited_size += cumsum_size
        logger.info(
            "Full Pass %s, Chunk %s: %s instruments loaded, %.2f/%.2f GB visited.",
            self.full_pass_idx,
            self.chunk_idx,
            len(self.metadata_idx),
            self.visited_size,
            self.total_size,
        )

    def _unload_current_chunk(self):
        for synths in self.loaded_synths.values():
            for synth in synths:
                synth.delete()
        self.loaded_synths = {}
        self.metadata_idx = np.empty(0, np.int32)
        # deleting the synths frees their memory, but glibc keeps about half of
        # it in its own arenas rather than returning it to the kernel -- where
        # the cgroup still counts it, and where a forked worker can still copy
        # it, since copy-on-write does not know a page is free. Measured over a
        # chunk swap, this puts rss back exactly where it started
        self.libc.malloc_trim(0)

    def _get_synths(self, file):
        return self.loaded_synths[file]

    def _clean_up_synths(self, synths):
        pass        # the chunk owns them until it is unloaded

    def __del__(self):
        self._unload_current_chunk()


class NaiveSimapDataset(BaseSimapDataset):
    def _get_synths(self, file):
        return load(self.simap_dir / file, self.sample_rate)

    def _clean_up_synths(self, synths):
        for synth in synths:
            synth.delete()


# https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py
def _collate_tensor_with_padding(batch, *, collate_fn_map=None):
    elem = batch[0]
    if all(e.shape == elem.shape for e in batch):
        return collate_tensor_fn(batch, collate_fn_map=collate_fn_map)
    max_len = max(e.shape[-1] for e in batch)
    out_shape = (len(batch), *elem.shape[:-1], max_len)
    if torch.utils.data.get_worker_info() is not None:
        numel = len(batch) * elem[..., :1].numel() * max_len
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(*out_shape).zero_()
    else:
        out = elem.new_zeros(*out_shape)
    for i, e in enumerate(batch):
        out[i, ..., : e.shape[-1]] = e
    return out


def collate_fn(batch):
    fn_map = {**default_collate_fn_map, torch.Tensor: _collate_tensor_with_padding}
    return collate(batch, collate_fn_map=fn_map)


def _loader_plan(dataset, weighted: bool):
    """
    How to walk one of these datasets: the sampler, and whether to shuffle.

    Uniform by default, where an epoch is a pass over the selection and every
    instrument appears once. weighted asks instead for the key count as a
    weight, which samples a note rather than an instrument -- an eighty-eight
    key instrument comes round eighty-eight times as often as a one key kit --
    and draws with replacement, so an epoch becomes a bootstrap that repeats
    some instruments and leaves out about a third.

    valid and test are walked in order, since the draw inside them is already
    seeded per instrument and a fixed order is what makes a whole pass
    reproducible. Weighting them is refused rather than honoured: a weighted
    bootstrap does not measure the split, it measures a resampling of it.
    """
    evaluating = dataset.split in ("valid", "test")
    if weighted and evaluating:
        raise ValueError(
            f"weighted sampling is for training, not the {dataset.split} split: "
            f"it would repeat some instruments and leave others out, so the "
            f"metric would not be over the split")
    if weighted:
        sampler = WeightedRandomSampler(
            dataset.get_sampler_weights(),
            num_samples=len(dataset),
            replacement=True,
        )
        return sampler, None          # a sampler and shuffle are exclusive
    return None, not evaluating


class ChunkedSimap:
    def __init__(
        self, sample_rate, ram_limit, batch_size, num_workers=0,
        weighted: bool = False, **dataset_kwargs
    ):
        self.dataset = ChunkedSimapDataset(
            sample_rate, ram_limit=ram_limit, **dataset_kwargs
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weighted = weighted
        _loader_plan(self.dataset, weighted)    # refuse a bad pairing up front

    def get_dataloader(self):
        self.dataset.update()
        sampler, shuffle = _loader_plan(self.dataset, self.weighted)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=True,
        )


class NaiveSimap:
    def __init__(self, sample_rate, batch_size, num_workers=0,
                 weighted: bool = False, **dataset_kwargs):
        self.dataset = NaiveSimapDataset(sample_rate, **dataset_kwargs)
        sampler, shuffle = _loader_plan(self.dataset, weighted)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    def get_dataloader(self):
        return self.dataloader
