"""
The keymap: every layer that sounds on every key of every instrument.

Stored as parquet with one row per layer, so reading it needs no soundfont knowledge:

    metadata_idx | key | layer | sample_idx | shift_semitones | lovel | hivel | loopmode
    sha | samplerate | channels | frame_num | loopstart | loopend | name
    gens | mods | opcodes

The second row describes the sample the layer plays, repeated on every layer that
plays it. The third says what the layer does to that sample, as json: sf2 and sf3
fill gens and mods, sfz fills opcodes, and no file fills both.

Rows are sorted by (metadata_idx, key, layer) and row groups hold whole instruments,
so one instrument's rows never straddle two groups and can be read without decoding
the rest of the file. A key with no sample assigned contributes no rows.

dataset.get_keymap() produces the per-instrument form this module serializes: arrays
of all layers concatenated in key order, plus an n_layers count per key.

loopmode:
    0: UNLOOPED
    1: LOOP_DURING_RELEASE
    2: START_ON_RELEASE
    3: LOOP_UNTIL_RELEASE
    nan: sfz (whose loopmode is not reliable)
    other values occur (58 for one instrument in simap, suspected as an author-side error)
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from simap.constants import KEY_NUM


KEYMAP_SCHEMA = pa.schema(
    [
        ("metadata_idx", pa.int32()),
        ("key", pa.uint8()),
        ("layer", pa.uint16()),  # index within the key, uint16 to match n_layers
        ("sample_idx", pa.uint16()),
        ("shift_semitones", pa.float32()),
        ("lovel", pa.uint8()),
        ("hivel", pa.uint8()),
        ("loopmode", pa.float32()),
        # the sample this layer plays. Repeated on every layer that plays it
        # rather than held in a table of its own: one row then says everything
        # about one layer, and a row group holds few enough distinct samples
        # that parquet stores each of these once and an index per row
        ("sha", pa.string()),  # null where the synth generates the sound
        ("samplerate", pa.uint32()),
        ("channels", pa.uint8()),
        ("frame_num", pa.uint32()),
        ("loopstart", pa.int32()),  # null for sfz, which states it per region
        ("loopend", pa.int32()),
        ("name", pa.string()),
        # what the layer does to the sample, as json so that two layers
        # synthesised alike share one dictionary entry. A list or map column
        # would spell out every entry on every row and cost twice as much
        ("gens", pa.string()),  # null for sfz
        ("mods", pa.string()),
        ("opcodes", pa.string()),  # the sfz vocabulary, null for sf2/3
    ]
)

# a row group holds whole instruments, so one instrument's rows never straddle two
# groups and reading a single instrument decodes exactly one of them
INSTRUMENTS_PER_ROW_GROUP = 2000


""" write """


def keymap_batch_table(keymap: list[dict], first_idx: int) -> pa.Table:
    """Flatten a run of per-instrument keymaps into one row per layer."""
    metadata_idx, key, layer = [], [], []
    for offset, km in enumerate(keymap):
        n = km["n_layers"].astype(np.int64)
        layer_num = int(n.sum())
        for field in ("shift_semitones", "sample_idx", "synth_params"):
            if layer_num != len(km[field]):
                raise ValueError(
                    f"instrument {first_idx + offset}: n_layers sums to "
                    f"{layer_num}, but {field} has {len(km[field])} entries"
                )
        starts = np.concatenate(([0], np.cumsum(n)[:-1]))
        metadata_idx.append(np.full(layer_num, first_idx + offset, dtype=np.int32))
        key.append(np.repeat(np.arange(KEY_NUM, dtype=np.uint8), n))
        # rank within the key: position minus where that key's run starts
        layer.append((np.arange(layer_num) - np.repeat(starts, n)).astype(np.uint16))

    # the sample fields spread back over the layers that play them
    per_layer = [km["samples"][i] for km in keymap for i in km["sample_idx"]]
    params = [g for km in keymap for g in km["synth_params"]]

    join = lambda arrays: np.concatenate(arrays) if arrays else np.empty(0)
    return pa.table(
        {
            "metadata_idx": join(metadata_idx),
            "key": join(key),
            "layer": join(layer),
            "sample_idx": join([km["sample_idx"] for km in keymap]),
            "shift_semitones": join([km["shift_semitones"] for km in keymap]),
            "loopmode": join([km["loopmode"] for km in keymap]),
            "lovel": join([km["lovel"] for km in keymap]),
            "hivel": join([km["hivel"] for km in keymap]),
            "sha": [r["sha"] for r in per_layer],
            "samplerate": [r["samplerate"] for r in per_layer],
            "channels": [r["channels"] for r in per_layer],
            "frame_num": [r["frame_num"] for r in per_layer],
            "loopstart": [r["loopstart"] for r in per_layer],
            "loopend": [r["loopend"] for r in per_layer],
            "name": [r["name"] for r in per_layer],
            # repr round trips a double exactly, which json.dumps uses, so the
            # values survive unchanged. They must: attenuation arrives scaled
            # by a float 0.4, and neighbouring inputs land 0.4 apart
            "gens": [None if g[0] is None else json.dumps(g[0]) for g in params],
            "mods": [None if g[1] is None else json.dumps(g[1]) for g in params],
            "opcodes": [None if g[2] is None else json.dumps(g[2]) for g in params],
        },
        schema=KEYMAP_SCHEMA,
    )


def write_keymap(keymap: list[dict], out_path: Path):
    """Write per-instrument keymaps as parquet, in the given (metadata) order."""
    starts = range(0, len(keymap), INSTRUMENTS_PER_ROW_GROUP)
    with pq.ParquetWriter(out_path, KEYMAP_SCHEMA, compression="zstd") as writer:
        for start in starts:
            batch = keymap[start : start + INSTRUMENTS_PER_ROW_GROUP]
            table = keymap_batch_table(batch, start)
            if table.num_rows > 0:
                writer.write_table(table, row_group_size=table.num_rows)
            del table
    _warn_plain_encoding(out_path)


def _warn_plain_encoding(out_path: Path):
    """
    Say so if a wide column did not get dictionary encoded.

    A dictionary lives in one row group, and parquet abandons it for that
    column chunk once it outgrows a limit, storing every value in full
    instead. Around forty layers share each distinct value here, so the
    fallback is the difference between a small file and a large one, and it is
    silent. Nothing is printed while it does not happen.
    """
    md = pq.ParquetFile(out_path).metadata
    names = md.schema.names
    for col in ("gens", "mods", "opcodes", "sha", "name"):
        if col not in names:
            continue
        i = names.index(col)
        plain = sum(
            1
            for g in range(md.num_row_groups)
            if "RLE_DICTIONARY" not in md.row_group(g).column(i).encodings
        )
        if plain:
            size = sum(
                md.row_group(g).column(i).total_compressed_size
                for g in range(md.num_row_groups)
            )
            print(
                f"  column {col} fell back to plain encoding in {plain} of "
                f"{md.num_row_groups} row groups, and takes {size / 1e6:.1f} MB"
            )


""" read """


class Keymap:
    """
    Reader for keymap.parquet.

    Two access modes, for the two things that need it: read_instrument() decodes one
    row group for a single instrument, and filter_keys_by_max_pitch_shift() makes one
    pass over the file to derive every instrument's qualified keys at once.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = pq.ParquetFile(self.path)
        # (first, last) metadata_idx per row group, from the footer statistics
        self.ranges = [
            (
                self.file.metadata.row_group(rg).column(0).statistics.min,
                self.file.metadata.row_group(rg).column(0).statistics.max,
            )
            for rg in range(self.file.metadata.num_row_groups)
        ]

    def read_instrument(self, metadata_idx: int) -> dict:
        """The layers of one instrument, as arrays ordered by (key, layer)."""
        for rg, (first, last) in enumerate(self.ranges):
            if first <= metadata_idx <= last:
                table = self.file.read_row_group(rg)
                selected = table.column("metadata_idx").to_numpy() == metadata_idx
                return {
                    name: table.column(name).to_numpy()[selected]
                    for name in table.column_names
                    if name != "metadata_idx"
                }
        raise IndexError(f"metadata_idx {metadata_idx} is not in {self.path}")

    def max_abs_shift(self) -> np.ndarray:
        """
        (instrument_num, KEY_NUM) of the largest |shift_semitones| on each key, and
        nan where no sample is assigned to the key.
        """
        columns = ["metadata_idx", "key", "shift_semitones"]
        table = pq.read_table(self.path, columns=columns)
        metadata_idx = table.column("metadata_idx").to_numpy()
        key = table.column("key").to_numpy()
        shift = np.abs(table.column("shift_semitones").to_numpy())

        if len(metadata_idx) == 0:
            return np.empty((0, KEY_NUM), dtype=np.float32)

        instrument_num = int(metadata_idx.max()) + 1
        out = np.full(instrument_num * KEY_NUM, np.nan, dtype=np.float32)

        # rows are written sorted by (metadata_idx, key, layer), so the layers of a
        # key are contiguous and reduceat can take their max in one vectorised pass
        flat = metadata_idx.astype(np.int64) * KEY_NUM + key
        if np.any(np.diff(flat) < 0):
            raise ValueError(f"{self.path} is not sorted by (metadata_idx, key)")
        starts = np.flatnonzero(np.concatenate(([True], flat[1:] != flat[:-1])))
        out[flat[starts]] = np.maximum.reduceat(shift, starts)
        return out.reshape(instrument_num, KEY_NUM)

    def is_sustained(self) -> np.ndarray:
        """
        (instrument_num, KEY_NUM) of whether each key holds a tone while the note is
        held, so that letting go of it is an event rather than nothing.

        A key sustains if any of its layers loops, 1 or 3. Layers at a key almost
        always sound together rather than alternating by velocity -- of the 1.1 per
        cent of keys whose layers disagree, 99 per cent overlap in velocity -- and
        the usual disagreement is an unlooped attack over a looped body, which does
        sustain. So this is an any() over the key's layers, not a vote.

        The rest are 0, for three different reasons:

            0   unlooped, so the sample simply runs out, however long the note
            2   FLUID_START_ON_RELEASE, which is silent until the key is let go,
                so neither its onset nor its sustain exists. A Polyphone extension
                rather than anything in the sf2 spec, and 128 keys of this corpus
            nan sfz, whose loopmode the loader does not trust. Read as not
                sustained on purpose: some real sustains are lost, but nothing is
                wrongly called one, and sfz is 465 instruments
            58  out of range, an author-side error, 128 keys

        An unmapped key is also 0, since there is nothing there to sustain. Where
        that distinction matters, max_abs_shift() is nan on exactly those keys.
        """
        columns = ["metadata_idx", "key", "loopmode"]
        table = pq.read_table(self.path, columns=columns)
        metadata_idx = table.column("metadata_idx").to_numpy()
        key = table.column("key").to_numpy()
        # nan compares false, which is the conservative reading sfz wants anyway
        loops = np.isin(table.column("loopmode").to_numpy(), (1.0, 3.0))

        if len(metadata_idx) == 0:
            return np.zeros((0, KEY_NUM), dtype=bool)

        instrument_num = int(metadata_idx.max()) + 1
        out = np.zeros(instrument_num * KEY_NUM, dtype=bool)

        # as in max_abs_shift: the rows of a key are contiguous, so one reduceat
        # answers every key at once
        flat = metadata_idx.astype(np.int64) * KEY_NUM + key
        if np.any(np.diff(flat) < 0):
            raise ValueError(f"{self.path} is not sorted by (metadata_idx, key)")
        starts = np.flatnonzero(np.concatenate(([True], flat[1:] != flat[:-1])))
        out[flat[starts]] = np.logical_or.reduceat(loops, starts)
        return out.reshape(instrument_num, KEY_NUM)

    def filter_keys_by_max_pitch_shift(
        self,
        max_pitch_shift_semitones: float = 3,
    ) -> list[list[int]]:
        """
        One list per instrument of the keys that qualify: a key qualifies when it
        has a sample and none of them is pitch shifted further than the given maximum.
        """
        shift = self.max_abs_shift()
        return [
            np.flatnonzero(row <= max_pitch_shift_semitones).tolist() for row in shift
        ]
