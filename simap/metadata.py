"""
The metadata (parquet): one entry per instrument, the same rows and fields as metadata.json.
The json stays the human-readable, greppable form. This parquet is the one that loads fast.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROW_GROUP_SIZE = 8192
DICTIONARY_COLUMNS = ("url", "author", "license", "file", "text_label")
INT_COLUMNS = ("synth_id", "channel_id", "bank", "preset")


""" write """


def write_metadata_parquet(metadata: list[dict], out_path: Path):
    """Write metadata as parquet: same rows, same order, same fields as the json."""
    if len(metadata) == 0:
        raise ValueError("metadata is empty")

    fields = list(metadata[0])  # keep the field order of the entries themselves
    known = set(DICTIONARY_COLUMNS) | set(INT_COLUMNS)
    if set(fields) != known:
        raise ValueError(
            f"metadata fields {sorted(set(fields) ^ known)} are not covered by the "
            f"parquet schema; update DICTIONARY_COLUMNS / INT_COLUMNS"
        )

    columns = {}
    for name in fields:
        values = [m[name] for m in metadata]
        if name in INT_COLUMNS:
            columns[name] = pa.array(values, type=pa.int16())
        else:
            columns[name] = pa.array(values, type=pa.string()).dictionary_encode()

    pq.write_table(
        pa.table(columns),
        out_path,
        row_group_size=ROW_GROUP_SIZE,
        compression="zstd",
    )


""" read """


class Metadata:
    """
    Reader for metadata.parquet.

    Indexed like the json list it replaces, but an entry is built only when asked
    for, and only its row group is decoded. text_labels() returns the label column
    alone, so a search never has to build an entry per instrument.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = pq.ParquetFile(self.path)
        row_groups = [
            self.file.metadata.row_group(rg).num_rows
            for rg in range(self.file.metadata.num_row_groups)
        ]
        # first row index of each row group, so an entry index locates its group
        self.first_row = [0]
        for num_rows in row_groups:
            self.first_row.append(self.first_row[-1] + num_rows)

    def __len__(self) -> int:
        return self.file.metadata.num_rows

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} entries")
        for rg in range(len(self.first_row) - 1):
            if index < self.first_row[rg + 1]:
                table = self.file.read_row_group(rg)
                offset = index - self.first_row[rg]
                return {
                    name: table.column(name)[offset].as_py()
                    for name in table.column_names
                }
        raise IndexError(index)  # unreachable while first_row covers every row

    def text_labels(self) -> list[str]:
        return pq.read_table(self.path, columns=["text_label"]).column(0).to_pylist()


def apply_label_map(labels: list[str], map_path: Path) -> list[str]:
    """
    The labels as some map rewrites them: a filter, a tidier form, a translation.

    The author's own label is what SIMAP ships and is the only text it asserts.
    Everything else -- sonnet5_text_filtering.json among them -- is an add-on, one
    model's reading of those labels, so it is applied on request rather than baked
    into the metadata. An empty value means the map rejected the label; what to do
    about that is the caller's business.

    A map must cover every label it is given. A partial one would quietly empty the
    labels it had no opinion about, which would look like filtering rather than
    like absence, so it raises instead.
    """
    mapping = json.loads(Path(map_path).read_text())
    missing = {label for label in labels if label not in mapping}
    if missing:
        raise ValueError(
            f"{map_path} covers {len(mapping):,} labels but {len(missing):,} of the "
            f"{len(set(labels)):,} given are absent, e.g. {sorted(missing)[:5]}. A "
            f"map has to be complete: the labels it missed would come back empty "
            f"and be indistinguishable from ones it rejected."
        )
    return [mapping[label] for label in labels]
