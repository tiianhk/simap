"""
Collect the retrieved GM banks and drum kits into one label map.

    <out>   129 keys: "1".."128" for the gm programs, and "percussion" for the
            drum kits. Each value is a sorted list of metadata_idx, joining to
            metadata.parquet by row.

The two selections are read from the directories gm_inst.py and gm_perc.py wrote,
not recomputed from the embeddings. Those directories are what was listened to,
and they may have been curated by hand afterwards -- folders deleted, thresholds
reconsidered -- so rebuilding the selection from scores would quietly discard that.
Reading them back also means this needs no embedding store and runs in seconds.

    melodic     every <inst-dir>/*/presets.csv, which names the gm_program and
                metadata_idx of each preset outright
    percussion  every <perc-dir>/<rank>_s<frac>_<metadata_idx>_<label>/, whose
                third underscore-separated field is the index. The label may
                contain underscores of its own, which is why the field is taken
                by position from the left rather than by splitting the whole name

A bank contributes one instrument to each of the 128 programs, while a kit
contributes one instrument to "percussion". So the percussion list is far longer
than any single program's, and the file is lopsided by construction rather than by
accident.

Usage:
    python scripts/subset/build_gm_subset.py \\
        --inst-dir <out_dir>/gm_inst \\
        --perc-dir <out_dir>/gm_perc \\
        --out <simap_dir>/general_midi_subset.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from simap.user_config import get_user_config

GM_N = 128
PERC_KEY = "percussion"


def item_dirs(root: Path) -> list[Path]:
    """The per-item folders of an output directory, in rank order."""
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not dirs:
        raise SystemExit(f"{root} holds no item folders")
    return dirs


def read_inst(inst_dir: Path) -> tuple[dict[int, list[int]], int]:
    """gm program to metadata_idx, from every bank's presets.csv."""
    by_program = defaultdict(list)
    folders = item_dirs(inst_dir)
    for folder in folders:
        csv_path = folder / "presets.csv"
        if not csv_path.is_file():
            raise SystemExit(f"{folder} has no presets.csv")
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                by_program[int(row["gm_program"])].append(int(row["metadata_idx"]))
    return by_program, len(folders)


def read_perc(perc_dir: Path) -> list[int]:
    """metadata_idx per kit, from the folder names."""
    idx = []
    for folder in item_dirs(perc_dir):
        field = folder.name.split("_")[2]
        if not field.isdigit():
            raise SystemExit(f"{folder.name} does not carry a metadata_idx")
        idx.append(int(field))
    return idx


def write_json(subset: dict, path: Path):
    """
    One key per line, so the file reads as 129 rows.

    Written out by hand because json.dump cannot do this: its indent is all or
    nothing, either the whole object on one line or every list element on a line
    of its own. What comes out is ordinary json, only the whitespace is ours.
    """
    items = list(subset.items())
    with open(path, "w") as f:
        f.write("{\n")
        for n, (key, value) in enumerate(items, 1):
            end = "," if n < len(items) else ""
            f.write(f" {json.dumps(key)}: {json.dumps(value)}{end}\n")
        f.write("}\n")


def validate(subset: dict, data_dir: Path) -> dict:
    """
    Check the map against the metadata before writing it.

    The decisive test is positional: an instrument filed under gm program p must
    sit at preset p-1, which is what catches a misparse or an off-by-one rather
    than letting it through as a plausible looking list.
    """
    meta = pq.read_table(data_dir / "metadata.parquet", columns=["bank", "preset"])
    bank = meta.column("bank").to_pandas().to_numpy()
    preset = meta.column("preset").to_pandas().to_numpy()
    n_rows = len(preset)

    everything = [i for v in subset.values() for i in v]
    if everything and (min(everything) < 0 or max(everything) >= n_rows):
        raise SystemExit(f"a metadata_idx falls outside the {n_rows:,} rows of "
                         f"{data_dir / 'metadata.parquet'}")
    if len(set(everything)) != len(everything):
        raise SystemExit("an instrument appears under more than one key")

    for program in range(1, GM_N + 1):
        rows = subset[str(program)]
        if not rows:
            continue
        wrong = [i for i in rows if preset[i] != program - 1]
        if wrong:
            raise SystemExit(
                f"gm program {program} should hold preset {program - 1}, but "
                f"{len(wrong)} of its {len(rows)} instruments do not: {wrong[:5]}")

    perc_banks = bank[subset[PERC_KEY]] if subset[PERC_KEY] else np.empty(0)
    counts = {}
    for value in np.unique(perc_banks[~np.isnan(perc_banks)]):
        counts[int(value)] = int((perc_banks == value).sum())
    return {
        "n_instruments": len(everything),
        "n_melodic": len(everything) - len(subset[PERC_KEY]),
        "n_percussion": len(subset[PERC_KEY]),
        "percussion_by_bank": dict(sorted(counts.items(),
                                          key=lambda kv: -kv[1])),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inst-dir", type=Path, required=True,
                   help="an output directory written by gm_inst.py")
    p.add_argument("--perc-dir", type=Path, required=True,
                   help="an output directory written by gm_perc.py")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="holds metadata.parquet, which the map is checked "
                        "against; default: the simap_dir from simap set-dir")
    p.add_argument("--out", type=Path, required=True, help="the json to write")
    args = p.parse_args()

    data_dir = args.data_dir or get_user_config().get("simap_dir")
    if data_dir is None:
        raise SystemExit("pass --data-dir, or run "
                         "`simap set-dir dataset <dir>` to set it once")
    data_dir = Path(data_dir)

    by_program, n_banks = read_inst(args.inst_dir)
    perc_idx = read_perc(args.perc_dir)
    print(f"{n_banks} banks -> {sum(len(v) for v in by_program.values()):,} "
          f"melodic instruments | {len(perc_idx):,} kits", file=sys.stderr)

    # every program is present even if nothing was retrieved for it, so the file
    # always has the same 129 keys and a consumer never has to test for one
    subset = {str(program): sorted(by_program.get(program, []))
              for program in range(1, GM_N + 1)}
    subset[PERC_KEY] = sorted(perc_idx)

    stats = validate(subset, data_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(subset, args.out)
    sizes = [len(v) for k, v in subset.items() if k != PERC_KEY]

    print(f"{stats['n_instruments']:,} instruments over {len(subset)} keys "
          f"({stats['n_melodic']:,} melodic, {stats['n_percussion']:,} percussion)")
    print(f"  per program: {min(sizes)}-{max(sizes)}")
    print(f"  percussion by bank: {stats['percussion_by_bank']}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
