"""
Split the corpus into train, valid and test, 8:1:1.

    <out>   three keys, "train", "valid" and "test", each a sorted list of
            metadata_idx joining to metadata.parquet by row. Every instrument
            appears in exactly one of them

Held out instruments are drawn only from the ones that share no recording with
any other instrument, so no sample heard in valid or test is heard in train.
That is a property of the corpus rather than of the draw: an instrument either
owns its audio outright or it does not, and only the first kind is eligible.
Instrument level deduplication is what makes the pool large enough -- half the
corpus owns its audio outright, where before deduplication it was under a
third, because a copy and its original each counted as sharing with the other.

What this costs
---------------
An instrument is eligible precisely because nobody else reused its samples,
which is not a neutral property: the widely shared grand piano can never be
held out, and the obscure one always can. So the held out sets lean towards the
tail of the corpus and are not a random sample of it. Of the 128 hand picked
world instruments, for instance, 44 are eligible and 84 are not. Nothing here
can fix that without giving up the leakage guarantee, which is the more
valuable of the two.

An instrument with no audio at all is eligible on a technicality, sharing
nothing because it has nothing. It is dropped from the pool rather than held
out, since it can only ever be a dead entry in an evaluation set.

Reproducibility
---------------
The draw is a function of the seed and the corpus. metadata_idx is a row number
and is not stable across corpus rebuilds, so this file is pinned to the
metadata it was built against: rebuilding the corpus requires rebuilding this,
and the two must travel together.

Usage:
    python scripts/subset/build_split.py \\
        --out <simap_dir>/split.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc

from simap.user_config import get_user_config


def owns_its_audio(keymap: Path, n_inst: int):
    """The instruments no other instrument shares a recording with."""
    t = pq.read_table(keymap, columns=["metadata_idx", "sha"])
    inst = t.column("metadata_idx").to_numpy()
    sha = pc.dictionary_encode(t.column("sha").combine_chunks())
    code = np.nan_to_num(sha.indices.to_numpy(zero_copy_only=False),
                         nan=-1).astype(np.int64)
    pairs = np.unique(np.stack([inst, code]), axis=1)
    inst, code = pairs[0], pairs[1]
    audible = code >= 0            # a layer naming no audio says nothing here
    inst, code = inst[audible], code[audible]
    users = np.bincount(code)
    shared = np.unique(inst[users[code] > 1])
    # an instrument with no audible layer at all is not eligible: it shares
    # nothing only because it has nothing
    return np.setdiff1d(np.unique(inst), shared), inst, code


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="default: the simap_dir from simap set-dir")
    p.add_argument("--out", type=Path, required=True, help="the json to write")
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    data_dir = a.data_dir or get_user_config().get("simap_dir")
    if data_dir is None:
        raise SystemExit("pass --data-dir, or run "
                         "`simap set-dir dataset <dir>` to set it once")
    data_dir = Path(data_dir)

    n_inst = len(json.loads((data_dir / "metadata.json").read_text()))
    pool, inst, code = owns_its_audio(data_dir / "keymap.parquet", n_inst)
    n_valid, n_test = round(n_inst * a.valid_frac), round(n_inst * a.test_frac)
    print(f"{n_inst:,} instruments, {len(pool):,} own their audio "
          f"({len(pool) / n_inst:.1%})", file=sys.stderr)
    if n_valid + n_test > len(pool):
        raise SystemExit(f"need {n_valid + n_test:,} held out instruments but "
                         f"only {len(pool):,} are eligible")

    # sorted, so the seed alone decides the draw
    drawn = np.random.default_rng(a.seed).choice(
        np.sort(pool), size=n_valid + n_test, replace=False)
    valid, test = np.sort(drawn[:n_valid]), np.sort(drawn[n_valid:])
    held = np.concatenate([valid, test])
    train = np.setdiff1d(np.arange(n_inst), held)

    # the guarantee this is all for, checked rather than assumed
    train_shas = set(code[np.isin(inst, train)].tolist())
    held_shas = set(code[np.isin(inst, held)].tolist())
    leaked = train_shas & held_shas
    if leaked:
        raise SystemExit(f"{len(leaked):,} recordings appear in both train and "
                         f"a held out split")

    split = {"train": [int(i) for i in train],
             "valid": [int(i) for i in valid],
             "test": [int(i) for i in test]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(split))
    for k, v in split.items():
        print(f"  {k:<6} {len(v):>7,}  {len(v) / n_inst:>6.1%}")
    print(f"  no recording is shared between train and the held out splits")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
