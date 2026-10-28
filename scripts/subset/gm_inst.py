"""
Score every bank on whether it follows the General MIDI sound set, under both
checkpoints at once. Bank retrieval: the unit ranked here is a bank, not an
instrument.

    <out>/rank_curve.png                        summed score against rank
    <out>/<rank>_s<frac>_<file>_bank<bank>/
        preset_scores.png                       score against preset
        presets.csv                             every preset, with its metadata_idx
        preset<gm>_<gm name>_key<k>.wav         one note, drawn at random

GM fixes what each program of a bank must be: program 1 is an acoustic grand
piano, 57 a trumpet, 128 a gunshot. So for a GM bank the specification is the
ground truth and no author label is needed. The score is

    sum over p in 1..128 of  cos( instrument(preset p-1) , text(gm name for p) )

and nothing else. It is positional on purpose: the question is whether preset 56
is a trumpet, not whether a trumpet is somewhere in the bank.

Numbering
---------
GM programs are 1 to 128 and SF2 presets are 0 to 127, so gm_sound_set.json's "1"
is the instrument at preset 0. The wav files and presets.csv are labelled with the
GM program, because that is what pairs with the name; the sf_preset column of
presets.csv carries the raw value that joins back to metadata.parquet.

What a bank is
--------------
A bank is one (file, bank) pair, and its presets are the instruments inside it.
No (file, bank, preset) triple occurs twice in the corpus, so the grouping is
unambiguous. Files in sfz format have no bank or preset at all and cannot take
part; they are dropped rather than scored as empty banks.

Summed over presets rather than averaged, so a higher score needs both more of the
sound set covered and a better match on each program. A missing preset contributes
nothing, which is what makes a one-preset bank -- 5,134 of the 11,537 in this
corpus -- sink on its own with no minimum-size filter. Presets outside 0..127 are
outside GM and contribute nothing either.

Every bank is scored, not just bank 0. Bank 0 is where a GM set belongs, not where
they are, and the rest supplies the negative control: a percussion bank or a
variation bank holding eight organs must score low, so where the rank curve falls
away is where GM banks stop and everything else begins.

Instrument embeddings, not keys
-------------------------------
Unlike the percussion map, which asks a positional question of the keys inside one
instrument, this asks a positional question of the instruments inside one bank. So
it reads the collapsed instrument_embedding.npz rather than the per-key store, and
takes the plain dot product against it: that file holds an unnormalised mean of
unit-length key vectors, and dot(q, mean(v_i)) == mean(dot(q, v_i)), so the score
is already the average per-key cosine. Dividing by mean_norm would turn it into
something else and would reward the instruments whose keys agree least.

One consequence is that the renders cannot reproduce what was scored, as they do
for percussion, where key k rendered is key k scored. A preset's score is a mean
over its 88 mapped keys, so any render is a sample of it.

Given that, the sample is drawn at random rather than fixed: one key from the ones
that preset actually has a sample on, and one length from NOTE_DUR_RANGE, with a
further RELEASE_TAIL second of silence after the note off. A fixed middle C would
make a bank of 128 presets 128 renders of the same pitch, which says nothing about
the registers a listener has not heard and flatters the instruments that were
sampled near it; drawing instead covers the keyboard across the bank. The varying
length is what exposes loop and release behaviour, which a constant 2 s hides. The
draw is seeded, so a rerun writes the same audio.

This is a further departure from the embedding conditions, which were a fixed
velocity 100, a 2 s note and a 3 s window. What is heard here is the instrument,
not the thing that was scored.

Two checkpoints
---------------
Both stores are found under --embedding-dir, and each is scored against the text
vectors built from its own checkpoint, so the pairing cannot be crossed: the same
string embeds to nearly orthogonal vectors under the two, and inst(fusion) has to
meet text(fusion). The two scores are then added, and the sum is the only thing
that ranks, thresholds and names a folder. --fusion-only and --music-only score
with one checkpoint instead of both.

Usage:
    python scripts/subset/gm_inst.py --out <out_dir>/gm_inst

--embedding-dir and --data-dir fall back to what `simap set-dir` recorded, so
neither is normally given. No checkpoint is needed: every vector this reads was
computed already.
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from simap.embedding_store import selected_stores
from simap.synth_interface import load
from simap.user_config import get_user_config

GM_N = 128                      # gm programs 1..128, sf2 presets 0..127
LOW_KEY, HIGH_KEY = 21, 108     # the range the embeddings were rendered over
SAMPLE_RATE = 48000
VELOCITY = 100
NOTE_DUR_RANGE = (0.5, 3.0)     # the note is held this long, drawn per preset
RELEASE_TAIL = 1.0              # and the window runs this much longer


def read_sound_set(path: Path) -> dict[int, str]:
    """The gm sound set, program number to official name, over 1..128."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    mapping = {int(program): str(name) for program, name in loaded.items()}
    missing = [n for n in range(1, GM_N + 1) if n not in mapping]
    if missing:
        raise ValueError(f"{path} has no gm name for programs {missing}")
    return mapping


def target_matrix(mapping: dict[int, str], text_path: Path) -> np.ndarray:
    """
    One row per sf2 preset, holding the text vector that preset should match.

    Indexed by preset rather than by gm program, so it can be gathered directly
    with the preset column of the metadata and the off-by-one is applied once,
    here, instead of at every use.
    """
    z = np.load(text_path)
    queries = [str(q) for q in z["queries"]]
    vectors = z["embedding"].astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    T = np.zeros((GM_N, vectors.shape[1]), np.float32)
    for program, name in mapping.items():
        if name not in queries:
            raise SystemExit(f"{text_path} has no vector for {name!r}")
        T[program - 1] = vectors[queries.index(name)]
    return T


def score_store(inst_path: Path, T: np.ndarray, preset_of: np.ndarray):
    """
    Each instrument against the gm name its own preset claims.

    One dot product per instrument rather than a full matrix, because the preset
    picks the target: there is no question of which name an instrument is scored
    against. Rows whose preset is outside 0..127 score zero and are reported as
    absent, so they cannot contribute to a bank.
    """
    z = np.load(inst_path)
    idx = z["metadata_idx"]
    emb = z["embedding"].astype(np.float32)
    preset = preset_of[idx]
    inside = preset >= 0
    scores = np.zeros(len(idx), np.float32)
    scores[inside] = np.einsum("ij,ij->i", emb[inside], T[preset[inside]])
    return idx, scores, inside


def mapped_keys(keymap_path: Path, wanted: np.ndarray) -> dict[int, np.ndarray]:
    """
    The keys in LOW_KEY..HIGH_KEY that each wanted instrument has a sample on.

    One filtered pass over the two columns that matter, rather than a
    Keymap.read_instrument() call per instrument: that decodes a whole row group
    each time, and the presets of one bank sit in the same few row groups, so the
    same group would be decompressed once per preset.
    """
    want = np.zeros(int(wanted.max()) + 1, bool)
    want[wanted] = True
    parts = []
    for batch in pq.ParquetFile(keymap_path).iter_batches(
            columns=["metadata_idx", "key"], batch_size=1 << 20):
        idx = batch.column("metadata_idx").to_numpy()
        key = batch.column("key").to_numpy().astype(np.int32)
        keep = idx < len(want)
        keep[keep] = want[idx[keep]]
        keep &= (key >= LOW_KEY) & (key <= HIGH_KEY)
        if keep.any():
            parts.append(np.stack([idx[keep].astype(np.int32), key[keep]], axis=1))
    if not parts:
        return {}
    # unique sorts by instrument then key, so one split gives the groups
    pairs = np.unique(np.concatenate(parts), axis=0)
    bounds = np.flatnonzero(np.diff(pairs[:, 0])) + 1
    return {int(g[0, 0]): g[:, 1] for g in np.split(pairs, bounds)}


def plot_presets(path: Path, title: str, components, combined, totals, mapping):
    """
    Score against preset: the sum that did the ranking, over the checkpoints it
    is made of. The x axis is the gm program, 1 to 128.
    """
    programs = np.arange(1, GM_N + 1)
    fig, ax = plt.subplots(figsize=(18, 5.6))
    for tag, scores in components.items():
        ax.plot(programs, scores, "-", lw=0.7, alpha=0.55,
                label=f"{tag}   sum {totals[tag]:.2f}")
    ax.plot(programs, combined, "-o", ms=2.5, lw=1.3, color="#1f4e79",
            label=f"both   sum {totals['both']:.2f}")
    ax.axhline(0, color="grey", lw=0.7, ls=":")
    mean = np.nanmean(combined)
    ax.axhline(mean, color="#c00000", lw=0.9, ls="--")
    ax.text(programs[-1], mean, f" mean {mean:.3f} ", ha="right", va="bottom",
            fontsize=8, color="#c00000")
    ax.set_xlim(0, GM_N + 1)
    ax.set_xticks(programs)
    ax.set_xticklabels([f"{p} {mapping[p]}" for p in programs],
                       rotation=90, fontsize=4.5)
    ax.set_xlabel("gm program, and the name it was scored against")
    ax.set_ylabel("cosine")
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def safe(text: str) -> str:
    return re.sub(r"[^\w\-. ]", "_", str(text)).strip().rstrip(". ") or "unnamed"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-dir", type=Path, default=None,
                   help="default: the embedding_dir from simap set-dir")
    p.add_argument("--fusion-only", action="store_true",
                   help="score with the fusion checkpoint alone")
    p.add_argument("--music-only", action="store_true",
                   help="score with the music checkpoint alone")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="default: the simap_dir from simap set-dir")
    p.add_argument("--sound-set", type=Path, default=None,
                   help="gm program number to official name, over 1..128; "
                        "default: gm_sound_set.json in the data dir")
    p.add_argument("--out", type=Path, required=True, help="without a suffix")
    p.add_argument("--near-max", type=float, default=0.9,
                   help="keep a bank scoring at least this fraction of the best "
                        "summed score")
    p.add_argument("--dedup-tol", type=float, default=0.0,
                   help="two banks are the same bank when no preset's score "
                        "differs by more than this in either store; off by "
                        "default, since the corpus is deduplicated at the "
                        "instrument level and this only adds near misses")
    p.add_argument("--top-k", type=int, default=None,
                   help="an extra hard cap after the threshold and the dedup")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the key and length drawn for every preset, so a "
                        "rerun writes the same audio")
    p.add_argument("--curve-only", action="store_true",
                   help="write rank_curve.png and nothing else, to choose "
                        "--near-max before paying for the renders")
    args = p.parse_args()

    config = get_user_config()
    embedding_dir = args.embedding_dir or config.get("embedding_dir")
    if embedding_dir is None:
        raise SystemExit("pass --embedding-dir, or run "
                         "`simap set-dir embedding <dir>` to set it once")
    data_dir = args.data_dir or config.get("simap_dir")
    if data_dir is None:
        raise SystemExit("pass --data-dir, or run "
                         "`simap set-dir dataset <dir>` to set it once")
    data_dir = Path(data_dir)
    sound_set = args.sound_set or data_dir / "gm_sound_set.json"

    mapping = read_sound_set(sound_set)
    print(f"{len(set(mapping.values()))} names over {len(mapping)} programs",
          file=sys.stderr)

    meta = pq.read_table(data_dir / "metadata.parquet",
                         columns=["file", "synth_id", "channel_id", "bank",
                                  "preset", "text_label"])
    files = meta.column("file").to_pylist()
    synth_ids = meta.column("synth_id").to_numpy()
    channel_ids = meta.column("channel_id").to_numpy()
    labels = meta.column("text_label").to_pylist()
    bank_col = meta.column("bank").to_pandas().to_numpy()
    preset_col = meta.column("preset").to_pandas().to_numpy()

    # -1 marks a row that cannot belong to a gm bank: sfz, which has no bank or
    # preset, and any preset outside the 128 programs gm defines
    in_gm = ~np.isnan(preset_col) & (preset_col >= 0) & (preset_col <= GM_N - 1)
    preset_of = np.where(in_gm, np.nan_to_num(preset_col, nan=-1), -1).astype(np.int32)

    stores = {}
    for store in selected_stores(embedding_dir, args.fusion_only, args.music_only):
        print(f"{store.tag}: scoring {store.inst_npz} "
              f"({store.info['checkpoint']})", file=sys.stderr)
        stores[store.tag] = score_store(
            store.inst_npz, target_matrix(mapping, store.text_npz(sound_set)),
            preset_of)
    tags = list(stores)

    metadata_idx, _, inside = stores[tags[0]]
    for tag in tags[1:]:
        if not np.array_equal(stores[tag][0], metadata_idx):
            raise SystemExit(f"{tag} covers different instruments from {tags[0]}")

    # one row per bank, one column per preset. A bank is (file, bank); the rows of
    # the instrument store are scattered into it by their own preset, and a preset
    # the bank does not define stays zero and contributes nothing
    rows_in = np.flatnonzero(inside)
    keys = [(files[int(metadata_idx[r])], int(bank_col[metadata_idx[r]]))
            for r in rows_in]
    bank_keys = list(dict.fromkeys(keys))
    bank_num = {k: i for i, k in enumerate(bank_keys)}
    which = np.array([bank_num[k] for k in keys], np.int32)
    cols = preset_of[metadata_idx[rows_in]]

    per_preset = {}
    for tag in tags:
        m = np.zeros((len(bank_keys), GM_N), np.float32)
        m[which, cols] = stores[tag][1][rows_in]
        per_preset[tag] = m
    present = np.zeros((len(bank_keys), GM_N), bool)
    present[which, cols] = True
    idx_of = np.full((len(bank_keys), GM_N), -1, np.int32)
    idx_of[which, cols] = metadata_idx[rows_in]
    print(f"{len(bank_keys):,} banks over {len(rows_in):,} instruments "
          f"({present.sum(axis=1).max()} presets at most)", file=sys.stderr)

    total = {t: per_preset[t].sum(axis=1) for t in tags}
    combined_preset = sum(per_preset[t] for t in tags)
    combined = sum(total[t] for t in tags)
    for t in tags:
        print(f"{t}: max {total[t].max():.2f}, median {np.median(total[t]):.2f}",
              file=sys.stderr)
    frac = combined / combined.max()
    order = np.argsort(-combined)
    rank = np.empty(len(order), np.int32)
    rank[order] = np.arange(1, len(order) + 1)

    rows = np.flatnonzero(frac >= args.near_max)
    rows = rows[np.argsort(rank[rows])]
    print(f"both: max {combined.max():.2f}, median {np.median(combined):.2f}, "
          f"keeps {len(rows):,} at {args.near_max} x max", file=sys.stderr)

    # one GM bank is re-uploaded, renamed and re-bundled endlessly, so the same
    # 128 presets recur across unrelated files. The per-preset score profile is
    # the fingerprint that finds them; the checkpoints are compared side by side
    # rather than added, because a chance collision in one projection is very
    # unlikely to be repeated in the other's.
    #
    # Off by default now that the corpus is deduplicated at the instrument
    # level, which removes the same copies exactly rather than by proximity.
    # What is left to it is the near miss: a bank rebuilt from resampled or
    # re-encoded audio, whose presets score alike but are not the same bytes
    if args.dedup_tol > 0:
        profile = np.hstack([per_preset[t][rows] for t in tags])
        keep, reps = [], np.empty((0, profile.shape[1]), np.float32)
        for row, prof in enumerate(profile):
            if len(reps) and (np.abs(reps - prof).max(axis=1) < args.dedup_tol).any():
                continue
            keep.append(row)
            reps = np.vstack([reps, prof[None, :]])
        print(f"  dedup at {args.dedup_tol}: {len(keep):,} distinct banks",
              file=sys.stderr)
        rows = rows[keep]
    if args.top_k:
        rows = rows[:args.top_k]

    args.out.mkdir(parents=True, exist_ok=True)
    by_file = defaultdict(list)
    keys_of, rng = {}, np.random.default_rng(args.seed)
    if args.curve_only:
        # the curve is a property of the scores, so nothing below is needed for it
        print("--curve-only: rank_curve.png and nothing else", file=sys.stderr)
    else:
        for i in rows:
            by_file[bank_keys[int(i)][0]].append(int(i))
        # which keys each preset actually has a sample on, so the draw cannot land
        # on a silent one. Only the selected banks are looked up
        need = np.unique(idx_of[rows][present[rows]])
        keys_of = mapped_keys(data_dir / "keymap.parquet", need)
        print(f"rendering {len(rows)} banks across {len(by_file)} files, "
              f"{len(need):,} presets", file=sys.stderr)

    start, done, failures = time.monotonic(), 0, []
    for path, items in sorted(by_file.items()):
        try:
            synths = load(data_dir / path, SAMPLE_RATE)
            if synths is None:
                raise RuntimeError("load returned None")
        except Exception as e:
            failures.append((path, f"{type(e).__name__}: {e}"))
            continue
        for i in items:
            bank = bank_keys[i][1]
            name = (f"{rank[i]:04d}_s{frac[i]:.2f}_"
                    f"{safe(path)[:48]}_bank{bank}")
            sub = args.out / name
            sub.mkdir(parents=True, exist_ok=True)
            here = present[i]
            plot_presets(
                sub / "preset_scores.png", f"{path}  bank {bank}",
                {t: np.where(here, per_preset[t][i], np.nan) for t in tags},
                np.where(here, combined_preset[i], np.nan),
                {**{t: float(total[t][i]) for t in tags},
                 "both": float(combined[i])}, mapping,
            )
            # drawn before the csv is written, so the csv records what was heard
            draw = {}
            for preset in np.flatnonzero(here):
                keys = keys_of.get(int(idx_of[i, preset]))
                if keys is None or not len(keys):
                    continue      # no sample inside 21..108, so nothing to render
                draw[int(preset)] = (int(rng.choice(keys)),
                                     float(rng.uniform(*NOTE_DUR_RANGE)))
            with open(sub / "presets.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["gm_program", "gm_name", "sf_preset", "metadata_idx",
                            "author_label", "render_key", "note_dur"]
                           + list(tags) + ["both"])
                for preset in np.flatnonzero(here):
                    j = int(idx_of[i, preset])
                    key, dur = draw.get(int(preset), (None, None))
                    w.writerow(
                        [preset + 1, mapping[preset + 1], preset, j, labels[j],
                         "" if key is None else key,
                         "" if dur is None else f"{dur:.2f}"]
                        + [f"{per_preset[t][i, preset]:.4f}" for t in tags]
                        + [f"{combined_preset[i, preset]:.4f}"])
            for preset, (key, dur) in draw.items():
                j = int(idx_of[i, preset])
                try:
                    audio = synths[int(synth_ids[j])].render_note(
                        int(channel_ids[j]), key, VELOCITY, dur, dur + RELEASE_TAIL)
                except Exception as e:
                    failures.append((f"{path}[{j}] key {key}", str(e)))
                    continue
                mono = audio if audio.ndim == 1 else audio.mean(axis=0)
                sf.write(sub / f"preset{preset + 1:03d}_"
                               f"{safe(mapping[preset + 1])}_key{key}.wav",
                         mono, SAMPLE_RATE, subtype="PCM_16")
            done += 1
            print(f"  {done}/{len(rows)} "
                  f"({done / max(time.monotonic() - start, 1e-9):.2f}/s)",
                  end="\r", file=sys.stderr, flush=True)
        for synth in synths:
            synth.delete()
    print(file=sys.stderr)
    if failures:
        print(f"{len(failures)} failures, first few: {failures[:3]}",
              file=sys.stderr)

    # the checkpoints are drawn in the sum's order, not each in its own, so the
    # plot shows what each contributed at the ranking that was actually used
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(1, len(combined) + 1)
    for t in tags:
        ax.plot(x, total[t][order], lw=0.7, alpha=0.55, label=t)
    ax.plot(x, combined[order], lw=1.1, color="#1f4e79", label="both")
    cut = args.near_max * combined.max()
    ax.axhline(cut, color="#c00000", lw=0.9, ls="--")
    ax.text(len(x), cut, f" {args.near_max} x max = {cut:.1f} ",
            fontsize=8, color="#c00000", va="bottom", ha="right")
    ax.axhline(0, color="grey", lw=0.7, ls=":")
    ax.set_xlabel("rank by the summed score")
    ax.set_ylabel("summed score over gm programs 1-128")
    ax.legend(fontsize=8)
    fig.suptitle("agreement with the general midi sound set", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "rank_curve.png", dpi=130)
    plt.close(fig)
    print(f"wrote {args.out} ({len(rows)} banks)")


if __name__ == "__main__":
    main()
