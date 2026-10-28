"""
Score every instrument on whether it follows the General MIDI percussion map,
under both checkpoints at once.

    <out>/rank_curve.png                       summed score against rank
    <out>/<rank>_s<frac>_<metadata_idx>_<label>/
        key_scores.png                         score against key
        key<k>_<gm name>.wav                   every mapped key, 35 to 81

The renders use velocity 100, a 2 s note in a 3 s window -- the same settings the
embeddings were made from, so what is heard is what was scored. They come from the
soundfont, not from a checkpoint, so they are rendered once however many stores
contributed to the score.

GM fixes what each key of a drum kit must be: note 35 is an acoustic bass drum,
42 a closed hi-hat, 81 an open triangle. So for a kit the specification is the
ground truth, and no author label is needed. The score is

    sum over k in 35..81 of  cos( audio(key k) , text(gm name for note k) )

and nothing else. It is positional on purpose: the question is whether key 42
sounds like a closed hi-hat, not whether a closed hi-hat is somewhere in the
instrument.

Summed over keys rather than averaged, so a higher score needs both more of the map
covered and a better match on each key. An unmapped key contributes nothing -- its
embedding is nan in the store, which is zeroed before the sum rather than left to
propagate.

gm_perc_map.json gives one name per note over 35..81, and all 47 are
distinct, so notes and names are one to one.

Every instrument is scored, not just the ones in bank 128. Bank 128 is where a kit
belongs, not where kits are: plenty of authors put one in bank 0, and filtering on
the bank would drop exactly the positives this is meant to find. Scoring everything
also supplies the negative control -- a melodic instrument is one timbre at 47
pitches where the map asks for 47 different timbres, so it must score low, and
where the rank curve falls away is where kits stop and everything else begins.

Two checkpoints
---------------
Both stores are found under --embedding-dir, and each is scored against the text
vectors built from its own checkpoint, so the pairing cannot be crossed: the same
string embeds to nearly orthogonal vectors under the two (mean cosine about
-0.008), and audio(fusion) has to meet text(fusion). --fusion-only and
--music-only score with one checkpoint instead of both.

The two per-checkpoint scores are then added, and the sum is the only thing that
ranks, thresholds and names a folder. Adding raw scores is fair here because the
two land on nearly the same scale -- measured over a 500 file sample, fusion peaks
at 24.95 against music's 22.48 and their spreads differ by 14 per cent -- so neither
checkpoint quietly outvotes the other. They are worth adding rather than picking
one because they disagree: the rank correlation between them is 0.72, so the sum
carries an agreement the individual scores do not.

Usage:
    python scripts/subset/gm_perc.py --out <out_dir>/gm_perc

--embedding-dir and --data-dir fall back to what `simap set-dir` recorded, so
neither is normally given. No checkpoint is needed: every vector this reads was
computed already.
"""

import argparse
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

LOW_KEY, HIGH_KEY = 21, 108
KEY_SPAN = HIGH_KEY - LOW_KEY + 1
PERC_LOW, PERC_HIGH = 35, 81
SAMPLE_RATE = 48000
VELOCITY, NOTE_DUR, RENDER_DUR = 100, 2.0, 3.0


def read_perc_map(path: Path) -> dict[int, str]:
    """The gm percussion map, note number to official name, over 35..81."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    mapping = {int(note): str(name) for note, name in loaded.items()}
    missing = [n for n in range(PERC_LOW, PERC_HIGH + 1) if n not in mapping]
    if missing:
        raise ValueError(f"{path} has no gm name for notes {missing}")
    return mapping


def target_matrix(mapping: dict[int, str], text_path: Path) -> np.ndarray:
    """
    One row per key of the store, holding the text vector the key should match.

    Rows outside the percussion range are zero, so the row-wise dot product is
    defined for the whole span and only the percussion keys can contribute.
    """
    z = np.load(text_path)
    queries = [str(q) for q in z["queries"]]
    vectors = z["embedding"].astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    T = np.zeros((KEY_SPAN, vectors.shape[1]), np.float32)
    for note, name in mapping.items():
        if name not in queries:
            raise SystemExit(f"{text_path} has no vector for {name!r}")
        T[note - LOW_KEY] = vectors[queries.index(name)]
    return T


def score_store(audio_dir: Path, T: np.ndarray):
    """Per-key percussion scores and the mapped-key mask, for one whole store."""
    lo, hi = PERC_LOW - LOW_KEY, PERC_HIGH - LOW_KEY + 1
    paths = sorted(q for q in audio_dir.rglob("*.npz")
                   if q.name != "instrument_embedding.npz")
    idx_parts, key_parts, mask_parts = [], [], []
    start = time.monotonic()
    for done, path in enumerate(paths, 1):
        with np.load(path) as z:
            metadata_idx, mask, emb = z["metadata_idx"], z["keys_mask"], z["embedding"]
        # nan times zero is still nan, so the unmapped keys are zeroed rather
        # than relying on the zero rows of T to silence them
        per_key = (np.where(mask[:, :, None], emb, 0.0) * T[None, :, :]).sum(axis=2)
        idx_parts.append(metadata_idx.astype(np.int32))
        key_parts.append(per_key[:, lo:hi].astype(np.float32))
        mask_parts.append(mask[:, lo:hi])
        if done % 500 == 0 or done == len(paths):
            print(f"  {done}/{len(paths)} files "
                  f"({done / max(time.monotonic() - start, 1e-9):.0f}/s)",
                  end="\r", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return (np.concatenate(idx_parts), np.concatenate(key_parts),
            np.concatenate(mask_parts))


def plot_keys(path: Path, title: str, notes, components, combined, totals, mapping):
    """
    Score against key: the sum that did the ranking, over the checkpoints it is
    made of.

    All three share one y axis, which the module docstring justifies -- the two
    checkpoints agree on scale to within about 10 per cent -- and which is what
    lets the sum read as the two lines added rather than as a third curve.
    """
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for tag, scores in components.items():
        ax.plot(notes, scores, "-", lw=0.8, alpha=0.55,
                label=f"{tag}   sum {totals[tag]:.2f}")
    ax.plot(notes, combined, "-o", ms=3, lw=1.5, color="#1f4e79",
            label=f"both   sum {totals['both']:.2f}")
    ax.axhline(0, color="grey", lw=0.7, ls=":")
    mean = np.nanmean(combined)
    ax.axhline(mean, color="#c00000", lw=0.9, ls="--")
    ax.text(notes[-1], mean, f" mean {mean:.3f} ", ha="right", va="bottom",
            fontsize=8, color="#c00000")
    ax.set_xticks(notes)
    ax.set_xticklabels([f"{n} {mapping[n]}" for n in notes], rotation=90, fontsize=6)
    ax.set_xlabel("key, and the general midi name it was scored against")
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
    p.add_argument("--perc-map", type=Path, default=None,
                   help="note number to official gm name, over 35..81; "
                        "default: gm_perc_map.json in the data dir")
    p.add_argument("--out", type=Path, required=True, help="without a suffix")
    p.add_argument("--near-max", type=float, default=0.85,
                   help="keep an instrument scoring at least this fraction of "
                        "the best summed score")
    p.add_argument("--dedup-tol", type=float, default=0.0,
                   help="two instruments are the same kit when no key's score "
                        "differs by more than this in either store; off by "
                        "default, since the corpus is deduplicated at the "
                        "instrument level and this only adds near misses")
    p.add_argument("--top-k", type=int, default=None,
                   help="an extra hard cap after the threshold and the dedup")
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
    perc_map = args.perc_map or data_dir / "gm_perc_map.json"

    mapping = read_perc_map(perc_map)
    notes = list(range(PERC_LOW, PERC_HIGH + 1))
    print(f"{len(set(mapping.values()))} names over {len(mapping)} notes",
          file=sys.stderr)

    stores = {}
    for store in selected_stores(embedding_dir, args.fusion_only, args.music_only):
        # the store says what range it was rendered over, and the key axis here
        # is built from LOW_KEY: a store rendered over anything else would be
        # silently misaligned rather than wrong in any visible way
        info = store.info
        if (info["low_key"], info["high_key"]) != (LOW_KEY, HIGH_KEY):
            raise SystemExit(
                f"{store.tag} covers keys {info['low_key']}-{info['high_key']}, "
                f"but this script assumes {LOW_KEY}-{HIGH_KEY}")
        print(f"{store.tag}: scoring {store.audio_dir} ({info['checkpoint']})",
              file=sys.stderr)
        stores[store.tag] = score_store(
            store.audio_dir, target_matrix(mapping, store.text_npz(perc_map)))
    tags = list(stores)

    # the stores were written by one pipeline over one corpus, so their rows and
    # their mapped keys should line up; if they ever do not, adding the scores
    # adds two different instruments together
    metadata_idx, _, key_mask = stores[tags[0]]
    for tag in tags[1:]:
        if not np.array_equal(stores[tag][0], metadata_idx):
            raise SystemExit(f"{tag} covers different instruments from {tags[0]}")
        if not np.array_equal(stores[tag][2], key_mask):
            raise SystemExit(f"{tag} has different mapped keys from {tags[0]}")
    per_key = {t: stores[t][1] for t in tags}
    total = {t: np.where(key_mask, per_key[t], 0.0).sum(axis=1) for t in tags}

    # one score decides everything from here: the checkpoints are added, not
    # thresholded apart and unioned, so there is a single rank and a single
    # fraction to read
    combined_key = sum(per_key[t] for t in tags)
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

    # the same kit recurs across preset slots of one file and across re-uploads of
    # one bank. The mean embedding cannot tell those apart -- it averages a kick
    # and a cowbell into one vector, and copies sit above 0.9999 either way -- but
    # the per-key profile can, so the dedup runs on that. The stores are compared
    # side by side rather than added: a chance collision in one checkpoint's
    # projection is very unlikely to be repeated in the other's, and adding first
    # would let a disagreement cancel out into a false match.
    #
    # Off by default now that the corpus is deduplicated at the instrument
    # level, which removes the same copies exactly rather than by proximity.
    # What is left to it is the near miss: a kit rebuilt from resampled or
    # re-encoded audio, whose keys score alike but are not the same bytes
    if args.dedup_tol > 0:
        profile = np.hstack([np.nan_to_num(per_key[t][rows]) for t in tags])
        keep, reps = [], np.empty((0, profile.shape[1]), np.float32)
        for row, prof in enumerate(profile):
            if len(reps) and (np.abs(reps - prof).max(axis=1) < args.dedup_tol).any():
                continue
            keep.append(row)
            reps = np.vstack([reps, prof[None, :]])
        print(f"  dedup at {args.dedup_tol}: {len(keep):,} distinct kits",
              file=sys.stderr)
        rows = rows[keep]
    if args.top_k:
        rows = rows[:args.top_k]

    meta = pq.read_table(data_dir / "metadata.parquet",
                         columns=["file", "synth_id", "channel_id", "text_label"])
    files = meta.column("file").to_pylist()
    synth_ids = meta.column("synth_id").to_numpy()
    channel_ids = meta.column("channel_id").to_numpy()
    labels = meta.column("text_label").to_pylist()

    args.out.mkdir(parents=True, exist_ok=True)
    by_file = defaultdict(list)
    if args.curve_only:
        # the curve is a property of the scores, so nothing below is needed for it
        print("--curve-only: rank_curve.png and nothing else", file=sys.stderr)
    else:
        for i in rows:
            by_file[files[int(metadata_idx[i])]].append(int(i))
        print(f"rendering {len(rows)} kits across {len(by_file)} files",
              file=sys.stderr)

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
            idx = int(metadata_idx[i])
            name = f"{rank[i]:04d}_s{frac[i]:.2f}_{idx}_{safe(labels[idx])[:32]}"
            sub = args.out / name
            sub.mkdir(parents=True, exist_ok=True)
            mapped = key_mask[i]
            plot_keys(
                sub / "key_scores.png", f"{labels[idx]!r}  {path}", notes,
                {t: np.where(mapped, per_key[t][i], np.nan) for t in tags},
                np.where(mapped, combined_key[i], np.nan),
                {**{t: float(total[t][i]) for t in tags},
                 "both": float(combined[i])},
                mapping,
            )
            for n in notes:
                if not mapped[n - PERC_LOW]:
                    continue          # the key is unmapped, nothing to render
                try:
                    audio = synths[int(synth_ids[idx])].render_note(
                        int(channel_ids[idx]), n, VELOCITY, NOTE_DUR, RENDER_DUR)
                except Exception as e:
                    failures.append((f"{path}[{idx}] key {n}", str(e)))
                    continue
                mono = audio if audio.ndim == 1 else audio.mean(axis=0)
                sf.write(sub / f"key{n}_{safe(mapping[n])}.wav", mono,
                         SAMPLE_RATE, subtype="PCM_16")
            done += 1
        for synth in synths:
            synth.delete()
        print(f"  {done}/{len(rows)} "
              f"({done / max(time.monotonic() - start, 1e-9):.1f}/s)",
              end="\r", file=sys.stderr, flush=True)
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
    ax.set_ylabel("summed score over keys 35-81")
    ax.legend(fontsize=8)
    fig.suptitle("agreement with the general midi percussion map", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "rank_curve.png", dpi=130)
    plt.close(fig)
    print(f"wrote {args.out} ({len(rows)} kits)")


if __name__ == "__main__":
    main()
