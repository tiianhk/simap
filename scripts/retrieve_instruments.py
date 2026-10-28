"""
Retrieve instruments for arbitrary text, under both CLAP checkpoints at once.

The text is embedded here and now, so any string can be asked rather than only the
ones some earlier run happened to put in a store.

    <out>/retrieval_info.json               the run, and each row's exact text
    <out>/<row>_<text>/
        scores.csv                          what was written, with author labels
        rank_curve.png                      summed score against rank
        <rank>_<score>_<metadata_idx>/
            key_curve.png                   score against key, per checkpoint
            key<k>_dur<d>.wav               a few notes from its mapped range

Each text's retrieved metadata_idx go to stdout in rank order as the run proceeds,
before the renders, so the answer is readable without opening scores.csv and is
there while the audio is still being written.

--out is optional. Without it nothing is written at all and only those indices are
printed, which is the cheap way to find out how many a text selects before paying
for the renders -- a broad text on the default --near-max can select hundreds, and
each one costs five renders and a plot.

The score for one instrument is

    sum over checkpoints of  dot( unit text vector , the instrument's key-mean )

and a dot against that unnormalised mean is exactly the mean of the instrument's
per-key cosines, so the number on rank_curve.png is the mean of the summed line on
key_curve.png. The two plots are two views of one quantity. See
build_instrument_embeddings.py for why the mean is not renormalised.

Because the score is a mean over keys and not a sum across them, an instrument with
two mapped keys is measured far less reliably than one with eighty-eight, and narrow
means have heavy enough tails to surface at the extremes of a ranking more often
than their share of the corpus. --min-keys is the guard, applied before scoring so
that each query's best score belongs to an instrument that can actually be written.

How many to write
-----------------
--near-max is a threshold: everything scoring at least that fraction of this text's
own best. A text the corpus is full of writes hundreds, a text with one clear winner
writes one, and neither needs a count chosen in advance. --top-k is a plain count,
and overrides --near-max when both are given. A best score at or below zero puts the
threshold above the maximum, so only the single best is written rather than nothing.

Two checkpoints
---------------
Both stores are found under --embedding-dir, and each one names the checkpoint it
was built from, which is then looked up in --laion-clap-dir. So the text is always
embedded with the checkpoint whose instrument embeddings it is scored against, and
the pairing cannot be crossed: a text embeds to nearly orthogonal vectors under
the two. A checkpoint that has not been downloaded is skipped with a word on
stderr, since retrieving under one of them is a supported way to work -- it just
puts the scores on a different scale. --fusion-only and --music-only choose one
deliberately. Adding the raw scores is fair because they land on nearly the same
scale -- over the 128 general midi names the mean per-text maximum is 0.528 for
fusion against 0.465 for music, and the spreads differ by 9 per cent. They are worth
adding rather than picking one because they disagree: the per-text rank correlation
between them averages 0.67, though it falls as low as 0.02 on individual texts,
where the sum is averaging two unrelated opinions.

The audio is not what was embedded. Embeddings come from one fixed note per key at
velocity 100, 2 s long in a 3 s render. These renders sample the key uniformly from
the instrument's mapped keys and the duration uniformly from 0.5 to 3 s, to hear
across the range rather than to reproduce the input. Notes are seeded per instrument,
so an instrument retrieved by several texts sounds the same under each.

The folder name is lossy: unsafe characters become "_", which can make two texts
collide. The row number prefix keeps folders distinct, and retrieval_info.json
records each row's exact string.

Usage:
    python scripts/retrieve_instruments.py \\
        --out <dir> --text "a bright metallic bell" "distorted electric guitar"

--embedding-dir, --laion-clap-dir and --data-dir fall back to what `simap set-dir`
recorded, so none is normally given.

Run it in the simap-clap environment: this needs laion-clap for the text branch as
well as a working fluidsynth for the renders. The text branch runs on cpu in a
second or two for a handful of strings, so --device is rarely worth changing.
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from simap.retrieve_inst import MAX_AUDIO_FILES, TEMPLATE, retrieve
from simap.synth_interface import load
from simap.user_config import get_user_config

LOW_KEY, HIGH_KEY = 21, 108  # as embedded; keys_mask row i is key LOW_KEY + i
SAMPLE_RATE = 48000
VELOCITY = 100
DUR_RANGE = (0.5, 3.0)  # note duration, uniform
TAIL = 1.0  # render_dur is note_dur + this, to let the release through


def folder_name(row: int, text: str) -> str:
    """
    A filesystem-safe folder for a text, kept distinct by its row number.

    Trailing dots and spaces go too: they are legal here but confusing to read and
    hostile to anything that later moves the tree to another filesystem.
    """
    safe = re.sub(r"[^\w\-. ]", "_", text).strip().rstrip(". ")
    return f"{row:03d}_{safe}"[:120] if safe else f"{row:03d}"


def plot_rank_curve(path: Path, text: str, sorted_scores: np.ndarray,
                    near_max: float | None, top_k: int | None, n_write: int):
    """Score against rank over every instrument, so a top score has a context."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ranks = np.arange(1, len(sorted_scores) + 1)
    ax.plot(ranks, sorted_scores, lw=0.9, color="#1f4e79")
    peak = float(sorted_scores[0])
    if top_k is not None:
        # a count, so the cut is vertical: it says nothing about score, which is
        # the point of asking for a count instead of a threshold
        ax.axvline(top_k, color="#2e7d32", lw=0.9, ls="-.")
        ax.text(top_k, sorted_scores.min(), f" --top-k {top_k} ",
                va="bottom", ha="left", fontsize=8, color="#2e7d32")
    elif peak > 0:
        cut = near_max * peak
        ax.axhline(cut, color="#2e7d32", lw=0.9, ls="-.")
        ax.text(len(sorted_scores), cut,
                f"{near_max} x max = {cut:.3f}, {n_write:,} written ",
                va="bottom", ha="right", fontsize=8, color="#2e7d32")
    else:
        # a negative best score makes the threshold sit above the maximum, so only
        # the single best instrument is written and the line would be meaningless
        ax.text(0.5, 0.9, f"every score is below zero (max {peak:.3f}): "
                          f"only the best is written",
                transform=ax.transAxes, ha="center", fontsize=8, color="#2e7d32")

    # percentiles of the score, so a rank reads as a fraction of the corpus rather
    # than as a bare count
    n = len(sorted_scores)
    for pct in (50, 90, 95, 99):
        value = float(np.percentile(sorted_scores, pct))
        ax.axhline(value, color="grey", lw=0.7, ls=":")
        ax.text(n, value, f" p{pct} {value:+.3f} ", va="bottom", ha="right",
                fontsize=7, color="grey")
    ax.set_xlabel(f"rank among all {n:,} instruments")
    ax.set_ylabel("summed mean cosine over the instrument's keys")
    ax.set_title(f"{text!r}", fontsize=10)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_key_curve(path: Path, text: str, label: str, keys: np.ndarray,
                   per_key: dict[str, np.ndarray], score: float, rank: int):
    """
    Score against key for one instrument, with the summed mean drawn on.

    Only mapped keys are plotted, and they are plotted as points joined by a line,
    so a gap in the line is an unmapped stretch rather than an interpolation. The
    mean of the summed line is the score that ranked the instrument, which is what
    the horizontal rule marks.
    """
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for tag, values in per_key.items():
        ax.plot(keys, values, "-", lw=0.8, alpha=0.55, label=tag)
    both = sum(per_key.values())
    ax.plot(keys, both, "-o", ms=2.5, lw=1.2, color="#1f4e79", label="both")
    ax.axhline(score, color="#c00000", lw=1.0, ls="--")
    ax.text(HIGH_KEY, score, f" mean {score:.3f} ", va="bottom", ha="right",
            fontsize=8, color="#c00000")
    ax.set_xlim(LOW_KEY - 1, HIGH_KEY + 1)
    ax.set_xlabel("MIDI key")
    ax.set_ylabel("cosine with the text")
    ax.set_title(f"#{rank}  {label!r}   vs {text!r}   ({len(keys)} mapped keys)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def sample_notes(keys: np.ndarray, n_notes: int, seed: int,
                 metadata_idx: int) -> list[tuple[int, float]]:
    """
    Keys and durations for one instrument, seeded by the instrument itself.

    Seeding on metadata_idx rather than on the loop means an instrument retrieved by
    several texts gets the same notes each time, so it can be rendered once and
    compared across texts.
    """
    rng = np.random.default_rng([seed, metadata_idx])
    # without replacement while there are keys to spare, so the notes spread over
    # the range instead of landing on one key repeatedly
    replace = len(keys) < n_notes
    chosen = rng.choice(keys, size=n_notes, replace=replace)
    durs = rng.uniform(*DUR_RANGE, size=n_notes)
    return [(int(k), float(d)) for k, d in zip(chosen, durs)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", nargs="+", default=None,
                        help="the strings to retrieve for, embedded on the spot")
    parser.add_argument("--audio", type=Path, default=None,
                        help="retrieve for sounds instead of strings: one audio "
                             "file, or a directory read as one query per file")
    parser.add_argument("--max-files", type=int, default=MAX_AUDIO_FILES,
                        help="how many files a directory may hold. clap embeds "
                             "them in one batch, so this is bounded by memory, "
                             "at roughly 0.3 GB a file (default: %(default)s)")
    parser.add_argument("--embedding-dir", type=Path, default=None,
                        help="default: the embedding_dir from simap set-dir")
    parser.add_argument("--laion-clap-dir", type=Path, default=None,
                        help="holds the .pt checkpoints; default: the "
                             "laion_clap_dir from simap set-dir")
    parser.add_argument("--fusion-only", action="store_true",
                        help="retrieve with the fusion checkpoint alone")
    parser.add_argument("--music-only", action="store_true",
                        help="retrieve with the music checkpoint alone")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="holds metadata.parquet and the sf2/sf3/sfz files; "
                             "default: the simap_dir from simap set-dir")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write; omitted, nothing is written and "
                             "only the retrieved indices are printed")
    parser.add_argument("--min-keys", type=int, default=1,
                        help="ignore instruments whose mean covers fewer mapped keys "
                             "than this. Applied before scoring, so each query's "
                             "best score is one that can actually be written")
    parser.add_argument("--near-max", type=float, default=0.9,
                        help="write every instrument scoring at least this fraction "
                             "of the query's best score. A threshold rather than a "
                             "count, so it adapts to how many the corpus holds")
    parser.add_argument("--top-k", type=int, default=None,
                        help="write exactly this many per query, best first; "
                             "overrides --near-max when both are given")
    parser.add_argument("--no-template", dest="template", action="store_const",
                        const=None, default=TEMPLATE,
                        help=f"embed the bare string instead of {TEMPLATE!r}")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-notes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.top_k is not None and args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")

    config = get_user_config()
    embedding_dir = args.embedding_dir or config.get("embedding_dir")
    if embedding_dir is None:
        raise SystemExit("pass --embedding-dir, or run "
                         "`simap set-dir embedding <dir>` to set it once")
    laion_clap_dir = args.laion_clap_dir or config.get("laion_clap_dir")
    if laion_clap_dir is None:
        raise SystemExit("pass --laion-clap-dir, or run "
                         "`simap set-dir laion_clap <dir>` to set it once")
    data_dir = args.data_dir or config.get("simap_dir")
    if data_dir is None:
        raise SystemExit("pass --data-dir, or run "
                         "`simap set-dir dataset <dir>` to set it once")
    data_dir = Path(data_dir)

    if bool(args.text) == bool(args.audio):
        parser.error("give either --text or --audio, not both and not neither")
    got = retrieve(args.text, embedding_dir, laion_clap_dir,
                   audio=args.audio, max_files=args.max_files,
                   fusion_only=args.fusion_only,
                   music_only=args.music_only,
                   min_keys=args.min_keys, device=args.device,
                   template=args.template)
    stores, tags = got.stores, got.tags
    metadata_idx, n_keys = got.metadata_idx, got.n_keys
    inst, text_vec, scores = got.inst, got.text_vec, got.scores

    # the whole table at once: the alternative is a parquet read per instrument, and
    # the columns are small enough that there is nothing to gain by being selective
    meta = pq.read_table(data_dir / "metadata.parquet",
                         columns=["file", "synth_id", "channel_id", "text_label"])
    files = meta.column("file").to_pylist()
    synth_ids = meta.column("synth_id").to_numpy()
    channel_ids = meta.column("channel_id").to_numpy()
    labels = meta.column("text_label").to_pylist()

    how = (f"--top-k {args.top_k}" if args.top_k is not None
           else f"{args.near_max} x each query's best score")
    print(f"{len(got.labels)} quer{'y' if len(got.labels) == 1 else 'ies'} x "
          f"{len(metadata_idx):,} eligible instruments, keeping {how}",
          file=sys.stderr)

    writing = args.out is not None
    if writing:
        args.out.mkdir(parents=True, exist_ok=True)
    # (metadata_idx) -> [(folder, rank, column, score)], so each instrument is
    # loaded and rendered once however many texts retrieved it
    wanted = defaultdict(list)
    info_rows = []
    for column, text in enumerate(got.labels):
        order, sorted_scores, n_write = got.select(column, args.near_max,
                                                   args.top_k)
        peak = float(sorted_scores[0])
        chosen = [int(metadata_idx[i]) for i in order[:n_write]]

        if writing:
            folder = args.out / folder_name(column, text)
            folder.mkdir(parents=True, exist_ok=True)
            plot_rank_curve(folder / "rank_curve.png", text, sorted_scores,
                            args.near_max, args.top_k, n_write)
            with open(folder / "scores.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["rank", "metadata_idx", "score"] + tags
                                + ["n_keys", "author_label", "file"])
                for rank, i in enumerate(order[:n_write], 1):
                    idx = int(metadata_idx[i])
                    writer.writerow(
                        [rank, idx, f"{scores[i, column]:.6f}"]
                        + [f"{float(inst[t][i] @ text_vec[t][column]):.6f}"
                           for t in tags]
                        + [int(n_keys[i]), labels[idx], files[idx]])
                    # the score travels with the task: it is already known here,
                    # and looking it back up later would mean searching the
                    # sorted index
                    wanted[idx].append(
                        (folder, rank, column, float(scores[i, column])))
            # only retrieval_info.json reads this, so it is built only when there
            # is a retrieval_info.json to write
            info_rows.append({"row": column, "query": text, "folder": folder.name,
                              "n_written": n_write, "top_score": peak,
                              "threshold": (args.near_max * peak
                                            if args.top_k is None and peak > 0
                                            else None),
                              "median_score": float(np.median(scores[:, column]))})

        # the result itself, on stdout and in rank order rather than sorted: best
        # first is the useful order to read, and a python list is the useful shape
        # to paste onward. It prints before the renders, which take far longer than
        # the scoring, so the answer is on screen while the audio is still being
        # written
        print(f"{text!r} ({len(chosen)}): {chosen}", flush=True)

    if not writing:
        return          # the indices are printed; there is nothing else to make

    written = np.array([r["n_written"] for r in info_rows])
    with open(args.out / "retrieval_info.json", "w") as f:
        json.dump(
            {
                "stores": {s.tag: {"checkpoint": got.ckpts[s.tag].name,
                                   "instruments": str(s.inst_npz),
                                   "audio_store": str(s.audio_dir)}
                           for s in stores},
                "score": "sum over checkpoints of dot with the unnormalised "
                         "key-mean == summed mean per-key cosine",
                "template": args.template,
                "n_instruments": int(len(metadata_idx)),
                "min_keys": args.min_keys,
                "near_max": args.near_max if args.top_k is None else None,
                "top_k": args.top_k,
                "written": {"total": int(written.sum()),
                            "distinct_instruments": len(wanted),
                            "per_text_min": int(written.min()),
                            "per_text_median": float(np.median(written)),
                            "per_text_max": int(written.max())},
                "n_notes": args.n_notes,
                "seed": args.seed,
                "velocity": VELOCITY,
                "note_duration_range": list(DUR_RANGE),
                "render_tail": TAIL,
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "queries": info_rows,
            },
            f,
            indent=2,
        )

    # group by source file so each npz and each soundfont is opened once
    by_file = defaultdict(list)
    for idx in wanted:
        by_file[files[idx]].append(idx)
    print(f"{len(wanted):,} distinct instruments across {len(by_file):,} files",
          file=sys.stderr)

    start, done, failures = time.monotonic(), 0, []
    for file, idxs in sorted(by_file.items()):
        # the per-key embeddings behind the mean, one store per checkpoint
        per_store = {}
        for store in stores:
            with np.load(store.audio_dir / (file + ".npz")) as z:
                per_store[store.tag] = (z["metadata_idx"], z["keys_mask"],
                                        z["embedding"])
        first = per_store[tags[0]]
        for tag in tags[1:]:
            if not np.array_equal(per_store[tag][1], first[1]):
                raise SystemExit(f"{file}: {tag} has different mapped keys from "
                                 f"{tags[0]}")
        where = {int(m): r for r, m in enumerate(first[0])}

        synths = None
        try:
            synths = load(data_dir / file, SAMPLE_RATE)
            if synths is None:
                raise RuntimeError("load returned None")
        except Exception as e:
            failures.append((file, f"{type(e).__name__}: {e}"))

        for idx in idxs:
            r = where[idx]
            mask = first[1][r]
            keys = np.flatnonzero(mask) + LOW_KEY
            notes = sample_notes(keys, args.n_notes, args.seed, idx)

            rendered = {}
            if synths is not None:
                synth = synths[int(synth_ids[idx])]
                channel = int(channel_ids[idx])
                for key, dur in notes:
                    try:
                        audio = synth.render_note(channel, key, VELOCITY,
                                                  dur, dur + TAIL)
                    except Exception as e:
                        failures.append((f"{file}[{idx}] key {key}",
                                         f"{type(e).__name__}: {e}"))
                        continue
                    rendered[(key, dur)] = (
                        audio if audio.ndim == 1 else audio.mean(axis=0))

            for folder, rank, column, score in wanted[idx]:
                # four decimals rather than three: adjacent ranks can agree to the
                # third, as 0.4912 and 0.4910 do
                sub = folder / f"{rank:03d}_{score:.4f}_{idx}"
                sub.mkdir(parents=True, exist_ok=True)
                per_key = {
                    tag: per_store[tag][2][r][mask] @ text_vec[tag][column]
                    for tag in tags
                }
                plot_key_curve(sub / "key_curve.png", got.labels[column],
                               labels[idx], keys, per_key, score, rank)
                for (key, dur), mono in rendered.items():
                    sf.write(sub / f"key{key:03d}_dur{dur:.2f}.wav", mono,
                             SAMPLE_RATE, subtype="PCM_16")
            done += 1

        if synths is not None:
            for synth in synths:
                synth.delete()
        rate = done / max(time.monotonic() - start, 1e-9)
        print(f"  {done}/{len(wanted)} instruments ({rate:.1f}/s)",
              end="\r", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    if failures:
        print(f"{len(failures)} failures:", file=sys.stderr)
        for what, why in failures[:20]:
            print(f"  {what}: {why}", file=sys.stderr)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
