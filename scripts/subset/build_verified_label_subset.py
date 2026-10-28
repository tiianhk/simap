"""
Keep the instruments whose own text label is confirmed by their audio.

    <out>   two keys, "direct" and "bridged", each a sorted list of metadata_idx
            joining to metadata.parquet by row

    --bridge-csv    optional, off by default: one row per save, naming the bridge
                    it rests on and the scores behind it. The subset says what is
                    in it; this says how the bridged half got there

This is verification, not relabelling: an instrument is kept under the label its
author gave it, or not kept at all. Nothing here ever writes a new label.

The rule
--------
Every instrument whose author label survived text filtering is put through three
tests in order and lands in exactly one bucket. Scores are the usual sum over the
two checkpoints of cos(text, instrument key-mean), so 0.8 is about 0.4 of mean
per-key cosine per checkpoint.

    (a) clap(own label, instrument) >= --clap-min            -> GREEN, "direct"

        CLAP agrees outright. Nothing more is needed.

    (b) overlap@k(own label) >= --overlap-cut                -> RED, final

        overlap@k asks whether CLAP can read the string at all: take the label's
        k nearest labels under CLAP and under Qwen, in the filtered vocabulary,
        and intersect them. A high overlap means CLAP's sense of what this label
        is near agrees with a text model's, so CLAP understood it, so its low
        score in (a) is a real verdict about the audio and must be respected.

        This is the conservative half of the rule and it rejects the bulk of the
        corpus. It is also what stops (c) from turning into relabelling: a name
        CLAP plainly knows, like "Muted Trumpet", never reaches the bridge.

    (c) some other label scores >= --clap-min on this audio and means nearly the
        same thing as the author's, qwen >= --qwen-min       -> GREEN, "bridged"

        The audio matches the label's meaning even though CLAP cannot parse its
        surface form. This is what rescues maker names and part numbers -- "C3
        6ft Grand" via "Grand Piano Hi" -- which CLAP reads as gibberish.

        Otherwise RED: either no label at all reaches --clap-min on this audio,
        or those that do mean something else.

Note that (c) takes a maximum over every candidate, typically tens to hundreds of
them, not a single pair. One candidate's score moving does not move a verdict.

Thresholds
----------
The defaults are the ones the renders were judged at. --overlap-cut is the one
that was actually contested: above 0.20 the population changes character, from
coinages CLAP cannot read to ordinary instrument names it can -- "Fretless Bass",
"Tubular Bells", "Square Lead" -- and bridging those is relabelling. Raising it to
0.30 adds about half again as many saves, almost all of that kind.

An earlier variant of (b) excluded neighbours sharing a whole word with the label.
It is not offered here: it moved the cut without changing the conclusion, and one
definition of overlap@k is easier to justify than two.

Reproducibility
---------------
Everything is recomputed from the embedding stores on each run, with no cached
intermediate and no randomness, so the file is a function of the stores alone.
The cost is two passes -- 16,864 labels against each other, then every instrument
against every label.

metadata_idx is a row number in metadata.parquet and is not stable across corpus
rebuilds, so this file is pinned to the metadata it was built against.

Usage:
    python scripts/subset/build_verified_label_subset.py \\
        --out <simap_dir>/verified_label_subset.json
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from simap.embedding_store import selected_stores
from simap.user_config import get_user_config

CHUNK = 2000


def unit(a: np.ndarray) -> np.ndarray:
    """Rows to unit length, in float32."""
    a = a.astype(np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def read_vocabulary(stores, qwen_npz: Path, label_map: Path):
    """
    The filtered label vocabulary, embedded by every model that scores it.

    All three stores were written from the same query file, so their query lists
    must be identical and in the same order. Checking is not a formality: the
    arrays are indexed by position throughout, and a mismatch would silently pair
    one label's CLAP vector with another's Qwen vector.
    """
    clap, queries = {}, None
    for store in stores:
        z = np.load(store.text_npz(label_map))
        got = [str(s) for s in z["queries"]]
        if queries is None:
            queries = got
        elif got != queries:
            raise SystemExit(f"{store.text_npz(label_map)} holds different queries "
                             f"from {stores[0].text_npz(label_map)}")
        clap[store.tag] = unit(z["embedding"])
    z = np.load(qwen_npz)
    if [str(s) for s in z["queries"]] != queries:
        raise SystemExit(f"{qwen_npz} holds different queries from the clap stores")
    return queries, clap, unit(z["embedding"])


def label_neighbourhoods(clap: dict, qwen: np.ndarray, k: int, qwen_min: float):
    """
    overlap@k for every label, and the qwen >= qwen_min mask, in one pass.

    Both come off the same pair of similarity matrices, and the matrices are the
    expensive part -- 16,864 squared under two models -- so they are built once
    and consumed before the chunk is dropped. The mask is kept whole because (c)
    needs an arbitrary row of it later; at this vocabulary size that is 284 MB of
    bool, which is cheaper than traversing the labels a second time.

    A label is never its own neighbour: the diagonal is put below every real score
    before the top-k is taken, so each side contributes k genuine neighbours and
    the overlap is not inflated by the one match both are guaranteed to share.
    """
    n = len(qwen)
    overlap = np.empty(n, np.float32)
    close = np.zeros((n, n), bool)
    tags = list(clap)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        rows = np.arange(start, stop)
        diag = np.arange(stop - start)
        c = sum(clap[t][start:stop] @ clap[t].T for t in tags)
        q = qwen[start:stop] @ qwen.T
        close[start:stop] = q >= qwen_min
        c[diag, rows] = -9.0
        q[diag, rows] = -9.0
        c_top = np.argpartition(-c, k, axis=1)[:, :k]
        q_top = np.argpartition(-q, k, axis=1)[:, :k]
        for i in range(stop - start):
            both = np.concatenate([c_top[i], q_top[i]])
            overlap[start + i] = (np.unique(both, return_counts=True)[1] == 2).sum() / k
        print(f"  labels {stop:,}/{n:,}", end="\r", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return overlap, close


def own_label(data_dir: Path, metadata_idx: np.ndarray, queries: list[str],
              label_map: Path):
    """
    Each instrument's position in the vocabulary, or -1 if it has none.

    Two hops: the author's raw label to its filtered form, and that to a row of
    the text store. A label can fail either -- filtering dropped it, or it
    filtered to something the store does not carry -- and both mean the same
    thing here, that the instrument cannot be judged.

    The raw labels come back too, since only they say what the author wrote.
    """
    with open(label_map) as f:
        filtered = json.load(f)
    labels = pq.read_table(data_dir / "metadata.parquet",
                           columns=["text_label"]).column("text_label").to_pylist()
    position = {s: i for i, s in enumerate(queries)}
    own = np.array([position.get(filtered.get(labels[i]), -1) for i in metadata_idx])
    return own, labels


def classify(inst: dict, clap: dict, qwen: np.ndarray, own: np.ndarray,
             judgeable: np.ndarray, overlap: np.ndarray, close: np.ndarray,
             clap_min: float, overlap_cut: float, audit: list | None):
    """
    Run (a), (b), (c) over every judgeable instrument, in chunks.

    Returns the row positions taken by each route and the counts of every bucket.
    The own label cannot appear among a bridge's candidates: reaching clap_min is
    what candidacy means, and an own label that reached it left at (a).

    When audit is a list, one record per save is appended to it. A save can rest
    on many qualifying bridges, and the record names the one with the highest
    qwen score -- the pair the semantic claim leans on hardest -- alongside how
    many candidates there were and how many of them qualified. That count is the
    part worth keeping: the test is an existence test over the whole candidate
    set, so a save backed by one bridge and a save backed by ninety are very
    different claims that the verdict alone cannot tell apart.
    """
    tags = list(inst)
    direct, bridged = [], []
    counts = dict(direct=0, bridged=0, red_legible=0, red_no_bridge=0, red_nothing=0)
    for start in range(0, len(judgeable), CHUNK):
        rows = judgeable[start:start + CHUNK]
        scores = sum(inst[t][rows].astype(np.float32) @ clap[t].T for t in tags)
        o = own[rows]

        is_direct = scores[np.arange(len(rows)), o] >= clap_min
        direct.append(rows[is_direct])

        rest = ~is_direct
        legible = rest & (overlap[o] >= overlap_cut)
        blind = rest & ~legible

        candidate = scores >= clap_min
        has_candidate = candidate.any(axis=1)
        agrees = np.zeros(len(rows), bool)
        for i in np.flatnonzero(blind & has_candidate):
            qualify = np.flatnonzero(candidate[i] & close[o[i]])
            agrees[i] = len(qualify) > 0
            if agrees[i] and audit is not None:
                w = qwen[qualify] @ qwen[o[i]]
                best = int(qualify[w.argmax()])
                audit.append(dict(row=int(rows[i]), own=int(o[i]), bridge=best,
                                  clap_own=float(scores[i, o[i]]),
                                  clap_bridge=float(scores[i, best]),
                                  qwen=float(w.max()),
                                  n_candidates=int(candidate[i].sum()),
                                  n_bridges=len(qualify)))
        bridged.append(rows[blind & agrees])

        counts["direct"] += int(is_direct.sum())
        counts["bridged"] += int((blind & agrees).sum())
        counts["red_legible"] += int(legible.sum())
        counts["red_nothing"] += int((blind & ~has_candidate).sum())
        counts["red_no_bridge"] += int((blind & has_candidate & ~agrees).sum())
        print(f"  instruments {start + len(rows):,}/{len(judgeable):,}",
              end="\r", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return np.concatenate(direct), np.concatenate(bridged), counts


def write_bridge_csv(path: Path, audit: list, metadata_idx: np.ndarray,
                     labels: list[str], queries: list[str], overlap: np.ndarray):
    """One row per save, in metadata_idx order, for reading the bridges by eye."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metadata_idx", "label", "filtered_label", "clap_own",
                         "overlap", "bridge", "clap_bridge", "qwen",
                         "n_candidates", "n_bridges"])
        for r in sorted(audit, key=lambda r: metadata_idx[r["row"]]):
            writer.writerow([
                int(metadata_idx[r["row"]]), labels[metadata_idx[r["row"]]],
                queries[r["own"]], f"{r['clap_own']:.3f}",
                f"{overlap[r['own']]:.2f}", queries[r["bridge"]],
                f"{r['clap_bridge']:.3f}", f"{r['qwen']:.3f}",
                r["n_candidates"], r["n_bridges"]])


def write_json(subset: dict, path: Path):
    """
    One key per line, so the file reads as two rows.

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


def validate(subset: dict, data_dir: Path, metadata_idx: np.ndarray):
    """Check the map against the metadata before writing it."""
    n_rows = pq.read_metadata(data_dir / "metadata.parquet").num_rows
    everything = [i for v in subset.values() for i in v]
    if everything and (min(everything) < 0 or max(everything) >= n_rows):
        raise SystemExit(f"a metadata_idx falls outside the {n_rows:,} rows of "
                         f"{data_dir / 'metadata.parquet'}")
    if len(set(everything)) != len(everything):
        raise SystemExit("an instrument appears under both routes")
    embedded = set(int(i) for i in metadata_idx)
    stray = [i for i in everything if i not in embedded]
    if stray:
        raise SystemExit(f"{len(stray)} kept instruments are not in the instrument "
                         f"store: {stray[:5]}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-dir", type=Path, default=None,
                   help="default: the embedding_dir from simap set-dir")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="default: the simap_dir from simap set-dir")
    p.add_argument("--label-map", type=Path, default=None,
                   help="author label to filtered label; also names the text "
                        "store the vocabulary is read from; default: "
                        "sonnet5_text_filtering.json in the data dir")
    p.add_argument("--out", type=Path, required=True, help="the json to write")
    p.add_argument("--bridge-csv", type=Path, default=None,
                   help="also write one row per save, naming the bridge it rests "
                        "on and the scores behind it; off by default, since it "
                        "records how the subset was arrived at rather than what "
                        "is in it")
    p.add_argument("--clap-min", type=float, default=0.8,
                   help="a label is confirmed by, or a candidate for, an "
                        "instrument at this summed score")
    p.add_argument("--overlap-cut", type=float, default=0.20,
                   help="a label at or above this is one clap can read, so its "
                        "own verdict stands and no bridge is allowed")
    p.add_argument("--qwen-min", type=float, default=0.8,
                   help="a bridge must mean this nearly the same thing")
    p.add_argument("--overlap-k", type=int, default=100,
                   help="neighbours per side in overlap@k")
    p.add_argument("--fusion-only", action="store_true",
                   help="score with the fusion checkpoint alone")
    p.add_argument("--music-only", action="store_true",
                   help="score with the music checkpoint alone")
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
    embedding_dir, data_dir = Path(embedding_dir), Path(data_dir)
    label_map = args.label_map or data_dir / "sonnet5_text_filtering.json"

    stores = selected_stores(embedding_dir, args.fusion_only, args.music_only)
    print(f"scoring with: {', '.join(s.tag for s in stores)}", file=sys.stderr)
    qwen_npz = embedding_dir / "qwen" / (label_map.name + ".npz")
    queries, clap, qwen = read_vocabulary(stores, qwen_npz, label_map)

    # the instrument side of every store, checked for alignment before anything is
    # scored: the rows are added together, so a mismatch would add two different
    # instruments rather than fail
    inst, metadata_idx = {}, None
    for store in stores:
        z = np.load(store.inst_npz)
        if metadata_idx is None:
            metadata_idx = z["metadata_idx"]
        elif not np.array_equal(z["metadata_idx"], metadata_idx):
            raise SystemExit(f"{store.inst_npz} covers different instruments from "
                             f"{stores[0].inst_npz}")
        inst[store.tag] = z["embedding"]

    own, labels = own_label(data_dir, metadata_idx, queries, label_map)
    judgeable = np.flatnonzero(own >= 0)
    print(f"{len(queries):,} labels | {len(metadata_idx):,} embedded instruments, "
          f"{len(judgeable):,} with a label in the vocabulary", file=sys.stderr)

    clock = time.time()
    overlap, close = label_neighbourhoods(clap, qwen, args.overlap_k, args.qwen_min)
    audit = [] if args.bridge_csv else None
    rows_direct, rows_bridged, counts = classify(
        inst, clap, qwen, own, judgeable, overlap, close, args.clap_min,
        args.overlap_cut, audit)
    print(f"  {time.time() - clock:.0f}s", file=sys.stderr)

    subset = {"direct": sorted(int(metadata_idx[i]) for i in rows_direct),
              "bridged": sorted(int(metadata_idx[i]) for i in rows_bridged)}
    validate(subset, data_dir, metadata_idx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(subset, args.out)
    if audit is not None:
        write_bridge_csv(args.bridge_csv, audit, metadata_idx, labels, queries,
                         overlap)

    green = counts["direct"] + counts["bridged"]
    judged, total = len(judgeable), len(metadata_idx)
    print(f"clap >= {args.clap_min}, overlap@{args.overlap_k} < {args.overlap_cut}, "
          f"qwen >= {args.qwen_min}   over {judged:,} labelled instruments\n")
    print(f"  GREEN                           {green:>8,d}  {green / judged:>6.1%} "
          f"of labelled  {green / total:>6.1%} of embedded")
    print(f"    (a) direct                    {counts['direct']:>8,d}")
    print(f"    (c) saved by the qwen bridge  {counts['bridged']:>8,d}")
    print(f"  RED                             "
          f"{counts['red_legible'] + counts['red_no_bridge'] + counts['red_nothing']:>8,d}")
    print(f"    (b) clap can read it, said no {counts['red_legible']:>8,d}")
    print(f"        blind, no bridge          {counts['red_no_bridge']:>8,d}")
    print(f"        blind, nothing reaches "
          f"{args.clap_min} {counts['red_nothing']:>8,d}")
    print(f"  no label in the vocabulary      {total - judged:>8,d}  (cannot be judged)")
    print(f"  wrote {args.out}")
    if audit is not None:
        print(f"  wrote {args.bridge_csv}")


if __name__ == "__main__":
    main()
