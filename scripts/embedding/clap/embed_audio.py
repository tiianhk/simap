"""
Render every mapped key from A0 to C8 of every instrument and store its CLAP embedding (not the audio). 

    <out_dir>/<the input file's own path, suffix replaced by .npz>
        metadata_idx : (n_instruments,)          int32
        keys_mask    : (n_instruments, KEY_SPAN)  bool    True where an embedding exists
        embedding    : (n_instruments, KEY_SPAN, dim) float32, nan where there is none

Row i of an instrument is always MIDI key LOW_KEY + i, so nothing has to be indexed:
embedding[instrument][key - LOW_KEY] is the whole addressing scheme, and it still holds
after the audio is gone. A False mask means the key is unmapped -- it has no sample
assigned and was never rendered -- which is exactly what keymap.parquet records, so
the mask is a cross-check on it rather than a separate fact.

Checkpoints
-----------
--ckpt is optional. Given a path, that checkpoint is loaded and the architecture is
inferred from its filename. Omitted, laion-clap downloads its own default -- but note
that it writes the weights into its site-packages directory, so they are lost on
reinstall and the download fails outright on a read-only install. Pass --ckpt to decide
where weights live. Note also that only the four 630k-* checkpoints are downloadable;
music_audioset_* is a separate release that has no fallback, so it must be given as a
path. --amodel and --fusion override the inference when a filename cannot supply it.

The text encoder pulls roberta-base through transformers, so its cache follows HF_HOME
from the environment rather than anything set here.

Usage:
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/embedding/clap/embed_audio.py \\
        --out-dir <dir> --ckpt <path to .pt> --num-workers 11

LD_LIBRARY_PATH is required: pyarrow otherwise loads the system libstdc++ first, and
libfluidsynth then cannot find the CXXABI it needs.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from simap.synth_interface import load

LOW_KEY, HIGH_KEY = 21, 108  # A0 to C8
KEY_SPAN = HIGH_KEY - LOW_KEY + 1
SAMPLE_RATE = 48000
VELOCITY = 100
NOTE_DUR = 2.0
RENDER_DUR = 3.0


def model_config(ckpt_name: str) -> dict:
    """
    The architecture a checkpoint needs, from its name.

    The music_* checkpoints are HTSAT-base and non-fusion, the 630k-* ones HTSAT-tiny,
    and the *-fusion-* ones need fusion on.
    """
    return {
        "enable_fusion": "fusion" in ckpt_name,
        "amodel": "HTSAT-base" if ckpt_name.startswith("music") else "HTSAT-tiny",
    }


def audio_cfg_for(amodel: str) -> dict:
    """The feature settings the mel needs, read from laion-clap's own model config."""
    import laion_clap.clap_module as clap_module

    path = (Path(clap_module.__file__).parent / "model_configs" / f"{amodel}.json")
    return json.loads(path.read_text())["audio_cfg"]


""" planning """


def out_path(out_dir: Path, file: str) -> Path:
    """mirror SIMAP's own path, appending .npz to the original filename."""
    return out_dir / (file + ".npz")


def plan(data_dir: Path, out_dir: Path) -> tuple[list, int]:
    """create tasks sorted by: files by path, instruments by metadata_idx, keys. resumeable."""
    # read metadata
    meta = pq.read_table(
        data_dir / "metadata.parquet", columns=["file", "synth_id", "channel_id"]
    )
    files = meta.column("file").to_pylist()
    synth_ids = meta.column("synth_id").to_numpy()
    channel_ids = meta.column("channel_id").to_numpy()

    # read keymap
    table = pq.read_table(data_dir / "keymap.parquet", columns=["metadata_idx", "key"])
    idx = table.column("metadata_idx").to_numpy()
    key = table.column("key").to_numpy().astype(np.int16)

    # assert metadata_idx is ascending
    if len(idx) and np.any(np.diff(idx.astype(np.int64)) < 0):
        raise ValueError(f"{data_dir / 'keymap.parquet'} is not sorted by metadata_idx")

    # find unique (metadata_idx, key) pairs
    first = np.empty(len(idx), bool)
    first[0:1] = True
    np.not_equal(idx[1:], idx[:-1], out=first[1:])
    first[1:] |= key[1:] != key[:-1]
    idx, key = idx[first], key[first]

    # select keys between LOW_KEY and HIGH_KEY
    in_range = (key >= LOW_KEY) & (key <= HIGH_KEY)
    idx, key = idx[in_range], key[in_range]

    # find boundaries
    bounds = np.flatnonzero(np.diff(idx.astype(np.int64))) + 1
    key_groups = np.split(key, bounds)
    owners = idx[np.concatenate(([0], bounds))] if len(idx) else idx

    # group them by file
    by_file: dict[str, list] = {}
    for metadata_idx, keys in zip(owners, key_groups):
        metadata_idx = int(metadata_idx)
        by_file.setdefault(files[metadata_idx], []).append(
            (
                metadata_idx,
                int(synth_ids[metadata_idx]),
                int(channel_ids[metadata_idx]),
                keys,
            )
        )

    tasks, todo_notes, done = [], 0, 0
    for file in sorted(by_file):  # by path, so the work order is reproducible
        # the whole of the resume logic: an output file exists only after its rename,
        # so its presence means every instrument in that file finished
        if out_path(out_dir, file).is_file():
            done += 1
            continue
        tasks.append((file, by_file[file]))
        todo_notes += sum(len(k) for _, _, _, k in by_file[file])  # for the log line only
    if done:
        print(f"{done} files already embedded, skipping them", file=sys.stderr)
    return tasks, todo_notes


""" synthesis, in worker processes """


def file_worker(data_dir: Path, file_queue: Queue, result_queue: Queue,
                audio_cfg: dict, fusion: bool):
    """
    Load one file at a time and render every in-range key of every instrument.
    Every mapped key is sent, silent or not; only unmapped keys are missing.
    The mel is computed here, not in the parent.
    """
    # imported in the child: the parent must not hold a cuda context when it forks,
    # and torch alone does not create one
    import torch
    from laion_clap.training.data import (
        get_audio_features, float32_to_int16, int16_to_float32,
    )

    torch.set_num_threads(1)  # eleven workers, one thread each, not eleven times eight
    truncating = "fusion" if fusion else "rand_trunc"
    # the two branches of the model read different things: the fusion branch takes the
    # mel, which is why it is computed here, and the plain one takes the waveform and
    # makes its own mel on the gpu. Sending the other key wastes the pipe, and sending
    # neither is a KeyError deep in the forward pass
    keep = ("longer", "mel_fusion" if fusion else "waveform")

    while True:
        task = file_queue.get()
        if task is None:
            break
        file, instruments = task
        try:
            synths = load(data_dir / file, SAMPLE_RATE)
        except Exception as e:  # a malformed file should not take the worker down
            result_queue.put(("failed", file, f"{type(e).__name__}: {e}", None, None))
            continue
        if synths is None:
            result_queue.put(("failed", file, "load returned None", None, None))
            continue

        for metadata_idx, synth_id, channel_id, keys in instruments:
            synth = synths[synth_id]
            audio = np.zeros((len(keys), int(SAMPLE_RATE * RENDER_DUR)), np.float32)
            for row, key in enumerate(keys):
                rendered = synth.render_note(
                    channel_id, int(key), VELOCITY, NOTE_DUR, RENDER_DUR
                )
                # clap wants mono; averaging
                mono = rendered if rendered.ndim == 1 else rendered.mean(axis=0)
                audio[row, : len(mono)] = mono[: audio.shape[1]]
            features = []
            for row in range(len(keys)):
                waveform = torch.from_numpy(
                    int16_to_float32(float32_to_int16(audio[row]))
                ).float()
                sample: dict = {}
                get_audio_features(sample, waveform, audio_cfg["clip_samples"],
                                   data_truncating=truncating,
                                   data_filling="repeatpad",
                                   audio_cfg=audio_cfg, require_grad=False)
                # numpy rather than tensors because torch registers a Queue reducer that
                # shares storage over file descriptors and needs the sender to outlive
                # the receiver's detach, which a worker that has finished its files does
                # not; numpy pickles by value instead
                features.append({
                    k: v.numpy() for k, v in sample.items() if k in keep
                })
            result_queue.put(("clips", file, metadata_idx, keys, features))
        for synth in synths:
            synth.delete()  # release the memory
        result_queue.put(("file_done", file, None, None, None))

    result_queue.put(None)


""" embedding, in the parent """


def write_file(out_dir: Path, file: str, rows: list, dim: int):
    """One npz per input file, holding every instrument it contains."""
    if not rows:
        return 0
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    metadata_idx = np.array([r[0] for r in rows], dtype=np.int32)
    mask = np.zeros((n, KEY_SPAN), dtype=bool)
    embedding = np.full((n, KEY_SPAN, dim), np.nan, dtype=np.float32)
    for i, (_, keys, vectors) in enumerate(rows):
        slot = keys - LOW_KEY
        mask[i, slot] = True
        embedding[i, slot] = vectors

    path = out_path(out_dir, file)
    path.parent.mkdir(parents=True, exist_ok=True)
    # written to a temporary name and renamed, so a killed job leaves no half file that
    # the next run would mistake for finished work. The handle is opened rather than
    # passed as a path because np.savez appends .npz to any name that lacks it.
    tmp = path.with_suffix(".npz.partial")
    with open(tmp, "wb") as f:
        np.savez(f, metadata_idx=metadata_idx, keys_mask=mask, embedding=embedding)
    tmp.replace(path)
    return int(mask.sum())


def run(data_dir: Path, out_dir: Path, ckpt: Path | None, config: dict,
        num_workers: int, queue_size: int, device: str, max_files: int | None):
    tasks, todo_notes = plan(data_dir, out_dir)
    if max_files:
        tasks = tasks[:max_files]
        todo_notes = sum(len(k) for _, ins in tasks for _, _, _, k in ins)
    if not tasks:
        print("nothing left to embed")
        return
    print(f"{len(tasks)} files, {todo_notes:,} notes to render", file=sys.stderr)

    file_queue: Queue = Queue()
    result_queue: Queue = Queue(maxsize=queue_size)
    for task in tasks:
        file_queue.put(task)
    for _ in range(num_workers):
        file_queue.put(None)

    # the workers are forked before cuda is touched: a process forked from a parent
    # that already holds a cuda context inherits a broken one
    workers = [
        Process(target=file_worker,
                args=(data_dir, file_queue, result_queue,
                      audio_cfg_for(config["amodel"]), config["enable_fusion"]))
        for _ in range(num_workers)
    ]
    for worker in workers:
        worker.start()

    import laion_clap  # imported late so the fork above stays clean

    print(f"loading {ckpt.name if ckpt else 'the laion-clap default'} as {config}",
          file=sys.stderr)
    model = laion_clap.CLAP_Module(device=device, **config)
    model.load_ckpt(str(ckpt)) if ckpt else model.load_ckpt()
    model.eval()

    import torch

    pending: dict[str, list] = {}
    finished = notes = dim = 0
    failures: list[tuple[str, str]] = []
    start = time.monotonic()
    alive = num_workers
    with torch.no_grad():
        while alive:
            message = result_queue.get()
            if message is None:
                alive -= 1
                continue
            kind, file, a, b, c = message
            if kind == "failed":
                failures.append((file, a))
                pending.pop(file, None)
                continue
            if kind == "file_done":
                notes += write_file(out_dir, file, pending.pop(file, []), dim)
                finished += 1
                rate = notes / max(time.monotonic() - start, 1e-9)
                print(f"  [{finished}/{len(tasks)}] {file}  "
                      f"{notes:,} notes, {rate:.0f} notes/s", flush=True)
                continue

            # plan() only emits an instrument that has at least one mapped key in
            # range, and the worker no longer filters, so keys is never empty here
            metadata_idx, keys, features = a, b, c
            # the mel is already done; this is the batched forward pass and nothing else
            vectors = model.model.get_audio_embedding(
                [{k: torch.from_numpy(v) for k, v in f.items()} for f in features]
            )
            vectors = vectors.detach().cpu().numpy().astype(np.float32)
            dim = vectors.shape[1]
            pending.setdefault(file, []).append((metadata_idx, keys, vectors))

    for worker in workers:
        worker.join()

    elapsed = time.monotonic() - start
    print(f"\nembedded {notes:,} notes from {finished} files in {elapsed/3600:.2f} h "
          f"({notes/max(elapsed,1e-9):.0f} notes/s)")
    if failures:
        print(f"{len(failures)} files failed to load:", file=sys.stderr)
        for file, why in failures[:20]:
            print(f"  {file}: {why}", file=sys.stderr)

    with open(out_dir / "embed_info.json", "w") as f:
        json.dump(
            {
                "checkpoint": ckpt.name if ckpt else "laion-clap default",
                **config,
                "embedding_dim": dim,
                "low_key": LOW_KEY,
                "high_key": HIGH_KEY,
                "sample_rate": SAMPLE_RATE,
                "velocity": VELOCITY,
                "note_duration": NOTE_DUR,
                "render_duration": RENDER_DUR,
                "embedded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="holds metadata.parquet, keymap.parquet and the sf2/sf3/sfz files")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="a .pt to load; omitted, laion-clap downloads its default")
    parser.add_argument("--amodel", default=None,
                        help="override the architecture inferred from the ckpt name")
    parser.add_argument("--fusion", action=argparse.BooleanOptionalAction, default=None,
                        help="override the fusion flag inferred from the ckpt name")
    parser.add_argument("--num-workers", type=int, default=11)
    parser.add_argument("--queue-size", type=int, default=24,
                        help="instruments in flight; the bound is what caps memory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-files", type=int, default=None,
                        help="stop after this many files, for a smoke test")
    args = parser.parse_args()

    # laion-clap's own defaults when there is no filename to read
    config = model_config(args.ckpt.name) if args.ckpt else {
        "enable_fusion": True, "amodel": "HTSAT-tiny",
    }
    if args.amodel is not None:
        config["amodel"] = args.amodel
    if args.fusion is not None:
        config["enable_fusion"] = args.fusion
    if args.ckpt and config != model_config(args.ckpt.name):
        print(f"warning: {config} does not match what {args.ckpt.name} suggests "
              f"({model_config(args.ckpt.name)}); loading will fail if this is wrong",
              file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(args.data_dir, args.out_dir, args.ckpt, config, args.num_workers,
        args.queue_size, args.device, args.max_files)


if __name__ == "__main__":
    main()
