"""
Collapse a per-key CLAP audio store to one embedding per instrument, by averaging
over the keys that were rendered.

    <embedding_dir>/clap_<tag>/inst/instrument_embedding.npz
        metadata_idx : (n_instruments,)      int32    joins to metadata.parquet
        embedding    : (n_instruments, dim)  float32  the mean, NOT renormalised
        n_keys       : (n_instruments,)      int16    keys the mean is taken over
        mean_norm    : (n_instruments,)      float32  the mean's own length
    <embedding_dir>/clap_<tag>/inst/instrument_embedding.json   what it was built from

The mean is left unnormalised on purpose. For a unit-length text vector q and
unit-length key embeddings v_i,

    q . mean(v_i)  ==  mean(q . v_i)

so a plain dot product against this file is exactly the average of the per-key
cosine similarities, and the whole 24 GB store never has to be read again at query
time. Taking a cosine instead would divide by mean_norm, which is not a constant:
it runs from about 0.46 to 1.0 across this corpus -- the short end is percussion,
whose keys are unrelated sounds rather than one instrument at many pitches -- so it
rescales every instrument by a different factor, and it rewards the instruments
whose keys agree least, which is the drum kits. That
conversion stays available -- cosine == dot / mean_norm -- which is why mean_norm is
stored rather than divided out here.

n_keys is kept because a mean over three keys and a mean over eighty-eight are not
equally trustworthy, and neither the vector nor its length records which it is.

Only mapped keys are averaged. keys_mask is what says which those are: a key with
no sample assigned was never rendered and is nan in the store, whereas a mapped key
that rendered silence is real data and stays in. So the mask is read rather than
nan being filtered, which would conflate the two.

Usage:
    python scripts/embedding/clap/build_instrument_embeddings.py     # every checkpoint
    python scripts/embedding/clap/build_instrument_embeddings.py --fusion-only

Both paths come from the store layout: <embedding_dir>/clap_<tag>/audio is read and
<embedding_dir>/clap_<tag>/inst is written. They are not free-form, because inst/ is
the only place the consumers look, so a collapse written anywhere else would have to
be moved before anything could read it. inst/ is a sibling of audio/ rather than
inside it: writing back into the store would bury a 0.3 GB file inside the 24 GB it
was made from, which is the one folder someone taking only the collapsed vectors
wants to skip.

The output records the checkpoint it was built from. A text store built with a
different checkpoint is not comparable with it -- the same string embeds to nearly
orthogonal vectors under the two checkpoints, mean cosine about 0.01 -- so the
retrieval script checks the two agree before scoring anything.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from simap.embedding_store import selected_stores
from simap.user_config import get_user_config


def store_files(audio_dir: Path) -> list[Path]:
    """Every npz in the store, in a fixed order so a rebuild is reproducible."""
    files = sorted(p for p in audio_dir.rglob("*.npz") if p.name != "instrument_embedding.npz")
    if not files:
        raise ValueError(f"{audio_dir} holds no .npz files")
    return files


def collapse(audio_dir: Path):
    """
    One mean per instrument, over the instrument's mapped keys.

    Accumulated into lists and concatenated once at the end rather than into a
    preallocated array, because the instrument count is not known until every file
    has been opened, and reading the store twice to find it out costs more than the
    lists do.
    """
    files = store_files(audio_dir)
    idx_parts, mean_parts, n_key_parts = [], [], []
    empty, dim, start = [], None, time.monotonic()

    for done, path in enumerate(files, 1):
        with np.load(path) as z:
            metadata_idx = z["metadata_idx"]
            mask = z["keys_mask"]
            emb = z["embedding"]
        if dim is None:
            dim = emb.shape[2]
        elif emb.shape[2] != dim:
            raise ValueError(f"{path} has dim {emb.shape[2]}, expected {dim}")

        counts = mask.sum(axis=1)
        if (counts == 0).any():
            # an instrument with no mapped key should never have been embedded, so
            # this means the store disagrees with itself rather than that the
            # instrument is unusual
            empty += [int(i) for i in metadata_idx[counts == 0]]

        # sum over the masked keys, then divide: nan sits outside the mask, so
        # zeroing it first lets one matrix op do every instrument in the file
        keys = np.where(mask[:, :, None], emb, 0.0)
        means = keys.sum(axis=1) / np.maximum(counts, 1)[:, None]

        idx_parts.append(metadata_idx.astype(np.int32))
        mean_parts.append(means.astype(np.float32))
        n_key_parts.append(counts.astype(np.int16))

        if done % 500 == 0 or done == len(files):
            rate = done / max(time.monotonic() - start, 1e-9)
            print(f"  {done}/{len(files)} files ({rate:.0f} files/s)",
                  end="\r", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    if empty:
        raise ValueError(
            f"{len(empty)} instruments have no mapped key, which the store should "
            f"never contain: metadata_idx {empty[:10]}"
        )

    metadata_idx = np.concatenate(idx_parts)
    embedding = np.concatenate(mean_parts)
    n_keys = np.concatenate(n_key_parts)

    order = np.argsort(metadata_idx, kind="stable")  # so rows are addressable
    metadata_idx, embedding, n_keys = (
        metadata_idx[order], embedding[order], n_keys[order],
    )
    if len(np.unique(metadata_idx)) != len(metadata_idx):
        raise ValueError("an instrument appears in more than one file of the store")
    if not np.isfinite(embedding).all():
        raise ValueError("a mean came out non-finite, so nan leaked past the mask")

    return metadata_idx, embedding, n_keys, len(files)


def build(audio_dir: Path, out: Path) -> None:
    """Collapse one checkpoint's per-key store into one npz beside its json."""
    out_dir = out.parent
    info_path = audio_dir / "embed_info.json"
    if not info_path.is_file():
        raise SystemExit(
            f"{info_path} is missing, so there is no record of which checkpoint the "
            f"store was embedded with; the retrieval script needs it to refuse a "
            f"mismatched text store"
        )
    audio_info = json.loads(info_path.read_text())

    metadata_idx, embedding, n_keys, n_files = collapse(audio_dir)
    mean_norm = np.linalg.norm(embedding, axis=1).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.partial")
    with open(tmp, "wb") as f:
        np.savez(f, metadata_idx=metadata_idx, embedding=embedding,
                 n_keys=n_keys, mean_norm=mean_norm)
    tmp.replace(out)

    with open(out.with_suffix(".json"), "w") as f:
        json.dump(
            {
                # where the per-key store sits relative to this file, so the two
                # stay connected when the tree is copied or renamed. An absolute
                # path would name this machine, and the leaf alone is "audio"
                # under the shipped layout, which distinguishes nothing
                "audio_store": os.path.relpath(audio_dir.resolve(),
                                               out_dir.resolve()),
                "checkpoint": audio_info.get("checkpoint"),
                "amodel": audio_info.get("amodel"),
                "enable_fusion": audio_info.get("enable_fusion"),
                "embedding_dim": int(embedding.shape[1]),
                "n_instruments": int(len(metadata_idx)),
                "n_keys_total": int(n_keys.sum()),
                "n_files": n_files,
                "aggregation": "mean over mapped keys, not renormalised",
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )

    print(f"{len(metadata_idx):,} instruments from {n_files:,} files, "
          f"{int(n_keys.sum()):,} keys averaged")
    print(f"  n_keys    {n_keys.min()}-{n_keys.max()}, median {int(np.median(n_keys))}")
    print(f"  mean_norm {mean_norm.min():.3f}-{mean_norm.max():.3f}, "
          f"median {np.median(mean_norm):.3f}")
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.0f} MB) and {out.with_suffix('.json')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-dir", type=Path, default=None,
                        help="default: the embedding_dir from simap set-dir")
    parser.add_argument("--fusion-only", action="store_true",
                        help="collapse the fusion store alone")
    parser.add_argument("--music-only", action="store_true",
                        help="collapse the music store alone")
    args = parser.parse_args()

    embedding_dir = args.embedding_dir or get_user_config().get("embedding_dir")
    if embedding_dir is None:
        raise SystemExit("pass --embedding-dir, or run "
                         "`simap set-dir embedding <dir>` to set it once")

    # each store is written in full before the next is read, so an error partway
    # leaves the ones already built intact rather than a half-collapsed directory
    for store in selected_stores(Path(embedding_dir), args.fusion_only,
                                 args.music_only):
        print(f"{store.tag}: {store.audio_dir}", file=sys.stderr)
        build(store.audio_dir, store.inst_npz)


if __name__ == "__main__":
    main()
