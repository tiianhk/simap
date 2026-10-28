"""
Embed text queries with Qwen3-Embedding, one npz per input file.

    <out_dir>/<the input file's own name, with .npz appended>
        queries   : (n_queries,)       unicode   the query strings, in file order
        embedding : (n_queries, dim)   float32   l2 normalised
    <out_dir>/embed_info.json          one for the run, not one per input

Why a second text encoder at all. CLAP's text branch has no world knowledge of
instrument names: measured against "Acoustic Grand Piano", it puts "Bösendorfer
Upright 130 Model" at +0.084 and "Steinway" at +0.027, both below a "Distortion
Guitar" control at +0.204, while every label containing the word "Piano" scores
above +0.7. It is matching the word, not the instrument. Qwen3-Embedding is used
for the semantic half of the problem -- which author labels mean the same thing --
and CLAP is left to do the part it is good at, which is the audio.

The two spaces are not comparable. A Qwen vector cannot be scored against a CLAP
audio vector; they are separate coordinate systems with different dimensions (4096
against 512). Qwen answers text-to-text questions, CLAP answers audio-to-text ones.

Instructions
------------
Qwen3-Embedding is instruction aware and its geometry shifts with the instruction,
so --instruct is part of what a store means and is recorded in embed_info.json. The
same instruction is applied to every line, because the comparisons here are
symmetric -- label against label, label against vocabulary term -- rather than the
asymmetric query-against-document the model's own default prompt assumes.

Input files
-----------
Either plain text, one query per non-empty line, or a json object whose values are
the queries, embedded in the file's own key order. There is no comment syntax: a
query file holds queries and nothing else, so nothing can collide with a label that
contains "#" as the sharp sign.

Usage:
    python scripts/embedding/qwen/embed_text.py \\
        --queries <simap_dir>/gm_sound_set.json <simap_dir>/gm_perc_map.json \\
                  <simap_dir>/sonnet5_text_filtering.json \\
        --out-dir <dir> --model <path to the snapshot>

Needs a gpu: the model is 8 B parameters, and on cpu the label file would take
hours rather than minutes. Everything downstream of this script works on the stored
vectors and needs no gpu at all, which is the point of storing them.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# the task description, which the model card asks developers to write for their own
# scenario and reports as worth 1 to 5 per cent on retrieval. The wording is not
# neutral: an instruction about instruments pulls maker names towards the
# instruments they make, which is the whole reason this model is here
INSTRUCT = ("Given the name of a synthesiser preset or sampled instrument, "
            "represent the musical instrument and timbre it denotes")
# the model's own format, from its stored "query" prompt and its
# get_detailed_instruct helper: "Instruct: {task}\nQuery:{text}", with no space
# after the colon. The text is appended by sentence-transformers
PROMPT = "Instruct: {}\nQuery:"


def read_queries(path: Path) -> list[str]:
    """
    One query per row, in file order: every non-empty line of a .txt, or every
    distinct value of a .json object.

    Duplicated from scripts/embedding/clap/embed_text.py rather than imported: a script importing
    a sibling script is worse than fifteen lines repeated, and the two must agree
    on parsing for their stores to be row-comparable.
    """
    if path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} is not a json object of key to query")
        rows = list(dict.fromkeys(str(v).strip() for v in loaded.values()))
    else:
        rows = path.read_text(encoding="utf-8").splitlines()
    queries = [q for q in (row.strip() for row in rows) if q]
    if not queries:
        raise ValueError(f"{path} holds no queries")
    duplicates = [q for q, n in Counter(queries).items() if n > 1]
    if duplicates:
        raise ValueError(
            f"{path} repeats {len(duplicates)} of its queries, which would break "
            f"the row-to-line correspondence: {duplicates[:5]}"
        )
    return queries


def out_path(out_dir: Path, source: Path) -> Path:
    """example: gm_perc_map.json -> gm_perc_map.json.npz"""
    return out_dir / (source.name + ".npz")


def write_store(out_dir: Path, source: Path, queries: list[str],
                vectors: np.ndarray) -> np.ndarray:
    """
    The queries and their vectors in one npz, so the two cannot drift apart.

    Written to a temporary name and renamed, so an interrupted run leaves no half
    file that the next one would skip as finished. The handle is opened rather than
    passed as a path because np.savez appends .npz to any name that lacks it.
    """
    path = out_path(out_dir, source)
    tmp = path.with_suffix(".npz.partial")
    with open(tmp, "wb") as f:
        np.savez(f, queries=np.array(queries), embedding=vectors)
    tmp.replace(path)
    return np.linalg.norm(vectors, axis=1)


def model_identity(model_path: str) -> dict:
    """
    The model id and revision, rather than wherever the weights happen to sit.

    A hugging face cache path is <cache>/models--<org>--<name>/snapshots/<sha>,
    so both can be recovered from one, and the revision is the part worth
    keeping: it says exactly which weights produced the vectors. Anything that
    is not cache shaped is recorded as given, with no revision.
    """
    parts = Path(model_path).parts
    if len(parts) >= 3 and parts[-2] == "snapshots" and parts[-3].startswith("models--"):
        return {
            "model": parts[-3].removeprefix("models--").replace("--", "/"),
            "revision": parts[-1],
        }
    return {"model": model_path, "revision": None}


def write_info(out_dir: Path, model: str, instruct: str, dim: int,
               normalised: bool, dtype: str):
    """One file for the whole run: every input was embedded the same way."""
    with open(out_dir / "embed_info.json", "w") as f:
        json.dump(
            {
                **model_identity(model),
                # the instruction is part of what the vectors mean, not a setting
                "instruct": instruct,
                "embedding_dim": dim,
                "compute_dtype": dtype,
                "l2_normalised": normalised,
                "pooling": "last token",
                "embedded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )


def run(query_files: list[Path], out_dir: Path, model_path: str, instruct: str,
        device: str, batch_size: int, dtype: str, overwrite: bool):
    # read and check every file before loading 15 GB of weights, so a typo in a
    # path or a duplicated query fails in a second rather than after the load
    parsed = {}
    for path in query_files:
        target = out_path(out_dir, path)
        if target.is_file() and not overwrite:
            print(f"{path.name}: already embedded at {target.name}, skipping "
                  f"(--overwrite to redo)", file=sys.stderr)
            continue
        parsed[path] = read_queries(path)
        print(f"{path.name}: {len(parsed[path])} queries -> {target.name}",
              file=sys.stderr)
    if not parsed:
        print("nothing left to embed")
        return

    import torch
    from sentence_transformers import SentenceTransformer

    print(f"loading {model_path}", file=sys.stderr)
    model = SentenceTransformer(
        model_path,
        device=device,
        model_kwargs={"torch_dtype": getattr(torch, dtype)},
        # last token pooling needs the real last token, and left padding is what
        # keeps it in a fixed position across a batch
        tokenizer_kwargs={"padding_side": "left"},
    )
    print(f"  dim {model.get_sentence_embedding_dimension()}, "
          f"max_seq_length {model.max_seq_length}", file=sys.stderr)

    for path, queries in parsed.items():
        print(f"\n{path.name}", file=sys.stderr)
        start = time.monotonic()
        vectors = model.encode(
            queries,
            prompt=PROMPT.format(instruct) if instruct else None,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        if len(vectors) != len(queries):
            raise RuntimeError(f"got {len(vectors)} vectors for {len(queries)} queries")
        rate = len(queries) / max(time.monotonic() - start, 1e-9)
        norms = write_store(out_dir, path, queries, vectors)
        # the tolerance is loose because the vectors are normalised in bfloat16,
        # whose ~3 decimal digits leave norms spread over about 1e-3 either way;
        # a tighter check reports a normalised store as unnormalised
        write_info(out_dir, model_path, instruct, int(vectors.shape[1]),
                   bool(np.allclose(norms, 1.0, atol=1e-2)), dtype)
        print(f"  wrote {out_path(out_dir, path).name}  {vectors.shape}  "
              f"norms {norms.min():.3f}-{norms.max():.3f}  "
              f"({rate:.0f} queries/s)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, nargs="+", required=True,
                        help="one or more text files, one query per line")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B",
                        help="a local snapshot directory, or a hub id to download")
    parser.add_argument("--instruct", default=INSTRUCT,
                        help="the task instruction prefixed to every line")
    parser.add_argument("--no-instruct", dest="instruct", action="store_const",
                        const="", help="embed the bare strings instead")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="compute dtype; the vectors are stored as float32")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(args.queries, args.out_dir, args.model, args.instruct, args.device,
        args.batch_size, args.dtype, args.overwrite)


if __name__ == "__main__":
    main()
