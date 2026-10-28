"""
Embed text queries with CLAP's text branch, one npz per input file.

    <out_dir>/<the input file's own name, with .npz appended>
        queries   : (n_queries,)       unicode   the query strings, in file order
        embedding : (n_queries, dim)   float32
    <out_dir>/embed_info.json          one for the run, not one per input

Row i of embedding is query i, in the order read_queries returns them.

Input files
-----------
Either plain text, one query per non-empty line, or a json object whose values are
the queries, embedded in the file's own key order. There is no comment syntax: a
query file holds queries and nothing else, so nothing can collide with a label that
contains "#" as the sharp sign.

Checkpoints
-----------
--ckpt is optional. Given a path, that checkpoint is loaded and the architecture is
inferred from its filename. Omitted, laion-clap downloads its own default into its
site-packages directory, so pass --ckpt to decide where weights live. Only the four
630k-* checkpoints are downloadable; music_audioset_* has no fallback and must be
given as a path. --amodel and --fusion override the inference. One checkpoint per 
run, and the caller picks --out-dir accordingly.

The text encoder pulls roberta-base through transformers, so its cache follows HF_HOME
from the environment rather than anything set here.

Usage:
    python scripts/embedding/clap/embed_text.py \\
        --queries <simap_dir>/gm_sound_set.json <simap_dir>/gm_perc_map.json \\
                  <simap_dir>/sonnet5_text_filtering.json \\
        --out-dir <dir> --ckpt <path to .pt>

--num-threads is there to use fewer cores than are available.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# laion-clap's own zero-shot code asks its questions this way, and so did the
# captions the text branch was trained on. Measured against exact author matches
# over the 118 general midi names with at least thirty of them, it raises mean AUC
# from 0.770 to 0.835 and wins on 96 of the 118, so it is the default rather than an
# option. The gains are largest for common nouns (Oboe 0.44 -> 0.91, Gunshot
# 0.57 -> 0.95) and it loses on rare ones (Shamisen 0.54 -> 0.35), which is the
# shape you would expect if a rare token is diluted by six common ones.
TEMPLATE = "This is a sound of {}."


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


def read_queries(path: Path) -> list[str]:
    """
    One query per row, in file order: every non-empty line of a .txt, or every
    distinct value of a .json object.

    A json keeps its own key order, so a gm map embeds in program or note order.
    Nothing is stripped out of a row -- a query file holds queries and nothing
    else, which is why there is no comment syntax to collide with a label that
    contains "#" as the sharp sign.
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
            f"{path} repeats {len(duplicates)} quer{'y' if len(duplicates) == 1 else 'ies'}, "
            f"which would break the row-to-line correspondence: {duplicates[:5]}"
        )
    return queries


def embed(model, queries: list[str], batch_size: int,
          template: str | None) -> np.ndarray:
    """
    The text branch, in batches. Nothing here reaches past laion-clap's wrapper.

    A template wraps every query before tokenising, and one is applied by default --
    see TEMPLATE for what it buys. A bare noun phrase is off the distribution the
    text branch was trained on, whose captions were sentences.

    The stored queries stay bare either way: the template is how a query is asked,
    not what it is, and keeping it out of the strings means two stores built with
    different templates remain row-for-row comparable.
    """
    import torch

    asked = [template.format(q) for q in queries] if template else queries
    vectors = []
    start = time.monotonic()
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            vectors.append(
                model.get_text_embedding(asked[i:i + batch_size], use_tensor=False)
            )
            done = min(i + batch_size, len(queries))
            print(f"  {done}/{len(queries)} "
                  f"({done / max(time.monotonic() - start, 1e-9):.0f} queries/s)",
                  end="\r", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return np.concatenate(vectors).astype(np.float32)


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


def write_info(out_dir: Path, ckpt: Path | None, config: dict, dim: int,
               normalised: bool, template: str | None):
    with open(out_dir / "embed_info.json", "w") as f:
        json.dump(
            {
                "checkpoint": ckpt.name if ckpt else "laion-clap default",
                **config,
                "embedding_dim": dim,
                "l2_normalised": normalised,
                # what was actually tokenised, since the stored queries are bare
                "template": template,
                "embedded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )


def run(query_files: list[Path], out_dir: Path, ckpt: Path | None, config: dict,
        device: str, batch_size: int, num_threads: int | None, overwrite: bool,
        template: str | None):
    import torch

    if num_threads:
        torch.set_num_threads(num_threads)

    # read and check every file before loading the model, so a typo in a path or a
    # duplicated query fails in a second rather than after the weights are in memory
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

    import laion_clap

    print(f"loading {ckpt.name if ckpt else 'the laion-clap default'} as {config}",
          file=sys.stderr)
    model = laion_clap.CLAP_Module(device=device, **config)
    model.load_ckpt(str(ckpt)) if ckpt else model.load_ckpt()
    model.eval()

    for path, queries in parsed.items():
        print(f"\n{path.name}", file=sys.stderr)
        vectors = embed(model, queries, batch_size, template)
        if len(vectors) != len(queries):
            raise RuntimeError(
                f"got {len(vectors)} vectors for {len(queries)} queries"
            )
        norms = write_store(out_dir, path, queries, vectors)
        normalised = bool(np.allclose(norms, 1.0, atol=1e-4))
        write_info(out_dir, ckpt, config, int(vectors.shape[1]), normalised,
                   template)
        print(f"  wrote {out_path(out_dir, path).name}  {vectors.shape}  "
              f"norms {norms.min():.3f}-{norms.max():.3f}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, nargs="+", required=True,
                        help="one or more text files, one query per line")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="a folder is created under this per input file")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="a .pt to load; omitted, laion-clap downloads its default")
    parser.add_argument("--amodel", default=None,
                        help="override the architecture inferred from the ckpt name")
    parser.add_argument("--fusion", action=argparse.BooleanOptionalAction, default=None,
                        help="override the fusion flag inferred from the ckpt name")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=None,
                        help="torch threads; leave unset for torch's own default")
    parser.add_argument("--template", default=TEMPLATE,
                        help=f"wrap each query before tokenising (default "
                             f"{TEMPLATE!r}); the stored query strings stay bare")
    parser.add_argument("--no-template", dest="template", action="store_const",
                        const=None, help="embed the bare query strings instead")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo a folder that already holds an embedding.npz")
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
    if args.template and "{}" not in args.template:
        raise SystemExit(f"--template must contain {{}}: {args.template!r}")
    run(args.queries, args.out_dir, args.ckpt, config, args.device,
        args.batch_size, args.num_threads, args.overwrite, args.template)


if __name__ == "__main__":
    main()
