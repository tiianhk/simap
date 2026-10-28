import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simap import sample_reader
from simap.embedding_store import ClapStore, selected_stores

# laion-clap's own zero-shot code asks its questions this way
TEMPLATE = "This is a sound of {}."

MAX_AUDIO_FILES = 32
AUDIO_SUFFIXES = sample_reader.AUDIO_SUFFIXES


def _load_model(ckpt: Path, config: dict, device: str):
    """
    One CLAP checkpoint, loaded quietly.

    config is the store's own amodel and enable_fusion rather than a guess from
    the checkpoint filename. Loading under the wrong architecture does not fail,
    it produces vectors that are quietly not comparable.
    """
    import laion_clap
    from transformers.utils import logging as hf_logging

    print(f"  loading {ckpt.name} as {config}", file=sys.stderr)
    held = io.StringIO()
    was = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        with contextlib.redirect_stdout(held), contextlib.redirect_stderr(held):
            model = laion_clap.CLAP_Module(device=device, **config)
            model.load_ckpt(str(ckpt))
    except Exception:
        sys.stderr.write(held.getvalue())
        raise
    finally:
        hf_logging.set_verbosity(was)
    return model


def embed_texts(ckpt: Path, config: dict, texts: list[str], device: str,
                template: str | None) -> np.ndarray:
    model = _load_model(ckpt, config, device)
    asked = [template.format(t) for t in texts] if template else list(texts)
    vectors = model.get_text_embedding(asked, use_tensor=False).astype(np.float32)
    del model
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def embed_audio(ckpt: Path, config: dict, paths: list[Path],
                device: str) -> np.ndarray:
    model = _load_model(ckpt, config, device)
    vectors = model.get_audio_embedding_from_filelist(
        [str(q) for q in paths], use_tensor=False).astype(np.float32)
    del model
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        bad = int(np.argmax(np.abs(norms - 1.0)))
        raise SystemExit(f"{paths[bad]} embedded to length {norms[bad]:.4f}, "
                         f"where clap returns unit vectors")
    return vectors


def audio_files(path: Path, max_files: int) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    found = sorted(q for q in path.rglob("*")
                   if q.is_file() and q.suffix.lower() in AUDIO_SUFFIXES)
    if not found:
        raise SystemExit(f"no audio under {path}; looked for "
                         f"{', '.join(sorted(AUDIO_SUFFIXES))}")
    if len(found) > max_files:
        raise SystemExit(f"{path} holds {len(found):,} audio files, more than "
                         f"the {max_files} this will embed at once. Point at a "
                         f"smaller directory, or raise the limit if the machine "
                         f"has the memory for it")
    return found


@dataclass
class Retrieval:
    """One ranking of the corpus against each of several queries."""

    labels: list[str]                   # the text asked, or the file's name
    stores: list[ClapStore]
    ckpts: dict[str, Path]
    metadata_idx: np.ndarray            # the eligible pool, after min_keys
    n_keys: np.ndarray
    inst: dict[str, np.ndarray]         # per checkpoint, so a caller can break
    text_vec: dict[str, np.ndarray]     # the score down into its parts
    scores: np.ndarray                  # (n_instruments, n_texts)

    @property
    def tags(self) -> list[str]:
        return [store.tag for store in self.stores]

    def select(self, column: int, near_max: float, top_k: int | None):
        order = np.argsort(-self.scores[:, column], kind="stable")
        sorted_scores = self.scores[order, column]
        peak = float(sorted_scores[0])
        if top_k is not None:
            n_keep = min(top_k, len(order))
        elif peak > 0:
            n_keep = int((sorted_scores >= near_max * peak).sum())
        else:
            n_keep = 1
        return order, sorted_scores, n_keep


def retrieve(texts: list[str] | None, embedding_dir: Path, laion_clap_dir: Path,
             *, audio: Path | None = None, max_files: int = MAX_AUDIO_FILES,
             fusion_only: bool = False, music_only: bool = False,
             min_keys: int = 1, device: str = "cpu",
             template: str | None = TEMPLATE) -> Retrieval:
    """
    Rank every eligible instrument against each query. Nothing is written.

    The query is either texts or audio, never both. audio is a path: one file
    is one query, and a directory is one query per audio file under it, up to
    max_files of them.
    """
    laion_clap_dir = Path(laion_clap_dir)
    if (texts and audio) or (not texts and audio is None):
        raise SystemExit("retrieve takes either texts or audio, not both")
    queries = audio_files(audio, max_files) if audio else None
    labels = [q.name for q in queries] if queries else list(texts)
    stores, ckpts = [], {}
    for store in selected_stores(embedding_dir, fusion_only, music_only):
        ckpt = laion_clap_dir / store.info["checkpoint"]
        if not ckpt.is_file():
            print(f"skipping {store.tag}: {ckpt.name} is not in {laion_clap_dir}",
                  file=sys.stderr)
            continue
        stores.append(store)
        ckpts[store.tag] = ckpt
    if not stores:
        raise SystemExit(
            f"no checkpoint in {laion_clap_dir} for any store in {embedding_dir}")
    print(f"retrieving with: {', '.join(s.tag for s in stores)}", file=sys.stderr)

    # the instrument side of every store, checked for alignment before anything is
    # scored: the rows are added together, so a mismatch would add two different
    # instruments rather than fail
    inst, metadata_idx, n_keys = {}, None, None
    for store in stores:
        z = np.load(store.inst_npz)
        if metadata_idx is None:
            metadata_idx, n_keys = z["metadata_idx"], z["n_keys"]
        elif not np.array_equal(z["metadata_idx"], metadata_idx):
            raise SystemExit(f"{store.inst_npz} covers different instruments from "
                             f"{stores[0].inst_npz}")
        inst[store.tag] = z["embedding"].astype(np.float32)

    # the filter is applied to the pool before scoring, so the peak each threshold
    # is measured against belongs to an instrument that can be returned
    if min_keys > 1:
        eligible = n_keys >= min_keys
        if not eligible.any():
            raise SystemExit(f"min_keys {min_keys} leaves no instruments; the most "
                             f"any has is {int(n_keys.max())}")
        print(f"min_keys {min_keys}: {int(eligible.sum()):,} of {len(n_keys):,} "
              f"instruments eligible", file=sys.stderr)
        metadata_idx, n_keys = metadata_idx[eligible], n_keys[eligible]
        inst = {tag: e[eligible] for tag, e in inst.items()}

    # one checkpoint in memory at a time: two CLAP models is gigabytes for no
    # reason, and the texts are all known up front
    text_vec = {}
    for store in stores:
        info = store.info
        config = {"amodel": info["amodel"],
                  "enable_fusion": info["enable_fusion"]}
        if queries:
            text_vec[store.tag] = embed_audio(
                ckpts[store.tag], config, queries, device)
        else:
            text_vec[store.tag] = embed_texts(
                ckpts[store.tag], config, texts, device, template)

    scores = sum(inst[tag] @ text_vec[tag].T for tag in (s.tag for s in stores))
    return Retrieval(labels, stores, ckpts, metadata_idx, n_keys,
                     inst, text_vec, scores)
