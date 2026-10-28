"""
Where the pieces of an embedding store are.

    <embedding_dir>/clap_<tag>/
        audio/                          per-key store, and embed_info.json
        inst/instrument_embedding.npz   one mean per instrument
        text/<query file name>.npz      one vector per query

simap-embedding ships as one directory holding every checkpoint's store, so all
of them are assumed to be there. The checkpoint files are a separate matter:
they are downloaded one by one, so only some may be on disk, and that only
affects free-text retrieval -- everything else reads embeddings that were
computed already and needs no checkpoint at all.

A store records the checkpoint that built it, in audio/embed_info.json, so the
checkpoint is read from the store rather than named beside it. That is what
keeps the pairing honest: text embedded under one checkpoint and scored against
instrument embeddings from another does not fail, it returns nonsense, the same
string landing on nearly orthogonal vectors under the two.

The checkpoint file itself is not this module's business -- it lives under
laion_clap_dir, which only free-text retrieval ever looks at.
"""

import json
from dataclasses import dataclass
from pathlib import Path

CLAP_TAGS = ("fusion", "music")


@dataclass(frozen=True)
class ClapStore:
    """One checkpoint's corner of the embedding directory."""

    tag: str
    root: Path

    @classmethod
    def under(cls, embedding_dir: Path, tag: str) -> "ClapStore":
        """So the clap_ prefix is written here rather than in every caller."""
        return cls(tag, Path(embedding_dir) / f"clap_{tag}")

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def inst_npz(self) -> Path:
        return self.root / "inst" / "instrument_embedding.npz"

    def text_npz(self, query_file: Path) -> Path:
        """embed_text.py names its output after its input, so this is derivable."""
        return self.root / "text" / (Path(query_file).name + ".npz")

    @property
    def info(self) -> dict:
        """audio/embed_info.json: the checkpoint, and how the audio was rendered."""
        path = self.audio_dir / "embed_info.json"
        if not path.is_file():
            raise SystemExit(f"{path} is missing, so there is no record of which "
                             f"checkpoint {self.root} was embedded with")
        return json.loads(path.read_text())


def selected_stores(embedding_dir: Path, fusion_only: bool,
                    music_only: bool) -> list[ClapStore]:
    """Every checkpoint's store, or just the one asked for."""
    if fusion_only and music_only:
        raise SystemExit("--fusion-only and --music-only are contradictory; omit "
                         "both to use every checkpoint")
    tags = ("fusion",) if fusion_only else ("music",) if music_only else CLAP_TAGS
    return [ClapStore.under(embedding_dir, tag) for tag in tags]
