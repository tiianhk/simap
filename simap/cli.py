from pathlib import Path

import click

from simap.user_config import get_user_config, set_user_config


@click.group()
def main():
    """SIMAP Command Line Interface"""
    pass


# Allowed keys for set-dir
_DIR_KEYS = ["dataset", "embedding", "laion_clap"]


@main.command()
@click.argument(
    "key",
    type=click.Choice(_DIR_KEYS, case_sensitive=False),
)
@click.argument(
    "directory",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
)
def set_dir(key: str, directory: str):
    """Set a directory in user config"""
    key = key.lower()
    config_key = key + "_dir"
    config = get_user_config()

    if config_key in config:
        if not click.confirm(
            f"{key} directory already set to {config[config_key]}. Overwrite?"
        ):
            click.echo(f"{key} directory not changed")
            return

    config[config_key] = directory
    if config_key == "dataset_dir":
        simap_dir = Path(directory) / "simap"
        config["simap_dir"] = str(simap_dir)
    set_user_config(config)
    click.echo(f"{key} directory set to {directory}")


@main.command()
def measure_memory():
    """
    Measure what each soundfont costs to load, and cache it.

    ChunkedSimapDataset does this by itself the first time it runs, so this is for
    doing it deliberately: before a training job rather than inside one, or again
    after moving to hardware or an environment where the numbers might differ. It
    always measures and always overwrites -- running it is the request.
    """
    from simap.memory_usage import CACHE_PATH, measure, report, save

    record = measure()
    save(record)
    report(record["usage"])
    click.echo(f"Wrote {CACHE_PATH}")


@main.command()
def show_config():
    """Show current user config"""
    config = get_user_config()
    for key in _DIR_KEYS:
        config_key = key + "_dir"
        value = config.get(config_key, "(not set)")
        click.echo(f"{config_key}: {value}")
    for key, value in config.items():
        if key not in (k + "_dir" for k in _DIR_KEYS):
            click.echo(f"{key}: {value}")


@main.command()
@click.option(
    "--download-only",
    is_flag=True,
    default=False,
    help="Only download the dataset, do not unpack, deduplicate, scan, or export",
)
@click.option(
    "--skip-scan",
    is_flag=True,
    default=False,
    help="Skip scanning. Use when external data (e.g. filtered text labels) has been added to the metadata",
)
@click.option(
    "--scan-workers",
    type=int,
    default=1,
    help="Number of scanning workers (default: 1)",
)
@click.option(
    "--also-export",
    is_flag=True,
    default=False,
    help="Export dataset for minimum redundancy",
)
def prepare_dataset(download_only: bool, skip_scan: bool, scan_workers: int, also_export: bool):
    """Download, unpack, deduplicate, scan, and export the dataset"""
    from simap.dataset import download, unpack, deduplicate, scan, export

    config = get_user_config()
    if "dataset_dir" in config:
        simap_dev_dir = Path(config["dataset_dir"]) / "simap_dev"
        simap_dev_dir.mkdir(parents=True, exist_ok=True)
    else:
        raise click.ClickException(
            "Please run first `simap set-dir dataset <dir>` to set the dataset directory."
        )

    server_errors = download(simap_dev_dir, max_retries=3)
    if server_errors > 0:
        click.echo(f"Failed to download {server_errors} files due to server errors.")
    if download_only:
        return

    unpack(simap_dev_dir)
    deduplicate(simap_dev_dir)
    if not skip_scan:
        scan(simap_dev_dir, num_workers=scan_workers)

    if also_export:
        simap_dir = Path(config["simap_dir"])
        if simap_dir.exists():
            click.confirm(
                f"SIMAP directory {simap_dir} already exists and will be overwritten for the new export. Continue?",
                abort=True,
            )
        export(simap_dev_dir, simap_dir)


@main.command()
@click.option("--inst-id", type=int, required=True, help="Index of the metadata entry to render")
@click.option("--pitch", type=int, required=True, help="MIDI note number to play")
@click.option("--velocity", type=int, default=100, show_default=True)
@click.option("--note-dur", type=float, default=1.0, show_default=True, help="Seconds the key is held")
@click.option("--render-dur", type=float, default=2.0, show_default=True, help="Seconds of audio rendered")
@click.option("--sample-rate", type=int, default=16000, show_default=True)
def render(
    inst_id: int,
    pitch: int,
    velocity: int,
    note_dur: float,
    render_dur: float,
    sample_rate: int,
):
    """Render one note from the inst-id-th metadata entry to a wav file."""
    import soundfile as sf
    from simap.synth_interface import load

    config = get_user_config()
    if "simap_dir" not in config:
        raise click.ClickException(
            "Please run first `simap set-dir dataset <dir>` to set the dataset directory."
        )
    simap_dir = Path(config["simap_dir"])
    out_dir = Path("renders")  # relative to where the command is run
    out_dir.mkdir(parents=True, exist_ok=True)

    # the parquet only decodes the row group holding the entry asked for, where the
    # json has to be parsed in full; both are indexed the same way
    parquet_path = simap_dir / "metadata.parquet"
    if parquet_path.is_file():
        from simap.metadata import Metadata

        metadata = Metadata(parquet_path)
    else:
        import json

        with open(simap_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
    if not 0 <= inst_id < len(metadata):
        raise click.ClickException(
            f"inst_id must be in the range [0, {len(metadata) - 1}], got {inst_id}"
        )
    entry = metadata[inst_id]

    synths = load(simap_dir / entry["file"], sample_rate)
    audio = synths[entry["synth_id"]].render_note(
        entry["channel_id"], pitch, velocity, note_dur, render_dur
    )
    for synth in synths:
        synth.delete()

    out_path = (
        out_dir
        / f"{inst_id}_p{pitch}_v{velocity}_d{note_dur:g}_r{render_dur:g}_sr{sample_rate}.wav"
    )
    sf.write(out_path, audio.T, sample_rate)  # (channels, samples) -> (samples, channels)

    click.echo(
        f"{entry['file']} synth {entry['synth_id']} channel {entry['channel_id']}: "
        f"{entry['text_label']}"
    )
    click.echo(f"Rendered to {out_path}")


@main.command()
@click.argument("queries", nargs=-1, required=True)
@click.option("--limit", type=int, default=10, show_default=True, help="Max rows per query")
def find_label(queries: tuple, limit: int):
    """
    Substring match on the author's label: print the inst_id and text label of
    every entry whose label contains a QUERY.
    """
    import json
    import random
    import unicodedata

    def norm(s: str) -> str:
        """Casefold and strip accents, so "Cuíca" and "cuica" match."""
        return (
            unicodedata.normalize("NFKD", s.casefold())
            .encode("ascii", "ignore")
            .decode()
        )

    config = get_user_config()
    if "simap_dir" not in config:
        raise click.ClickException(
            "Please run first `simap set-dir dataset <dir>` to set the dataset directory."
        )
    with open(Path(config["simap_dir"]) / "metadata.json", "r") as f:
        metadata = json.load(f)

    # normalize each label once, and test it against every query in the same pass
    normed_queries = list(enumerate(norm(query) for query in queries))
    hits = [[] for _ in queries]
    for i, m in enumerate(metadata):
        n = norm(m["text_label"])
        for k, q in normed_queries:
            if q in n:
                hits[k].append((i, m["text_label"]))

    for query, matches in zip(queries, hits):
        if len(matches) > limit:
            click.echo(f"=== {query} ({len(matches)} matches, random sample of {limit}) ===")
            # seeded by the query so the same search always prints the same sample
            matches = sorted(random.Random(query).sample(matches, limit))
        else:
            click.echo(f"=== {query} ({len(matches)} matches) ===")
        for i, text_label in matches:
            click.echo(f"{i}\t{text_label}")


@main.command()
@click.argument("texts", nargs=-1)
@click.option("--audio", type=click.Path(exists=True, path_type=Path), default=None,
              help="retrieve for a sound instead of a description: one audio "
                   "file, or a directory read as one query per file in it")
@click.option("--top-k", type=int, default=None,
              help="return exactly this many per query; overrides --near-max")
@click.option("--near-max", type=float, default=0.9, show_default=True,
              help="return every instrument scoring at least this fraction of the "
                   "query's best score, so a query the corpus is full of returns "
                   "many and one with a clear winner returns one")
@click.option("--min-keys", type=int, default=1, show_default=True,
              help="ignore instruments whose mean covers fewer mapped keys than this")
@click.option("--fusion-only", is_flag=True, help="use the fusion checkpoint alone")
@click.option("--music-only", is_flag=True, help="use the music checkpoint alone")
@click.option("--device", default="cpu", show_default=True)
def retrieve_inst(texts: tuple, audio: Path, top_k: int, near_max: float,
              min_keys: int, fusion_only: bool, music_only: bool, device: str):
    """
    Find instruments matching free text or a sound, by embedding it with CLAP.

    Unlike find-label, which matches the author's own words, this scores the
    audio: an instrument labelled "Instr_07" can still be returned for
    "distorted electric guitar". With --audio the query is a sound instead, and
    a directory is read as one query per file rather than as one instrument.

    A directory is limited to a few dozen files, because CLAP embeds them in a
    single batch and the memory that takes is what binds. Nothing is written;
    use scripts/retrieve_instruments.py for plots, renders and a larger limit.
    """
    try:
        import laion_clap  # noqa: F401
    except ImportError:
        raise click.ClickException(
            "retrieve-inst needs laion-clap, which is in the simap-clap environment "
            "rather than this one"
        )
    from simap.metadata import Metadata
    from simap.retrieve_inst import retrieve

    config = get_user_config()
    missing = [k for k in ("embedding_dir", "laion_clap_dir", "simap_dir")
               if k not in config]
    if missing:
        raise click.ClickException(
            f"not configured: {', '.join(missing)}. Run `simap set-dir "
            f"<dataset|embedding|laion_clap> <dir>` for each."
        )
    if top_k is not None and top_k < 1:
        raise click.ClickException("--top-k must be at least 1")
    if bool(texts) == bool(audio):
        raise click.ClickException(
            "give either texts or --audio, not both and not neither")

    # max_files is left at the default rather than exposed: this command is for
    # a quick look, and the script is where a bigger search belongs
    got = retrieve(list(texts), config["embedding_dir"],
                   config["laion_clap_dir"], audio=audio,
                   fusion_only=fusion_only, music_only=music_only,
                   min_keys=min_keys, device=device)
    metadata = Metadata(Path(config["simap_dir"]) / "metadata.parquet")

    for column, label in enumerate(got.labels):
        order, sorted_scores, n_keep = got.select(column, near_max, top_k)
        click.echo(f"=== {label} ({n_keep} of {len(got.metadata_idx):,}) ===")
        for rank, i in enumerate(order[:n_keep], 1):
            idx = int(got.metadata_idx[i])
            entry = metadata[idx]
            click.echo(f"{rank}\t{sorted_scores[rank - 1]:+.3f}\t{idx}\t"
                       f"{entry['text_label']}\t{entry['file']}")
