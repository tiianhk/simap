"""
What loading each soundfont costs in resident memory, measured and cached.

ChunkedSimapDataset holds soundfonts in one long-lived process until a RAM budget
is reached, so what it needs to budget is exactly this: how much a process's
resident memory grows when it loads a given file. It is measured rather than
estimated, because the sample data is only part of it -- fluidsynth's own
structures, allocator overhead and fragmentation are what make a load cost more
than its samples, and file size is a poor proxy for the total.

gc.collect() and malloc_trim() before each load return freed memory to the OS, so
a file's measurement does not depend on which file preceded it.

How accurate this is
--------------------
Not very, per file. What makes the numbers usable is that a chunk
sums hundreds of files and that noise is independent, so the total is far steadier
than any one entry. Treat a single value as an estimate, treat a chunk total as a
budget, and leave ram_limit some headroom.

Sample rate does not matter.

The cache
---------
One flat file beside the user config, covering every file the metadata names:

    {hostname, measured_at, sample_rate, libfluidsynth, glibc, platform,
     python, n_files, total_gb, usage: {relative path: GB}}

Nothing is keyed on the provenance. Hostnames come and go with the scheduler, so
keying on one would force a re-measure on nearly every job; the corpus is the same
download for everyone; and the rate is immaterial. If the hardware or the
environment changes enough to matter, that is a judgement for the reader to make,
and `simap measure-memory` rebuilds the file whenever they make it.
"""

import ctypes
import gc
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import psutil
from platformdirs import user_config_dir
from tqdm import tqdm

from simap.synth_interface import load
from simap.user_config import get_user_config

CACHE_PATH = Path(user_config_dir("simap")) / "memory_usage.json"
SAMPLE_RATE = 16000     # only to build a synth with; the result does not depend on it


def _data_dir() -> Path:
    config = get_user_config()
    if "simap_dir" not in config:
        raise ValueError(
            "The dataset directory is not configured. "
            "Run `simap set-dir dataset <path>` to set the dataset directory."
        )
    return Path(config["simap_dir"])


def _files(data_dir: Path) -> list[str]:
    """Every soundfont the metadata names, once each, in metadata order."""
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    return list(dict.fromkeys(m["file"] for m in metadata))


def _libfluidsynth_version() -> str:
    try:
        fn = ctypes.CDLL("libfluidsynth.so.3").fluid_version_str
        fn.restype = ctypes.c_char_p
        return fn().decode()
    except Exception:
        return "unknown"


def measure(sample_rate: int = SAMPLE_RATE) -> dict:
    """
    Load every soundfont in the corpus, one at a time, and record what each costs.

    Takes no paths: the corpus is wherever `simap set-dir dataset` points, and the
    metadata there is the only list of files worth measuring.

    Every file the metadata names is known to load -- that was established when the
    metadata was built -- so a failure here means the copy on disk is damaged. The
    pass finishes before saying so, to name every broken file at once rather than
    one per attempt, and nothing is written: a cache that quietly omitted them would
    turn a damaged download into a slightly smaller corpus.
    """
    data_dir = _data_dir()
    files = _files(data_dir)
    print(f"measuring {len(files):,} soundfonts under {data_dir}", file=sys.stderr)

    process = psutil.Process()
    libc = ctypes.CDLL("libc.so.6")     # linux only, as is malloc_trim
    usage: dict[str, float] = {}
    failed: list[str] = []

    for file in tqdm(files, desc="memory usage", unit="file"):
        gc.collect()
        libc.malloc_trim(0)
        before = process.memory_info().rss
        try:
            synths = load(data_dir / file, sample_rate)
        except Exception:
            synths = None
        after = process.memory_info().rss
        if synths is None:
            failed.append(file)
            continue
        usage[file] = (after - before) / 1024**3
        for synth in synths:
            synth.delete()

    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(files):,} soundfonts under {data_dir} did not "
            f"load, e.g. {failed[:3]}. Every file the metadata names loaded when "
            f"the metadata was built, so this copy is incomplete or damaged; "
            f"nothing has been cached.")
    return {
        "hostname": socket.gethostname(),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_rate": sample_rate,
        "libfluidsynth": _libfluidsynth_version(),
        "glibc": ".".join(platform.libc_ver()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "n_files": len(usage),
        "total_gb": round(sum(usage.values()), 3),
        "usage": usage,
    }


def report(usage: dict[str, float]) -> None:
    values = sorted(usage.values())
    if not values:
        return
    print(f"{len(values):,} files, {sum(values):.1f} GB total, "
          f"min {values[0]:.4f} / median {median(values):.4f} / "
          f"max {values[-1]:.4f} GB")


def save(record: dict, path: Path = CACHE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def load_cache(path: Path = CACHE_PATH) -> dict | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_or_measure(sample_rate: int = SAMPLE_RATE,
                    path: Path = CACHE_PATH) -> dict[str, float]:
    """
    The cached cost of every soundfont, measuring the corpus if there is no cache.

    A cache that is present but does not cover everything the metadata names means
    the corpus has changed under it. That raises rather than quietly re-measuring,
    because re-measuring is minutes of work and a training run should not discover
    it halfway through: the reader can decide when to spend them.
    """
    record = load_cache(path)
    if record is None:
        record = measure(sample_rate)
        save(record, path)
        report(record["usage"])
        print(f"wrote {path}", file=sys.stderr)
        return record["usage"]

    usage = record["usage"]
    missing = [f for f in _files(_data_dir()) if f not in usage]
    if missing:
        raise ValueError(
            f"{path} covers {len(usage):,} soundfonts but {len(missing):,} named by "
            f"the metadata are absent, e.g. {missing[:3]}. The corpus has changed "
            f"since it was measured; run `simap measure-memory` to rebuild it.")
    return usage
