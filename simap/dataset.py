from collections import deque, defaultdict
from pathlib import Path
import json
import shutil
import time
import zipfile
import hashlib
from multiprocessing import Process, Queue
import sys

import numpy as np
import requests
from tqdm import tqdm
import rarfile
import py7zr

from simap.synth_interface import load, SynthInterface
from simap.keymap import write_keymap
from simap.sample_reader import read_samples
from simap.metadata import write_metadata_parquet
from simap.constants import (
    START_ARTIFACT_ID,
    END_ARTIFACT_ID,
    KEY_NUM,
    LICENSES,
    FORMATS,
    ARCHIVE_SUFFIXES,
    INSTRUMENT_SUFFIXES,
)


""" download """


def download(simap_dev_dir: Path, max_retries: int) -> int:

    download_ids = [str(i) for i in range(START_ARTIFACT_ID, END_ARTIFACT_ID + 1)]
    existed = {p.name for p in simap_dev_dir.iterdir() if p.is_dir()}

    queue = deque(download_ids)
    retry_counts = {aid: 0 for aid in download_ids}

    try:
        with open(simap_dev_dir / "authors_and_licenses.json", "r") as f:
            authors_and_licenses = json.load(f)
    except:
        authors_and_licenses = {}

    new_downloads = 0
    key_errors = 0
    assertion_errors = 0
    client_errors = 0
    server_errors_after_retry = 0
    other_errors = 0

    with (
        requests.Session() as session,
        tqdm(total=len(download_ids), desc="Download") as pbar,
    ):
        while queue:
            artifact_id = queue.popleft()

            if artifact_id in existed:
                pbar.update(1)
                continue

            try:
                url = f"https://musical-artifacts.com/artifacts/{artifact_id}.json"
                response = session.get(url)
                response.raise_for_status()
                data = response.json()

                assert data["license"] in LICENSES, "License"
                if artifact_id not in authors_and_licenses:
                    authors_and_licenses[artifact_id] = {
                        "author": data["author"],
                        "license": data["license"],
                    }
                suffix = Path(data["file"]).suffix.lower()
                if suffix in ARCHIVE_SUFFIXES:
                    assert set(data["formats"]) & set(FORMATS), "Format"
                else:
                    assert suffix in INSTRUMENT_SUFFIXES, "Suffix"

                time.sleep(1)
                response = session.get(data["file"])
                response.raise_for_status()

                path = Path(data["file"])
                path = simap_dev_dir / artifact_id / path.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(response.content)
                new_downloads += 1
                pbar.update(1)

                if retry_counts[artifact_id] > 0:
                    print(
                        f"Successfully downloaded {artifact_id} after {retry_counts[artifact_id]} retries"
                    )

            except KeyError as e:
                path = simap_dev_dir / artifact_id / f"KeyError:{e.args[0]}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                pbar.update(1)
                key_errors += 1
                continue

            except AssertionError as e:
                path = simap_dev_dir / artifact_id / f"AssertionError:{e.args[0]}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                pbar.update(1)
                assertion_errors += 1
                continue

            except requests.exceptions.HTTPError as e:
                code = e.response.status_code
                if code == 429:
                    print(
                        f"Rate limit exceeded. Consider sleep longer between requests."
                    )
                    sys.exit(1)
                if 400 <= code < 500:  # client errors
                    path = simap_dev_dir / artifact_id / f"ClientError:{code}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    pbar.update(1)
                    client_errors += 1
                    continue
                if 500 <= code < 600:  # server errors (will retry)
                    retry_counts[artifact_id] += 1
                    if retry_counts[artifact_id] == max_retries:
                        pbar.update(1)
                        server_errors_after_retry += 1
                        print(
                            f"Failed to download {artifact_id} after {max_retries} attempts"
                        )
                        continue
                    queue.append(artifact_id)  # put back to the end of the queue
                    print(f"Error downloading {artifact_id}: {e}, will retry")

            except Exception as e:
                path = (
                    simap_dev_dir / artifact_id / f"OtherError:{e.__class__.__name__}"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                pbar.update(1)
                other_errors += 1
                print(f"Error downloading {artifact_id}: {e}")

            finally:
                time.sleep(1)

    with open(simap_dev_dir / "authors_and_licenses.json", "w") as f:
        json.dump(authors_and_licenses, f, indent=2)

    print(f"Existed: {len(existed)}")
    print(f"New downloads: {new_downloads}")
    print(f"Key errors: {key_errors}")
    print(f"Assertion errors: {assertion_errors}")
    print(f"Client errors: {client_errors}")
    print(f"Other errors: {other_errors}")

    return server_errors_after_retry


""" unpack """


def get_archive_files(path):
    return [
        f
        for f in Path(path).rglob("*")
        if f.is_file() and f.suffix.lower() in ARCHIVE_SUFFIXES
    ]


def extract_archive(archive_file):
    out_dir = archive_file.parent / archive_file.stem

    try:
        out_dir.mkdir(parents=True)
    except:  # skip if the dir already exists
        return

    try:
        if zipfile.is_zipfile(archive_file):
            with zipfile.ZipFile(archive_file) as zf:
                zf.extractall(out_dir)
        elif rarfile.is_rarfile(archive_file):
            with rarfile.RarFile(archive_file) as rf:
                rf.extractall(out_dir)
        elif archive_file.suffix.lower() == ".7z":
            with py7zr.SevenZipFile(archive_file, mode="r") as sz:
                sz.extractall(out_dir)

        nested = get_archive_files(out_dir)
        if len(nested) > 0:
            for file in nested:
                extract_archive(file)

    except Exception as e:
        print(f"Error extracting archive {archive_file}: {e}")


def unpack(simap_dev_dir: Path):
    files = get_archive_files(simap_dev_dir)
    for file in tqdm(files, desc="Unpack"):
        extract_archive(file)


""" deduplicate """


# File-level dedup is here, inst-level dedup is in the next section
# Deduplication functions (chunk_reader, get_hash, deduplicate) adapted from:
# https://gist.github.com/tfeldmann/fc875e6630d11f2256e746f67a09c1ae


def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes."""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def get_hash(filepath, first_chunk_only=False, hash_algo=hashlib.sha1):
    """Compute hash of file (optionally only first 1024 bytes)."""
    hashobj = hash_algo()
    with filepath.open("rb") as f:
        if first_chunk_only:
            hashobj.update(f.read(1024))
        else:
            for chunk in chunk_reader(f):
                hashobj.update(chunk)
    return hashobj.digest()


def deduplicate(simap_dev_dir: Path):
    files_by_size = defaultdict(list)
    files_by_small_hash = defaultdict(list)
    files_by_full_hash = defaultdict(list)

    for file_path in simap_dev_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in INSTRUMENT_SUFFIXES:
            continue
        try:
            file_path = file_path.resolve()
            file_size = file_path.stat().st_size
        except Exception as e:
            print(f"failed to resolve or get size for {file_path}: {e}")
            continue
        files_by_size[file_size].append(file_path)

    # For all files with the same file size, get their hash on the first 1024 bytes
    for file_size, files in files_by_size.items():
        if len(files) < 2:
            continue
        for file_path in files:
            try:
                small_hash = get_hash(file_path, first_chunk_only=True)
            except Exception as e:
                print(f"failed to get hash for {file_path}: {e}")
                continue
            files_by_small_hash[(file_size, small_hash)].append(file_path)

    # For files with the same small hash, check full file hash
    for files in files_by_small_hash.values():
        if len(files) < 2:
            continue
        for file_path in files:
            try:
                full_hash = get_hash(file_path, first_chunk_only=False)
            except Exception as e:
                print(f"failed to get hash for {file_path}: {e}")
                continue
            files_by_full_hash[full_hash].append(file_path)

    # Keep one duplicate and rename the others
    group_cnt = 0
    for files in files_by_full_hash.values():
        if len(files) < 2:
            continue
        files = sorted(
            files, key=lambda p: (int(p.relative_to(simap_dev_dir.resolve()).parts[0]), str(p))
        )  # keep the lowest artifact id, so new artifacts never evict existing files
        print(f"Duplicate group / {group_cnt}:")
        print(f" - {files[0]}")
        for file_path in files[1:]:
            new_path = file_path.with_name(file_path.name + ".duplicate")
            try:
                file_path.rename(new_path)
                print(f" - {new_path} (renamed)")
            except Exception as e:
                print(f" - {file_path} (failed to rename: {e})")
        print("\n")
        group_cnt += 1
    if group_cnt == 0:
        print("No duplicates!")


""" scan """


def synth_params(layer: dict):
    if layer["gens"] is None:  # sfz
        opcodes = layer["opcodes"]
        return None, None, None if opcodes is None else tuple(sorted(opcodes.items()))
    # sf2/3
    mods = tuple(
        sorted(
            (m["dest"], m["src1"], m["flags1"], m["src2"], m["flags2"],
             m["trans"], m["amount"])
            for m in layer["mods"]
        )
    )
    return tuple(sorted(layer["gens"].items())), mods, None


def get_keymap(synth: SynthInterface, channel: int, samples: dict) -> dict:
    """
    Record every layer that sounds on every key, concatenated in key order.

    samples is what sample_reader found in the file, keyed as a layer reports
    its sample_id, so each sample this instrument reaches can be described
    without going back to the file.
    """
    n_layers = np.zeros(KEY_NUM, dtype=np.uint16)
    shift_semitones: list[float] = []
    loopmode: list[float] = []
    lovel: list[int] = []
    hivel: list[int] = []
    sample_idx: list[int] = []
    sample_ids: dict = {}
    params: list = []
    unique_params: dict = {}
    used: list[dict] = []  # one per sample_idx, in the order they first sound

    for key in range(KEY_NUM):
        layers = synth.get_layers(channel, key)
        if len(layers) > np.iinfo(np.uint16).max:
            raise ValueError(f"key {key} has {len(layers)} layers, too many for uint16")
        n_layers[key] = len(layers)
        for layer in layers:
            sample_id = layer["sample_id"]
            if sample_id not in sample_ids:
                sample_ids[sample_id] = len(sample_ids)
                used.append(samples[sample_id])
            sample_idx.append(sample_ids[sample_id])
            shift_semitones.append(layer["pitch_shift_semitones"])
            loopmode.append(layer["loopmode"])
            lovel.append(layer["lovel"])
            hivel.append(layer["hivel"])
            k = synth_params(layer)
            params.append(unique_params.setdefault(k, k))

    if len(sample_ids) > np.iinfo(np.uint16).max:
        raise ValueError(f"{len(sample_ids)} distinct samples, too many for uint16")

    return {
        "n_layers": n_layers,
        "shift_semitones": np.array(shift_semitones, dtype=np.float32),
        "loopmode": np.array(loopmode, dtype=np.float32),
        "lovel": np.array(lovel, dtype=np.uint8),
        "hivel": np.array(hivel, dtype=np.uint8),
        "sample_idx": np.array(sample_idx, dtype=np.uint16),
        "synth_params": params,
        "samples": used,
    }


def file_worker(
    worker_id: int,
    simap_dev_dir: Path,
    sample_rate: int,
    file_queue: Queue,
    result_queue: Queue,
):
    """
    if there are files in the file_queue
    get one file from the queue
    load synths from the file
    retrieve information about the synths
    add a (metadata entry, keymap) pair per instrument to the result queue
    """
    while True:
        file = file_queue.get()  # get one file from the file queue
        if file is None:
            break  # None is the finishing signal, so break the loop
        synths = load(file, sample_rate)
        if synths is None:
            result_queue.put(
                "file_done"
            )  # add a "file_done" signal to the result queue
            continue  # continue to the next file
        rel = file.relative_to(simap_dev_dir)
        samples = {
            s["sample_id"]: s
            for s in read_samples(file, root=simap_dev_dir / rel.parts[0])
        }
        for i, synth in enumerate(synths):
            for j in range(synth.channel_num):
                keymap = get_keymap(synth, j, samples)
                if keymap["n_layers"].sum() == 0:  # skip instruments with zero mapped keys
                    continue
                bank, preset = synth.get_bank_preset(j)
                entry = {
                    "url": f"https://musical-artifacts.com/artifacts/{rel.parts[0]}",
                    "author": None,  # placeholder, will be filled in later
                    "license": None,  # same
                    "file": str(rel),
                    "synth_id": i,
                    "channel_id": j,
                    "bank": bank,
                    "preset": preset,
                    "text_label": synth.get_text_label(j),
                }
                # paired, so scan() can keep the two aligned when it sorts
                result_queue.put((entry, keymap))
            synth.delete()
        result_queue.put("file_done")
    result_queue.put(None)


def inst_level_dedup(metadata: list[dict], keymap: list[dict]):
    sample_ids: dict = {}
    param_ids: dict = {}
    per_inst_rows = []
    for km in keymap:
        codes = np.array(
            [sample_ids.setdefault(
                (s["sha"], s["samplerate"], s["loopstart"], s["loopend"]),
                len(sample_ids))
             for s in km["samples"]], dtype=np.int64)
        silent = np.array([s["sha"] is None for s in km["samples"]])
        # synth params are interned across the scan, so two layers synthesised
        # alike are one object and identity is as good as equality, and cheaper
        params = np.fromiter(
            (param_ids.setdefault(id(k), len(param_ids))
             for k in km["synth_params"]), dtype=np.int64,
            count=len(km["synth_params"]))
        per_inst_rows.append((codes[km["sample_idx"]], params,
                              silent[km["sample_idx"]]))

    audible = np.concatenate([r[2] for r in per_inst_rows]) == False  # noqa: E712
    rows = np.column_stack([
        np.concatenate([r[0] for r in per_inst_rows]).astype(np.float64),
        np.concatenate([r[1] for r in per_inst_rows]).astype(np.float64),
        np.concatenate([km["shift_semitones"] for km in keymap]).astype(np.float64),
        np.concatenate([km["loopmode"] for km in keymap]).astype(np.float64),
        np.concatenate([km["lovel"] for km in keymap]).astype(np.float64),
        np.concatenate([km["hivel"] for km in keymap]).astype(np.float64),
    ])[audible]
    # loopmode is nan for sfz, and nan does not equal itself, so a column
    # holding one would make every layer in it look distinct
    rows = np.nan_to_num(rows, nan=-1e18)
    _, layer_id = np.unique(rows, axis=0, return_inverse=True)
    del rows
    layer_id = np.asarray(layer_id).ravel()

    # which key each surviving layer is on, and which instrument
    key = np.concatenate([np.repeat(np.arange(KEY_NUM, dtype=np.int64),
                                    km["n_layers"]) for km in keymap])[audible]
    inst = np.concatenate([
        np.full(int(km["n_layers"].sum()), i, dtype=np.int64)
        for i, km in enumerate(keymap)])[audible]

    change = (inst[1:] != inst[:-1]) | (key[1:] != key[:-1])
    bounds = np.flatnonzero(np.r_[True, change, True])
    lid = layer_id.tolist()
    seen: dict = {}
    keys_of: dict = defaultdict(list)
    for n in range(len(bounds) - 1):
        lo = bounds[n]
        sig = (int(key[lo]), tuple(sorted(lid[lo:bounds[n + 1]])))
        keys_of[int(inst[lo])].append((sig[0], seen.setdefault(sig, len(seen))))
    del seen, lid

    groups: dict = defaultdict(list)
    for i in range(len(keymap)):
        groups[tuple(sorted(keys_of.get(i, [])))].append(i)
    kept = sorted(min(g) for g in groups.values())
    print(f"Instrument level dedup: {len(keymap)} -> {len(kept)}, "
          f"{len(keymap) - len(kept)} removed")
    return [metadata[i] for i in kept], [keymap[i] for i in kept]


def scan(
    simap_dev_dir: Path,
    num_workers: int,
    sample_rate: int = 16000,
):
    file_queue: Queue = Queue()
    result_queue: Queue = Queue()
    workers: list[Process] = []

    for worker_id in range(num_workers):
        p = Process(
            target=file_worker,
            args=(worker_id, simap_dev_dir, sample_rate, file_queue, result_queue),
        )
        p.start()  # start the worker
        workers.append(p)

    files = [
        file
        for file in simap_dev_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in INSTRUMENT_SUFFIXES
    ]
    for file in files:
        file_queue.put(file)  # add files to the file queue
    for _ in range(num_workers):
        file_queue.put(None)  # add finishing signals for each worker

    rows = []  # (metadata entry, keymap) pairs
    unique_params: dict = {}
    finished_worker = 0
    with tqdm(total=len(files), desc="Scan") as pbar:
        while finished_worker < num_workers:
            result = result_queue.get()  # collect results from the result queue
            if result is None:  # worker done signal
                finished_worker += 1
            elif result == "file_done":
                pbar.update(1)
            else:
                keymap = result[1]
                keymap["synth_params"] = [
                    unique_params.setdefault(k, k) for k in keymap["synth_params"]
                ]
                rows.append(result)

    for p in workers:
        p.join()  # wait for the workers to finish

    rows.sort(
        key=lambda r: (
            int(r[0]["file"].split("/")[0]),
            r[0]["file"],
            r[0]["synth_id"],
            r[0]["channel_id"],
        )
    )  # sort the pairs, by artifact id numerically so new artifacts append at the end
    metadata = [row[0] for row in rows]
    keymap = [row[1] for row in rows]  # sorted as one, so keymap[i] describes metadata[i]

    with open(simap_dev_dir / "authors_and_licenses.json", "r") as f:
        authors_and_licenses = json.load(f)
    for m in metadata:
        artifact_id = m["file"].split("/")[0]
        m["author"] = authors_and_licenses[artifact_id]["author"]
        m["license"] = authors_and_licenses[artifact_id]["license"]
        assert m["license"] is not None

    metadata, keymap = inst_level_dedup(metadata, keymap)

    out_path = simap_dev_dir / "metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)  # write the metadata

    keymap_path = simap_dev_dir / "keymap.parquet"
    write_keymap(keymap, keymap_path)

    layer_num = sum(int(km["n_layers"].sum()) for km in keymap)
    print(
        f"Found {len(metadata)} synths from {len(files)} files, metadata saved to {out_path}"
    )
    print(f"Wrote {layer_num} keymap layers to {keymap_path}")


""" export """


def export(simap_dev_dir: Path, simap_dir: Path):
    """
    Export dataset from simap_dev/ to simap/ for minimum redundancy.
    Copies only artifact folders that appear in metadata, excluding .duplicate
    files and archive files (.zip, .rar, .7z).

    The keymap is positionally aligned to the metadata, so both are copied from the
    same scan; taking them from different scans would silently misalign them.
    """
    if simap_dir.exists():
        shutil.rmtree(simap_dir)
    simap_dir.mkdir(parents=True)

    with open(simap_dev_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    artifact_ids = {m["file"].split("/")[0] for m in metadata}
    for artifact_id in tqdm(artifact_ids, desc="Export"):
        src_dir = simap_dev_dir / artifact_id
        dst_dir = simap_dir / artifact_id
        for src_file in src_dir.rglob("*"):
            if not src_file.is_file():
                continue
            if src_file.suffix.lower() in ARCHIVE_SUFFIXES + [".duplicate"]:
                continue
            rel = src_file.relative_to(src_dir)
            dst_file = dst_dir / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

    shutil.copy2(simap_dev_dir / "metadata.json", simap_dir / "metadata.json")
    write_metadata_parquet(metadata, simap_dir / "metadata.parquet")
    shutil.copy2(simap_dev_dir / "keymap.parquet", simap_dir / "keymap.parquet")
    print(f"Dataset exported to {simap_dir}")
