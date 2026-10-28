"""
Read the audio samples a file uses, from the file rather than a synth object.

That key is what synth_interface reports as a layer's sample_id:

    sf2, sf3   (start, end) from the sample header, the span fluidsynth
               preserves as source_start and source_end
    sfz        the resolved path of the audio file

Each sample is identified by sha, a hash of its audio, which is what makes
two recordings comparable across files and formats. The audio itself is only
returned when asked for.
"""

import hashlib
import io
import mmap
import struct
from pathlib import Path

import numpy as np
import soundfile as sf

SHDR_SIZE = 46
HASH = hashlib.sha1  # the audio identifier
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".w64", ".mp3"}
SAMPLETYPE_OGG_VORBIS = 0x10
SAMPLETYPE_ROM = 0x8000


def _record(sample_id, name=None, sha=None, pcm=None, samplerate=None,
            channels=None, frame_num=None, loopstart=None, loopend=None,
            origpitch=None, pitchadj=None, sampletype=None):
    """Template for a single sample."""
    return {
        "sample_id": sample_id,
        "name": sample_id if name is None else name,
        "sha": sha,
        "pcm": pcm,
        "samplerate": samplerate,
        "channels": channels,
        "frame_num": frame_num,
        "loopstart": loopstart,
        "loopend": loopend,
        "origpitch": origpitch,
        "pitchadj": pitchadj,
        "sampletype": sampletype,
    }


def _chunks(buf, start, end):
    """Walk a riff chunk list, yielding (id, offset of its body, size)."""
    i = start
    while i + 8 <= end:
        cid = buf[i : i + 4].decode("latin-1")
        size = struct.unpack_from("<I", buf, i + 4)[0]
        yield cid, i + 8, size
        i += 8 + size + (size & 1)  # chunks are word aligned


def _soundfont_chunks(buf):
    """(smpl, shdr, version) of a soundfont, each as (offset, size)."""
    smpl = shdr = version = None
    total = 8 + struct.unpack_from("<I", buf, 4)[0]
    for cid, off, size in _chunks(buf, 12, total):
        if cid != "LIST":
            continue
        kind = buf[off : off + 4].decode("latin-1")
        for c, o, s in _chunks(buf, off + 4, off + size):
            if kind == "sdta" and c == "smpl":
                smpl = (o, s)
            elif kind == "pdta" and c == "shdr":
                shdr = (o, s)
            elif kind == "INFO" and c == "ifil":
                version = struct.unpack_from("<HH", buf, o)
    return smpl, shdr, version


def _read_soundfont(path, with_pcm):
    with open(path, "rb") as fh:
        buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    if buf[:4] != b"RIFF" or buf[8:12] != b"sfbk":
        raise ValueError(f"{path} is not a soundfont")
    smpl, shdr, version = _soundfont_chunks(buf)
    if shdr is None:
        raise ValueError(f"{path} has no sample header chunk")
    # sf3 holds each sample as its own ogg stream rather than as pcm, and the
    # extension does not say which it is: one file in simap is sf3 inside a
    # name ending .sf2, so the version in the header is what decides
    sf3 = version is not None and version[0] == 3

    offset, size = shdr
    for i in range(size // SHDR_SIZE - 1):  # the last record is a terminator
        base = offset + i * SHDR_SIZE
        name = buf[base : base + 20].split(b"\0")[0].decode("latin-1", "replace")
        start, end, loopstart, loopend, samplerate = struct.unpack_from(
            "<IIIII", buf, base + 20
        )
        origpitch, pitchadj = struct.unpack_from("<Bb", buf, base + 40)
        sampletype = struct.unpack_from("<H", buf, base + 44)[0]

        pcm = b""
        vorbis = bool(sampletype & SAMPLETYPE_OGG_VORBIS) or sf3
        if smpl and end > start and not sampletype & SAMPLETYPE_ROM:
            if vorbis:
                # for sf3 the span is in bytes, and each one is a whole stream
                blob = buf[smpl[0] + start : smpl[0] + end]
                pcm = sf.read(io.BytesIO(blob), dtype="int16")[0].tobytes()
            else:
                pcm = bytes(buf[smpl[0] + start * 2 : smpl[0] + end * 2])

        # a vorbis sample's loop is already relative to the sample, while a
        # pcm one is relative to the whole chunk (fluid_defsfont.c:335)
        if vorbis:
            lo, hi = loopstart, loopend
        else:
            lo, hi = loopstart - start, loopend - start

        # end is one past the last frame here, but fluidsynth stores the
        # last frame, so drop one to match what a layer reports
        yield _record(
            (start, end - 1),
            name=name,
            sha=HASH(pcm).hexdigest() if pcm else None,
            pcm=pcm if with_pcm else None,
            samplerate=samplerate,
            channels=1,  # an sf2 sample header describes one channel
            frame_num=len(pcm) // 2,
            loopstart=lo,
            loopend=hi,
            origpitch=origpitch,
            pitchadj=pitchadj,
            sampletype=sampletype,
        )


def _read_sfz(path, with_pcm, root):
    """
    For simplicity, we keep a record of every audio file in the folder an sfz belongs to, 
    which forms a superset of what it actually use. 
    """
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        # as metadata names a file: the artifact and the path within it,
        # which is unique across simap where a bare path would not be
        name = f"{root.name}/{file.relative_to(root)}"
        try:
            audio, samplerate = sf.read(file, dtype="int16", always_2d=True)
        except Exception:
            # not audio after all, despite the suffix
            yield _record(file.resolve(), name=name)
            continue
        pcm = audio.tobytes()
        yield _record(
            file.resolve(),
            name=name,
            sha=HASH(pcm).hexdigest() if pcm else None,
            pcm=pcm if with_pcm else None,
            samplerate=samplerate,
            channels=audio.shape[1],
            frame_num=len(audio),
        )

    for osc in ("*sine", "*saw", "*triangle", "*square", "*silence"):
        yield _record(osc, sha=osc)  # sfizz generates these; used directly as identities

    yield _record(None)  # broken sfz files, measured n=3 in simap


def read_samples(path, with_pcm=False, root=None):
    """Yield every sample a file holds."""
    path = Path(path)
    if path.suffix.lower() == ".sfz":
        yield from _read_sfz(path, with_pcm, Path(root) if root else path.parent)
    else:
        yield from _read_soundfont(path, with_pcm)
