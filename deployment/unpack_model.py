"""Verify and unpack the evaluated BioKG serving artifacts during image builds."""

import hashlib
import io
from pathlib import Path
import sys
import zipfile


def unpack(destination):
    directory = Path(__file__).resolve().parent
    bundle = b"".join(
        (directory / f"biokg-serving.zip.part-{part:02d}").read_bytes()
        for part in range(2)
    )
    expected = "2eb61258a172f0315359a45dcd9c0fd17722e06239c8d7af17667fd8107034e9"
    if hashlib.sha256(bundle).hexdigest() != expected:
        raise ValueError("BioKG serving archive does not match the evaluated release")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        for filename in ("rgcn_v2.pt", "pgexplainer.pt"):
            (destination / filename).write_bytes(archive.read(filename))


if __name__ == "__main__":
    unpack(Path(sys.argv[1]))
