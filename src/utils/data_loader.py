"""Prepare configured artifacts before importing model-backed routes."""
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse

import gdown
import requests

from src.utils.paths import DRUGBANK_DATA_DIR, MODEL_CHECKPOINT


def is_google_drive_url(url):
    return urlparse(url).hostname == "drive.google.com"


def download_file(url, dest_path):
    destination = Path(dest_path)
    if destination.is_file() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".part",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        if is_google_drive_url(url):
            result = gdown.download(url, str(temporary), quiet=True)
            if result is None:
                raise RuntimeError(
                    "Google Drive did not return a downloadable file"
                )
        else:
            with requests.get(
                url, stream=True, timeout=(15, 300)
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        output.write(chunk)

        if temporary.stat().st_size == 0:
            raise RuntimeError("Download returned an empty file")

        with temporary.open("rb") as downloaded:
            prefix = downloaded.read(512).lstrip().lower()

        if prefix.startswith((b"<!doctype html", b"<html")):
            raise RuntimeError(
                "Download returned an HTML page instead of an artifact"
            )

        temporary.replace(destination)
        print(f"[INFO] Downloaded {destination.name}")
    except Exception as error:
        raise RuntimeError(
            f"Could not prepare {destination.name}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def ensure_required_files():
    if os.getenv("ENABLE_PREDICTIONS", "false").lower() != "true":
        return

    model_url = os.getenv("MODEL_URL")

    if not (
        MODEL_CHECKPOINT.is_file()
        and MODEL_CHECKPOINT.stat().st_size > 0
    ):
        if not model_url:
            raise RuntimeError(
                "Missing model checkpoint. Set MODEL_URL to a trusted "
                "downloadable checkpoint or provide models/rgcn_v2.pt."
            )
        download_file(model_url, MODEL_CHECKPOINT)

    for variable, filename in (
        ("DRUGBANK_JSON_URL", "drugbank_slim.json"),
        ("DRUGBANK_XML_URL", "drugbank_all_drug_links.xml"),
    ):
        url = os.getenv(variable)
        if url:
            download_file(url, DRUGBANK_DATA_DIR / filename)