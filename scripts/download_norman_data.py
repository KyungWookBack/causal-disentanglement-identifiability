#!/usr/bin/env python3
"""Download the scPerturb Norman data used in the report from Zenodo."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from experiment_utils import file_sha256

ZENODO_RECORD = 7041849
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
FILENAME = "NormanWeissman2019_filtered.h5ad"


def fetch_metadata() -> dict:
    with urllib.request.urlopen(ZENODO_API, timeout=60) as response:
        return json.load(response)


def check_file(path: Path, entry: dict) -> None:
    if path.stat().st_size != entry["size"]:
        raise ValueError(f"Unexpected file size: {path}")
    algorithm, expected = entry["checksum"].split(":", 1)
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("external_data"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    metadata = fetch_metadata()
    entry = next(item for item in metadata["files"] if item["key"] == FILENAME)
    source = {
        "record_id": ZENODO_RECORD,
        "url": f"https://zenodo.org/records/{ZENODO_RECORD}",
        "doi": metadata["metadata"]["doi"],
        "title": metadata["metadata"]["title"],
        "license": metadata["metadata"]["license"]["id"],
        "filename": FILENAME,
        "size_bytes": entry["size"],
        "checksum": entry["checksum"],
        "download_url": entry["links"]["self"],
    }
    if args.metadata_only:
        print(json.dumps(source, indent=2))
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    destination = args.outdir / FILENAME
    if not destination.exists() or args.overwrite:
        temporary = destination.with_suffix(".h5ad.part")
        print(f"Downloading {FILENAME} ({entry['size'] / 1e6:.0f} MB)...", flush=True)
        with urllib.request.urlopen(entry["links"]["self"], timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        check_file(temporary, entry)
        temporary.replace(destination)
    else:
        check_file(destination, entry)

    source["sha256"] = file_sha256(destination)
    destination.with_suffix(".source.json").write_text(json.dumps(source, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
