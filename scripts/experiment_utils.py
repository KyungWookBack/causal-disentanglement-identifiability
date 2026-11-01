"""Record the settings and source versions used for a numerical run."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_config(path: Path, settings: dict, notes: str) -> None:
    """Write run settings, package versions, and SHA-256 digests of the source files to JSON."""
    scripts = Path(__file__).resolve().parent
    versions = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "scikit-learn", "anndata"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = Path(sys.argv[0]).name
    relevant = {entry, Path(__file__).name}
    if entry == "run_norman_realdata_experiment.py":
        relevant.add("direction_magnitude_decomposition.py")
    payload = {
        "settings": {key: value for key, value in settings.items() if key != "outdir"},
        "notes": notes,
        "python": platform.python_version(),
        "packages": versions,
        "source_sha256": {name: file_sha256(scripts / name) for name in sorted(relevant)
                          if (scripts / name).is_file()},
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
