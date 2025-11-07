from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple
import zipfile

import pandas as pd


logger = logging.getLogger(__name__)


REQUIRED_FILES = ("dataset_schema.json", "annotations.jsonl")
REQUIRED_KEYS = ("from", "to", "text", "translated")


def _read_dataset_zip(zpath: Path) -> Iterable[dict]:
    """Yield records from a single Darvin dataset ZIP.

    Expects the ZIP root to contain dataset_schema.json and annotations.jsonl.
    Returns only rows that contain required keys (from/to/text/translated).
    """
    try:
        with zipfile.ZipFile(zpath, "r") as zf:
            names = set(zf.namelist())
            if not all(name in names for name in REQUIRED_FILES):
                raise ValueError(f"invalid_dataset_zip: missing required files in {zpath}")
            with zf.open("annotations.jsonl", "r") as fh:
                for raw in fh:
                    try:
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if not isinstance(obj, dict):
                            continue
                        if all(k in obj for k in REQUIRED_KEYS):
                            yield {
                                "from": obj["from"],
                                "to": obj["to"],
                                "text": obj["text"],
                                "translated": obj["translated"],
                            }
                    except Exception:
                        # skip invalid line
                        continue
    except Exception as exc:
        raise ValueError(f"failed_to_read_dataset_zip: {zpath}: {exc}")


def _iter_training_zip_paths_from_bundle(bundle_path: Path) -> List[Path]:
    """Extract training bundle structure and return inner dataset zip temp paths.

    The bundle contains files under dataset/training/*.zip. Extract each inner
    zip to a temporary location for reading.
    """
    tmp_dir = Path(os.getenv("DARVIN_WORK_DIR", "/tmp/workspace/train/.darvin_tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_paths: List[Path] = []
    with zipfile.ZipFile(bundle_path, "r") as outer:
        for name in outer.namelist():
            if not name.startswith("dataset/training/") or not name.lower().endswith(".zip"):
                continue
            # Persist inner zip to temp for further processing
            target = tmp_dir / Path(name).name
            try:
                with outer.open(name, "r") as src, target.open("wb") as dst:
                    dst.write(src.read())
                out_paths.append(target)
            except Exception:
                continue
    if not out_paths:
        raise ValueError("bundle_missing_training_zip")
    return out_paths


def _iter_dataset_zips_from_dir(ds_dir: Path) -> List[Path]:
    """Return all dataset zip files under a directory.

    Accept either:
    - raw dataset zips directly under ds_dir (e.g., *.zip), or
    - an extracted bundle dir that contains dataset/training/*.zip.
    """
    if not ds_dir.exists() or not ds_dir.is_dir():
        raise ValueError(f"dataset_dir_not_found: {ds_dir}")

    # Prefer dataset/training/*.zip if present
    training_dir = ds_dir / "dataset" / "training"
    if training_dir.exists():
        zips = sorted(training_dir.glob("*.zip"))
        if zips:
            return zips

    # Fallback: any *.zip directly under ds_dir
    zips = sorted([p for p in ds_dir.glob("*.zip") if p.is_file()])
    if not zips:
        raise ValueError(f"no_dataset_zip_found_in_dir: {ds_dir}")
    return zips


def load_dataframe(
    bundle_path: str | None = None,
    dataset_dir: str | None = None,
) -> pd.DataFrame:
    """Load Darvin-format datasets into a DataFrame with columns
    [from, to, text, translated].

    You may specify either a bundle zip path (preferred) or a directory that
    contains dataset zips or an extracted bundle.
    """
    zips: List[Path] = []
    if bundle_path:
        bp = Path(bundle_path)
        if not bp.exists():
            raise ValueError(f"bundle_not_found: {bp}")
        zips = _iter_training_zip_paths_from_bundle(bp)
    else:
        base = Path(dataset_dir or "")
        if not base:
            raise ValueError("dataset_source_missing")
        zips = _iter_dataset_zips_from_dir(base)

    rows: List[dict] = []
    for zp in zips:
        for obj in _read_dataset_zip(zp):
            rows.append(obj)

    if not rows:
        raise ValueError("annotations_empty")
    df = pd.DataFrame(rows, columns=["from", "to", "text", "translated"])
    # enforce string type to be consistent with downstream processing
    df = df.astype(str)
    return df

