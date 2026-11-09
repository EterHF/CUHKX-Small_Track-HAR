"""Run the frozen finalist from an organizer-supplied Testing directory."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml


def target_config(path: Path, submission_id: str | None) -> dict:
    """Resolve exactly one Selected target; never fall back for an unknown ID."""
    config = yaml.safe_load(path.read_text())
    primary_id = str(config["submission_id"])
    requested = primary_id if submission_id in {None, "primary"} else submission_id
    if requested != primary_id:
        try:
            config = config["alternate_submissions"][requested]
        except KeyError as error:
            raise ValueError(f"Unknown Selected submission ID: {requested}") from error
    if str(config["submission_id"]) != requested:
        raise ValueError("Submission ID and checkpoint configuration disagree")
    return config


def test_files(data_dir: Path) -> tuple[Path, Path, Path]:
    """Accept Testing itself, the track root, or the documented local data root."""
    for root in (
        data_dir,
        data_dir / "Testing",
        data_dir / "Small-Model-Track/Testing",
    ):
        csv_path = root / "test_file/test.csv"
        sample = root / "test_file/sample_submission.csv"
        if csv_path.is_file() and sample.is_file():
            return root, csv_path, sample
    raise FileNotFoundError(
        "Expected Testing/test_file/{test.csv,sample_submission.csv} under data_dir "
        "(data_dir may also be Testing itself)."
    )


def sample_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        key = "path" if "path" in fields else "id" if "id" in fields else None
        if key is None:
            raise ValueError(f"Missing path/id column: {path}")
        ids = [row[key].rstrip("/").split("/")[-1] for row in reader]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate sample identifiers: {path}")
    if any(item in {"", ".", ".."} or "\\" in item for item in ids):
        raise ValueError(f"Invalid sample identifier: {path}")
    return ids


def raw_root(data_dir: Path, testing: Path, temporary: Path) -> Path:
    for root in (
        testing / "data/small_model_track_test",
        testing / "small_model_track_test",
        data_dir / "processed/test/small_model_track_test",
    ):
        if root.is_dir():
            return root
    archive = testing / "data/small_model_track_test.zip"
    if not archive.is_file():
        raise FileNotFoundError(
            "Expected data/small_model_track_test/ or data/small_model_track_test.zip "
            "inside Testing. Extracted data are also accepted at "
            "data_dir/processed/test/small_model_track_test/."
        )
    extracted = temporary / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (extracted / member.filename).resolve()
            if not target.is_relative_to(extracted.resolve()):
                raise ValueError(f"Archive member escapes extraction directory: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symbolic links are not supported inside the data archive")
        handle.extractall(extracted)
    result = extracted / "small_model_track_test"
    if not result.is_dir():
        raise FileNotFoundError("Archive must contain small_model_track_test/")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("submission_id", nargs="?", help="Selected ID; defaults to primary")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = target_config(args.config, args.submission_id)
    bundle = args.config.resolve().parent / config["bundle"]
    if bundle.stat().st_size != config["bundle_bytes"]:
        raise ValueError("Final checkpoint size mismatch")
    data_dir = args.data_dir.resolve(strict=True)
    testing, test_csv, sample_csv = test_files(data_dir)
    ids = sample_ids(test_csv)
    if ids != sample_ids(sample_csv):
        raise ValueError("test.csv and sample_submission.csv sample order differs")
    output = args.output_csv.resolve()
    if output.is_relative_to(data_dir):
        raise ValueError("Choose output_csv outside the read-only input data directory")
    with tempfile.TemporaryDirectory(prefix="cuhkx_input_") as directory:
        root = raw_root(data_dir, testing, Path(directory))
        missing = [item for item in ids if not (root / item).is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing sample directories: {missing[:5]}")
        print(json.dumps({
            "submission_id": config["submission_id"],
            "samples": len(ids),
            "test_root": str(root),
            "output_csv": str(output),
        }), flush=True)
        if args.validate_only:
            return
        subprocess.run([
            sys.executable, "-m", "visual_temporal.release.replay",
            "--bundle", str(bundle),
            "--test-root", str(root),
            "--test-csv", str(test_csv),
            "--sample-submission", str(sample_csv),
            "--output", str(output),
            "--device", os.environ.get("CUHKX_DEVICE", "cuda:0"),
            "--batch-size", str(config["batch_size"]),
            "--detector-batch-size", str(config["detector_batch_size"]),
            "--workers", str(config["workers"]),
        ], check=True)


if __name__ == "__main__":
    main()
