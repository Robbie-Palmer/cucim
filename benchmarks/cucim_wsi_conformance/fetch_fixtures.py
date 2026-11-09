# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fetch, generate, and verify the public WSI conformance fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from generate_anisotropic_tiff import generate

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "fixtures.json"
DEFAULT_DESTINATION = HERE / "data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, member: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError(f"fixture is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != member["bytes"]:
        raise RuntimeError(
            f"{path}: expected {member['bytes']} bytes, found {actual_size}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != member["sha256"]:
        raise RuntimeError(
            f"{path}: expected SHA-256 {member['sha256']}, found {actual_sha256}"
        )


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "cucim-conformance/1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def _extract_selected(
    archive_path: Path, destination: Path, members: list[dict[str, Any]]
) -> None:
    requested = {member["path"] for member in members}
    with tarfile.open(archive_path, "r:*") as archive:
        available = {item.name: item for item in archive.getmembers()}
        missing = requested - available.keys()
        if missing:
            raise RuntimeError(f"archive is missing members: {sorted(missing)}")
        for name in sorted(requested):
            item = available[name]
            if (
                not item.isfile()
                or Path(name).is_absolute()
                or ".." in Path(name).parts
            ):
                raise RuntimeError(f"unsafe archive member: {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(item)
            if extracted is None:
                raise RuntimeError(f"could not read archive member: {name}")
            with extracted, target.open("wb") as output:
                shutil.copyfileobj(extracted, output, length=1024 * 1024)


def acquire(manifest_path: Path, destination: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "cucim-wsi-conformance-fixtures-v1":
        raise RuntimeError("unsupported fixture manifest")
    destination.mkdir(parents=True, exist_ok=True)

    for source in manifest["sources"]:
        members = source["members"]
        try:
            for member in members:
                verify(destination / member["path"], member)
        except RuntimeError:
            pass
        else:
            continue

        with tempfile.TemporaryDirectory(
            prefix="cucim-wsi-fixtures-"
        ) as temporary:
            download = Path(temporary) / "download"
            _download(source["url"], download)
            if source["url"].endswith(".tar"):
                _extract_selected(download, destination, members)
            elif len(members) == 1:
                target = destination / members[0]["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(download, target)
            else:
                raise RuntimeError(
                    "a non-archive source must have exactly one member"
                )

        for member in members:
            verify(destination / member["path"], member)

    for member in manifest["generated"]:
        target = destination / member["path"]
        generate(target)
        verify(target, member)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    acquire(args.manifest, args.destination)


if __name__ == "__main__":
    main()
