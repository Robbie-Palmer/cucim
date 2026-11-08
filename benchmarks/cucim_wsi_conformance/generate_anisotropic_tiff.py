# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Generate a byte-reproducible anisotropic generic tiled TIFF fixture."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

WIDTH = 257
HEIGHT = 193
TILE_WIDTH = 128
TILE_HEIGHT = 128
MPP_X = 0.25
MPP_Y = 0.5

_ASCII = 2
_SHORT = 3
_LONG = 4
_RATIONAL = 5


def _align(value: int, boundary: int = 4) -> int:
    return (value + boundary - 1) // boundary * boundary


def _pixel(x: int, y: int) -> bytes:
    """Return an asymmetric RGB pattern that reveals flips and rotations."""
    return bytes(
        (
            (3 * x + 5 * y) % 256,
            (x ^ (2 * y)) % 256,
            (7 * x + 11 * y) % 256,
        )
    )


def _tiles() -> list[bytes]:
    result: list[bytes] = []
    tiles_across = (WIDTH + TILE_WIDTH - 1) // TILE_WIDTH
    tiles_down = (HEIGHT + TILE_HEIGHT - 1) // TILE_HEIGHT
    for tile_y in range(tiles_down):
        for tile_x in range(tiles_across):
            tile = bytearray(TILE_WIDTH * TILE_HEIGHT * 3)
            for row in range(TILE_HEIGHT):
                y = tile_y * TILE_HEIGHT + row
                if y >= HEIGHT:
                    break
                for column in range(TILE_WIDTH):
                    x = tile_x * TILE_WIDTH + column
                    if x >= WIDTH:
                        break
                    offset = (row * TILE_WIDTH + column) * 3
                    tile[offset : offset + 3] = _pixel(x, y)
            result.append(bytes(tile))
    return result


def generate(path: Path) -> None:
    """Write the canonical fixture to *path* without third-party libraries."""
    tile_data = _tiles()
    tile_bytes = TILE_WIDTH * TILE_HEIGHT * 3
    description = b"cuCIM public anisotropic MPP conformance fixture\x00"
    software = b"cucim-wsi-conformance\x00"

    # Classic little-endian TIFF.  Keeping all layout calculations explicit
    # makes the output identical across Python, tifffile, and libtiff versions.
    entry_count = 18
    cursor = 8 + 2 + entry_count * 12 + 4

    bits_offset = _align(cursor)
    cursor = bits_offset + 6
    description_offset = _align(cursor)
    cursor = description_offset + len(description)
    x_resolution_offset = _align(cursor)
    cursor = x_resolution_offset + 8
    y_resolution_offset = _align(cursor)
    cursor = y_resolution_offset + 8
    software_offset = _align(cursor)
    cursor = software_offset + len(software)
    tile_offsets_offset = _align(cursor)
    cursor = tile_offsets_offset + 4 * len(tile_data)
    tile_counts_offset = _align(cursor)
    cursor = tile_counts_offset + 4 * len(tile_data)
    tile_data_offset = _align(cursor)
    tile_offsets = [
        tile_data_offset + index * tile_bytes for index in range(len(tile_data))
    ]

    entries = [
        (254, _LONG, 1, 0),  # NewSubfileType
        (256, _LONG, 1, WIDTH),
        (257, _LONG, 1, HEIGHT),
        (258, _SHORT, 3, bits_offset),
        (259, _SHORT, 1, 1),  # Compression = none
        (262, _SHORT, 1, 2),  # PhotometricInterpretation = RGB
        (270, _ASCII, len(description), description_offset),
        (274, _SHORT, 1, 1),  # Orientation = top-left
        (277, _SHORT, 1, 3),
        (282, _RATIONAL, 1, x_resolution_offset),
        (283, _RATIONAL, 1, y_resolution_offset),
        (284, _SHORT, 1, 1),  # PlanarConfiguration = contiguous
        (296, _SHORT, 1, 3),  # ResolutionUnit = centimeter
        (305, _ASCII, len(software), software_offset),
        (322, _LONG, 1, TILE_WIDTH),
        (323, _LONG, 1, TILE_HEIGHT),
        (324, _LONG, len(tile_data), tile_offsets_offset),
        (325, _LONG, len(tile_data), tile_counts_offset),
    ]

    output = bytearray(tile_data_offset + tile_bytes * len(tile_data))
    output[:8] = struct.pack("<2sHI", b"II", 42, 8)
    struct.pack_into("<H", output, 8, entry_count)
    for index, (tag, field_type, count, value) in enumerate(entries):
        struct.pack_into(
            "<HHII", output, 10 + index * 12, tag, field_type, count, value
        )
    struct.pack_into("<I", output, 10 + entry_count * 12, 0)

    struct.pack_into("<HHH", output, bits_offset, 8, 8, 8)
    output[description_offset : description_offset + len(description)] = (
        description
    )
    struct.pack_into("<II", output, x_resolution_offset, 40_000, 1)
    struct.pack_into("<II", output, y_resolution_offset, 20_000, 1)
    output[software_offset : software_offset + len(software)] = software
    struct.pack_into(
        f"<{len(tile_offsets)}I", output, tile_offsets_offset, *tile_offsets
    )
    struct.pack_into(
        f"<{len(tile_data)}I",
        output,
        tile_counts_offset,
        *([tile_bytes] * len(tile_data)),
    )
    for offset, tile in zip(tile_offsets, tile_data, strict=True):
        output[offset : offset + len(tile)] = tile

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
