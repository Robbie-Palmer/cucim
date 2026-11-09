# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SUITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_ROOT))

import conformance  # noqa: E402
from generate_anisotropic_tiff import (  # noqa: E402
    HEIGHT,
    MPP_X,
    MPP_Y,
    WIDTH,
    generate,
)


def test_fixture_manifest_is_complete_and_unambiguous() -> None:
    manifest = json.loads((SUITE_ROOT / "fixtures.json").read_text())
    members = [
        member for source in manifest["sources"] for member in source["members"]
    ] + manifest["generated"]

    assert manifest["schema_version"] == "cucim-wsi-conformance-fixtures-v1"
    assert {member["format"] for member in members} >= {
        "aperio-svs",
        "leica-scn",
        "tiled-tiff",
        "generic-tiled-tiff",
    }
    assert len({member["path"] for member in members}) == len(members)
    for member in members:
        assert member["bytes"] > 0
        assert len(member["sha256"]) == 64
        assert member["license_spdx"]
        assert member["attribution"]


def test_anisotropic_fixture_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first.tiff"
    second = tmp_path / "second.tiff"
    generate(first)
    generate(second)

    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size == 295292
    assert hashlib.sha256(first.read_bytes()).hexdigest() == (
        "c682a1f76d91051563153a20354d3578d46427d7eb6242eea98a1b1163a56fa7"
    )
    assert (WIDTH, HEIGHT, MPP_X, MPP_Y) == (257, 193, 0.25, 0.5)


def test_pyramid_comparison_checks_dimensions_and_downsamples() -> None:
    assert conformance._pyramid_matches(
        ((100, 80), (50, 40)),
        ((100, 80), (50, 40)),
        (1.0, 2.0),
        (1.0, 2.0000001),
    )
    assert not conformance._pyramid_matches(
        ((100, 80),),
        ((80, 100),),
        (1.0,),
        (1.0,),
    )
    assert not conformance._pyramid_matches(
        ((100, 80), (50, 40)),
        ((100, 80), (50, 40)),
        (1.0, 2.0),
        (1.0, 4.0),
    )


def test_pixel_tolerance_ignores_only_transparent_openslide_pixels() -> None:
    openslide = np.array(
        [[[10, 20, 30, 255], [200, 200, 200, 0]]], dtype=np.uint8
    )
    cucim = np.array([[[11, 20, 29], [0, 0, 0]]], dtype=np.uint8)

    agreement = conformance._pixel_agreement(np, openslide, cucim)
    assert agreement == {
        "comparable": True,
        "samples": 3,
        "maximum_absolute_error": 1,
        "mean_absolute_error": 2 / 3,
    }
    assert conformance._outside_rgb_zero(np, openslide, cucim)


def test_alpha_comparison_distinguishes_rgb_from_rgba_at_edges() -> None:
    openslide = np.array([[[10, 20, 30, 255], [0, 0, 0, 0]]], dtype=np.uint8)
    rgb = openslide[:, :, :3]
    rgba = openslide.copy()

    implicit = conformance._alpha_agreement(np, openslide, rgb)
    explicit = conformance._alpha_agreement(np, openslide, rgba)
    assert implicit["representation"] == "implicit-opaque-rgb"
    assert not implicit["exact"]
    assert explicit["representation"] == "explicit-alpha"
    assert explicit["exact"]


def test_mpp_axes_are_compared_independently() -> None:
    assert conformance._close(0.25, 0.25)
    assert conformance._close(0.5, 0.5)
    assert not conformance._close(0.25, 0.5)
    assert not conformance._close(0.5, 0.25)
