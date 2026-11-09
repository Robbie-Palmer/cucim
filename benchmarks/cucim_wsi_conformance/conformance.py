# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compare cuCIM/cuslide2 WSI semantics with OpenSlide."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import time
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "fixtures.json"
DEFAULT_FIXTURE_ROOT = HERE / "data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _check(name: str, passed: bool | None, **details: Any) -> dict[str, Any]:
    status = "skipped" if passed is None else ("pass" if passed else "fail")
    return {"name": name, "status": status, "details": details}


def _close(
    left: float | None, right: float | None, relative: float = 1e-4
) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= max(1e-6, abs(left) * relative)


def _pyramid_matches(
    left_dimensions: tuple[tuple[int, int], ...],
    right_dimensions: tuple[tuple[int, int], ...],
    left_downsamples: tuple[float, ...],
    right_downsamples: tuple[float, ...],
) -> bool:
    return (
        left_dimensions == right_dimensions
        and len(left_downsamples) == len(right_downsamples)
        and all(
            _close(left, right, relative=1e-6)
            for left, right in zip(
                left_downsamples, right_downsamples, strict=True
            )
        )
    )


def _bounds(properties: Any) -> tuple[int, int, int, int] | None:
    keys = (
        "openslide.bounds-x",
        "openslide.bounds-y",
        "openslide.bounds-width",
        "openslide.bounds-height",
    )
    values = tuple(_optional_int(properties.get(key)) for key in keys)
    if any(value is None for value in values):
        return None
    return values  # type: ignore[return-value]


def _flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not isinstance(value, dict):
        return result
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(_flatten_mapping(item, path))
        else:
            result[path.lower()] = item
    return result


def _cucim_bounds(metadata_value: Any) -> tuple[int, int, int, int] | None:
    flattened = _flatten_mapping(metadata_value)
    suffixes = ("bounds-x", "bounds-y", "bounds-width", "bounds-height")
    values: list[int | None] = []
    for suffix in suffixes:
        matches = [
            value
            for key, value in flattened.items()
            if key.endswith(suffix) or key.endswith(suffix.replace("-", "_"))
        ]
        values.append(_optional_int(matches[0]) if matches else None)
    if any(value is None for value in values):
        return None
    return tuple(values)  # type: ignore[return-value]


def _icc_observation(value: Any) -> dict[str, Any]:
    if not value:
        return {"present": False, "bytes": 0, "sha256": None}
    raw = bytes(value)
    return {
        "present": True,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _cucim_icc(metadata_value: Any) -> dict[str, Any]:
    matches = {
        key: value
        for key, value in _flatten_mapping(metadata_value).items()
        if "icc" in key or "color_profile" in key
    }
    byte_value = next(
        (
            value
            for value in matches.values()
            if isinstance(value, (bytes, bytearray))
        ),
        None,
    )
    result = _icc_observation(byte_value)
    result["metadata_keys"] = sorted(matches)
    return result


def _array_shape(array: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in array.shape)


def _normalize_array(array: Any) -> Any:
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise RuntimeError(
            f"expected HWC RGB/RGBA data, found shape {_array_shape(array)}"
        )
    return array


def _pixel_agreement(
    np: Any, openslide_rgba: Any, cucim_pixels: Any
) -> dict[str, Any]:
    openslide_rgba = _normalize_array(openslide_rgba)
    cucim_pixels = _normalize_array(cucim_pixels)
    if openslide_rgba.shape[:2] != cucim_pixels.shape[:2]:
        return {
            "comparable": False,
            "reason": "shape mismatch",
            "openslide_shape": _array_shape(openslide_rgba),
            "cucim_shape": _array_shape(cucim_pixels),
        }
    mask = openslide_rgba[:, :, 3] != 0
    if not bool(mask.any()):
        return {
            "comparable": False,
            "reason": "OpenSlide returned no in-bounds pixels",
        }
    difference = np.abs(
        openslide_rgba[:, :, :3][mask].astype(np.int16)
        - cucim_pixels[:, :, :3][mask].astype(np.int16)
    )
    return {
        "comparable": True,
        "samples": int(difference.size),
        "maximum_absolute_error": int(difference.max()),
        "mean_absolute_error": float(difference.mean()),
    }


def _alpha_agreement(
    np: Any, openslide_rgba: Any, cucim_pixels: Any
) -> dict[str, Any]:
    openslide_rgba = _normalize_array(openslide_rgba)
    cucim_pixels = _normalize_array(cucim_pixels)
    if openslide_rgba.shape[:2] != cucim_pixels.shape[:2]:
        return {"comparable": False, "reason": "shape mismatch"}
    openslide_alpha = openslide_rgba[:, :, 3]
    if cucim_pixels.shape[2] == 4:
        cucim_alpha = cucim_pixels[:, :, 3]
        return {
            "comparable": True,
            "representation": "explicit-alpha",
            "exact": bool(np.array_equal(openslide_alpha, cucim_alpha)),
            "openslide_transparent_pixels": int((openslide_alpha == 0).sum()),
            "cucim_transparent_pixels": int((cucim_alpha == 0).sum()),
        }
    return {
        "comparable": True,
        "representation": "implicit-opaque-rgb",
        "exact": bool((openslide_alpha == 255).all()),
        "openslide_transparent_pixels": int((openslide_alpha == 0).sum()),
        "cucim_transparent_pixels": None,
    }


def _outside_rgb_zero(np: Any, openslide_rgba: Any, cucim_pixels: Any) -> bool:
    openslide_rgba = _normalize_array(openslide_rgba)
    cucim_pixels = _normalize_array(cucim_pixels)
    if openslide_rgba.shape[:2] != cucim_pixels.shape[:2]:
        return False
    outside = openslide_rgba[:, :, 3] == 0
    return bool(outside.any()) and bool(
        (cucim_pixels[:, :, :3][outside] == 0).all()
    )


def _read_cucim_region(
    cupy: Any, image: Any, location: tuple[int, int], size: int
) -> tuple[Any, bool]:
    region = image.read_region(
        location=location,
        size=(size, size),
        level=0,
        device="cuda",
    )
    device_resident = hasattr(region, "__cuda_array_interface__")
    array = cupy.asarray(region)
    cupy.cuda.get_current_stream().synchronize()
    device_resident = device_resident or hasattr(
        array, "__cuda_array_interface__"
    )
    return cupy.asnumpy(array), device_resident


def _time_readers(
    cupy: Any,
    slide: Any,
    image: Any,
    location: tuple[int, int],
    size: int,
    repetitions: int,
) -> dict[str, Any]:
    slide.read_region(location, 0, (size, size))
    _read_cucim_region(cupy, image, location, size)
    openslide_seconds: list[float] = []
    cucim_seconds: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        slide.read_region(location, 0, (size, size))
        openslide_seconds.append(time.perf_counter() - started)

        started = time.perf_counter()
        _read_cucim_region(cupy, image, location, size)
        cucim_seconds.append(time.perf_counter() - started)
    return {
        "warmup_repetitions": 1,
        "measured_repetitions": repetitions,
        "openslide_median_milliseconds": 1000
        * statistics.median(openslide_seconds),
        "cucim_cuda_median_milliseconds": 1000
        * statistics.median(cucim_seconds),
    }


def _associated_observations(
    np: Any, cupy: Any, slide: Any, image: Any, tolerance: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    openslide_names = tuple(
        sorted(str(name) for name in slide.associated_images)
    )
    cucim_names = tuple(sorted(str(name) for name in image.associated_images))
    observations: dict[str, Any] = {}
    pixels_pass = True
    for name in sorted(set(openslide_names) & set(cucim_names)):
        openslide_image = slide.associated_images[name]
        openslide_pixels = np.asarray(openslide_image.convert("RGBA"))
        record: dict[str, Any] = {
            "openslide_shape": _array_shape(openslide_pixels),
            "openslide_icc": _icc_observation(
                openslide_image.info.get("icc_profile")
            ),
        }
        try:
            cucim_image = image.associated_image(name, device="cuda")
            cucim_array = cupy.asarray(cucim_image)
            cupy.cuda.get_current_stream().synchronize()
            cucim_pixels = cupy.asnumpy(cucim_array)
            record["cucim_cuda_resident"] = hasattr(
                cucim_array, "__cuda_array_interface__"
            )
            cucim_pixels = _normalize_array(cucim_pixels)
            agreement = _pixel_agreement(np, openslide_pixels, cucim_pixels)
            record["cucim_shape"] = _array_shape(cucim_pixels)
            record["pixel_agreement"] = agreement
            pixels_pass = pixels_pass and bool(
                agreement.get("comparable")
                and agreement["maximum_absolute_error"] <= tolerance
            )
        except (OSError, RuntimeError, ValueError) as error:
            record["cucim_error"] = str(error)
            pixels_pass = False
        observations[name] = record
    return (
        {
            "openslide_names": openslide_names,
            "cucim_names": cucim_names,
            "names_match": openslide_names == cucim_names,
            "images": observations,
        },
        {
            "names": openslide_names == cucim_names,
            "pixels": pixels_pass if observations else None,
        },
    )


def _inspect_tiff(tifffile: Any, path: Path) -> dict[str, Any]:
    try:
        with tifffile.TiffFile(path) as source:
            page = source.pages[0]
            orientation = page.tags.get("Orientation")
            icc = page.tags.get("InterColorProfile") or page.tags.get(
                "ICCProfile"
            )
            return {
                "primary_ifd_dimensions": (
                    int(page.imagewidth),
                    int(page.imagelength),
                ),
                "compression": str(page.compression.name).lower(),
                "is_tiled": bool(page.is_tiled),
                "tile_size": (
                    (int(page.tilewidth), int(page.tilelength))
                    if page.tilewidth and page.tilelength
                    else None
                ),
                "orientation": int(orientation.value) if orientation else 1,
                "icc": _icc_observation(icc.value if icc else None),
            }
    except (OSError, ValueError, tifffile.TiffFileError) as error:
        return {"error": str(error)}


def _probe_fixture(
    np: Any,
    cupy: Any,
    openslide: Any,
    tifffile: Any,
    root: Path,
    member: dict[str, Any],
    region_size: int,
    repetitions: int,
    tolerance: int,
) -> dict[str, Any]:
    path = root / member["path"]
    actual_bytes = path.stat().st_size
    actual_sha256 = _sha256(path)
    integrity = (
        actual_bytes == member["bytes"] and actual_sha256 == member["sha256"]
    )
    if not integrity:
        raise RuntimeError(f"fixture integrity check failed: {path}")

    result: dict[str, Any] = {
        "fixture": member,
        "integrity": {
            "bytes": actual_bytes,
            "sha256": actual_sha256,
            "verified": True,
        },
        "tiff": _inspect_tiff(tifffile, path),
        "checks": [],
    }
    checks: list[dict[str, Any]] = result["checks"]

    with openslide.OpenSlide(path) as slide:
        from cucim import CuImage

        image = CuImage(str(path))
        try:
            openslide_dimensions = tuple(
                (int(width), int(height))
                for width, height in slide.level_dimensions
            )
            openslide_downsamples = tuple(
                float(value) for value in slide.level_downsamples
            )
            openslide_mpp = (
                _optional_float(slide.properties.get("openslide.mpp-x")),
                _optional_float(slide.properties.get("openslide.mpp-y")),
            )
            openslide_bounds = _bounds(slide.properties)
            openslide_icc = _icc_observation(
                slide.read_region((0, 0), 0, (1, 1)).info.get("icc_profile")
            )

            resolutions = image.resolutions
            cucim_dimensions = tuple(
                (int(value[0]), int(value[1]))
                for value in resolutions["level_dimensions"]
            )
            cucim_downsamples = tuple(
                float(value) for value in resolutions["level_downsamples"]
            )
            spacing = tuple(float(value) for value in image.spacing("XY"))
            cucim_mpp = (
                spacing[0] if len(spacing) > 0 else None,
                spacing[1] if len(spacing) > 1 else None,
            )
            metadata_value = image.metadata
            cucim_bounds = _cucim_bounds(metadata_value)
            cucim_icc = _cucim_icc(metadata_value)

            result["openslide"] = {
                "vendor": slide.properties.get("openslide.vendor"),
                "level_dimensions": openslide_dimensions,
                "level_downsamples": openslide_downsamples,
                "mpp": openslide_mpp,
                "bounds": openslide_bounds,
                "icc": openslide_icc,
            }
            result["cucim"] = {
                "level_dimensions": cucim_dimensions,
                "level_downsamples": cucim_downsamples,
                "mpp": cucim_mpp,
                "bounds": cucim_bounds,
                "direction": image.direction,
                "origin": image.origin,
                "icc": cucim_icc,
            }

            checks.append(
                _check(
                    "primary_image_selection",
                    openslide_dimensions[0] == cucim_dimensions[0],
                    openslide=openslide_dimensions[0],
                    cucim=cucim_dimensions[0],
                )
            )
            checks.append(
                _check(
                    "pyramid_dimensions_and_downsamples",
                    _pyramid_matches(
                        openslide_dimensions,
                        cucim_dimensions,
                        openslide_downsamples,
                        cucim_downsamples,
                    ),
                    openslide_dimensions=openslide_dimensions,
                    cucim_dimensions=cucim_dimensions,
                    openslide_downsamples=openslide_downsamples,
                    cucim_downsamples=cucim_downsamples,
                )
            )
            checks.append(
                _check(
                    "mpp_x",
                    _close(openslide_mpp[0], cucim_mpp[0]),
                    openslide=openslide_mpp[0],
                    cucim=cucim_mpp[0],
                )
            )
            checks.append(
                _check(
                    "mpp_y",
                    _close(openslide_mpp[1], cucim_mpp[1]),
                    openslide=openslide_mpp[1],
                    cucim=cucim_mpp[1],
                )
            )
            checks.append(
                _check(
                    "bounds",
                    openslide_bounds == cucim_bounds,
                    openslide=openslide_bounds,
                    cucim=cucim_bounds,
                )
            )
            checks.append(
                _check(
                    "icc_profile_exposure",
                    openslide_icc["present"] == cucim_icc["present"],
                    openslide=openslide_icc,
                    cucim=cucim_icc,
                )
            )

            width, height = openslide_dimensions[0]
            size = min(region_size, width, height)
            center = (max(0, (width - size) // 2), max(0, (height - size) // 2))
            openslide_center = np.asarray(
                slide.read_region(center, 0, (size, size)).convert("RGBA")
            )
            cucim_center, center_device = _read_cucim_region(
                cupy, image, center, size
            )
            center_pixels = _pixel_agreement(np, openslide_center, cucim_center)
            center_alpha = _alpha_agreement(np, openslide_center, cucim_center)

            edge = (max(0, width - size // 2), max(0, height - size // 2))
            openslide_edge = np.asarray(
                slide.read_region(edge, 0, (size, size)).convert("RGBA")
            )
            try:
                cucim_edge, edge_device = _read_cucim_region(
                    cupy, image, edge, size
                )
                edge_pixels = _pixel_agreement(np, openslide_edge, cucim_edge)
                edge_alpha = _alpha_agreement(np, openslide_edge, cucim_edge)
                outside_zero = _outside_rgb_zero(np, openslide_edge, cucim_edge)
                edge_error = None
            except (OSError, RuntimeError, ValueError) as error:
                cucim_edge = None
                edge_device = False
                edge_pixels = {
                    "comparable": False,
                    "reason": "cuCIM read failed",
                }
                edge_alpha = {
                    "comparable": False,
                    "reason": "cuCIM read failed",
                }
                outside_zero = False
                edge_error = str(error)

            result["regions"] = {
                "center": {
                    "location": center,
                    "size": size,
                    "pixel_agreement": center_pixels,
                    "alpha": center_alpha,
                },
                "edge": {
                    "location": edge,
                    "size": size,
                    "pixel_agreement": edge_pixels,
                    "alpha": edge_alpha,
                    "outside_rgb_zero": outside_zero,
                    "cucim_error": edge_error,
                },
            }
            checks.append(
                _check(
                    "cuda_residency",
                    center_device and edge_device,
                    center=center_device,
                    edge=edge_device,
                )
            )
            checks.append(
                _check(
                    "pixel_tolerance",
                    bool(
                        center_pixels.get("comparable")
                        and center_pixels["maximum_absolute_error"] <= tolerance
                    ),
                    tolerance=tolerance,
                    **center_pixels,
                )
            )
            checks.append(
                _check(
                    "edge_tiles",
                    bool(
                        edge_pixels.get("comparable")
                        and edge_pixels["maximum_absolute_error"] <= tolerance
                        and outside_zero
                    ),
                    tolerance=tolerance,
                    outside_rgb_zero=outside_zero,
                    cucim_error=edge_error,
                    pixel_agreement=edge_pixels,
                )
            )
            alpha_exact = bool(
                center_alpha.get("exact") and edge_alpha.get("exact")
            )
            checks.append(
                _check(
                    "alpha_behavior",
                    alpha_exact,
                    center=center_alpha,
                    edge=edge_alpha,
                )
            )

            orientation = result["tiff"].get("orientation")
            identity_direction = image.direction == [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
            orientation_pass = (
                identity_direction
                if orientation == 1
                else bool(
                    center_pixels.get("comparable")
                    and center_pixels["maximum_absolute_error"] <= tolerance
                )
            )
            checks.append(
                _check(
                    "orientation",
                    orientation_pass,
                    tiff_orientation=orientation,
                    cucim_direction=image.direction,
                )
            )

            associated, associated_checks = _associated_observations(
                np, cupy, slide, image, tolerance
            )
            result["associated_images"] = associated
            checks.append(
                _check(
                    "associated_image_names",
                    associated_checks["names"],
                    openslide=associated["openslide_names"],
                    cucim=associated["cucim_names"],
                )
            )
            checks.append(
                _check(
                    "associated_image_pixels",
                    associated_checks["pixels"],
                    tolerance=tolerance,
                    common=tuple(associated["images"]),
                )
            )

            result["timing"] = _time_readers(
                cupy, slide, image, center, size, repetitions
            )
        finally:
            image.close()

    result["conformant"] = not any(
        check["status"] == "fail" for check in result["checks"]
    )
    return result


def _cpu_model() -> str:
    try:
        for line in (
            Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _memory_bytes() -> int | None:
    try:
        for line in (
            Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        ):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _nvidia_driver_version() -> str | None:
    try:
        first_line = (
            Path("/proc/driver/nvidia/version")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
    except (OSError, IndexError):
        return None
    match = re.search(r"\s{2,}(\d+\.\d+(?:\.\d+)?)\s", first_line)
    return match.group(1) if match else first_line


def _environment(
    cucim: Any, cupy: Any, openslide: Any, tifffile: Any
) -> dict[str, Any]:
    properties = cupy.cuda.runtime.getDeviceProperties(0)

    def gpu_property(name: str) -> Any:
        return properties.get(name, properties.get(name.encode()))

    gpu_name = gpu_property("name")
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode(errors="replace")
    gpu_uuid = gpu_property("uuid")
    if isinstance(gpu_uuid, (bytes, bytearray)) and len(gpu_uuid) == 16:
        gpu_uuid = f"GPU-{uuid.UUID(bytes=bytes(gpu_uuid))}"
    else:
        gpu_uuid = None
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "gpu": {
            "name": gpu_name,
            "uuid": gpu_uuid,
            "compute_capability": f"{gpu_property('major')}.{gpu_property('minor')}",
            "memory_bytes": int(gpu_property("totalGlobalMem")),
            "driver_version": _nvidia_driver_version(),
            "cuda_driver_api_version": str(
                cupy.cuda.runtime.driverGetVersion()
            ),
            "cuda_runtime_version": str(cupy.cuda.runtime.runtimeGetVersion()),
        },
        "dependencies": {
            "python": platform.python_version(),
            "cucim": getattr(
                cucim,
                "__version__",
                _package_version("cucim", "cucim-cu12", "cucim-cu13"),
            ),
            "cucim_source_commit": os.environ.get(
                "CUCIM_CONFORMANCE_SOURCE_COMMIT"
            )
            or getattr(cucim, "__git_commit__", None),
            "cucim_embedded_source_commit": getattr(
                cucim, "__git_commit__", None
            ),
            "cupy": cupy.__version__,
            "nvimagecodec": _package_version(
                "nvidia-nvimgcodec-cu12", "nvidia-nvimgcodec-cu13"
            ),
            "nvjpeg": _package_version(
                "nvidia-nvjpeg-cu12", "nvidia-nvjpeg-cu13"
            ),
            "nvjpeg2000": _package_version(
                "nvidia-nvjpeg2k-cu12", "nvidia-nvjpeg2k-cu13"
            ),
            "nvtiff": _package_version(
                "nvidia-nvtiff-cu12", "nvidia-nvtiff-cu13"
            ),
            "nvcomp": _package_version(
                "nvidia-libnvcomp-cu12", "nvidia-libnvcomp-cu13"
            ),
            "openslide_python": openslide.__version__,
            "openslide_library": openslide.__library_version__,
            "tifffile": tifffile.__version__,
            "numpy": _package_version("numpy"),
        },
    }


def run(
    manifest_path: Path,
    fixture_root: Path,
    implementation_label: str,
    region_size: int,
    repetitions: int,
    tolerance: int,
) -> dict[str, Any]:
    if os.environ.get("ENABLE_CUSLIDE2") != "1":
        raise RuntimeError(
            "set ENABLE_CUSLIDE2=1 so this suite exercises cuslide2"
        )

    import cupy
    import numpy
    import openslide
    import tifffile

    import cucim

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    members = [
        member for source in manifest["sources"] for member in source["members"]
    ] + manifest["generated"]
    fixtures = [
        _probe_fixture(
            numpy,
            cupy,
            openslide,
            tifffile,
            fixture_root,
            member,
            region_size,
            repetitions,
            tolerance,
        )
        for member in members
    ]
    failed_checks = sum(
        check["status"] == "fail"
        for fixture in fixtures
        for check in fixture["checks"]
    )
    return {
        "schema_version": "cucim-wsi-conformance-report-v1",
        "implementation_label": implementation_label,
        "fixture_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "schema_version": manifest["schema_version"],
        },
        "environment": _environment(cucim, cupy, openslide, tifffile),
        "method": {
            "oracle": "OpenSlide metadata and decoded pixels",
            "pixel_absolute_tolerance": tolerance,
            "region_size": region_size,
            "timing": (
                "time.perf_counter wall time; one warm-up and median of measured "
                "repetitions; cuCIM synchronized after every CUDA read; OpenSlide "
                "measured first; ordinary filesystem I/O with warm OS cache; no GDS"
            ),
        },
        "limitations": [
            "OpenSlide is a compatibility oracle, not independent ground truth for every vendor tag.",
            "Warm-cache single-region timings are diagnostic and are not a throughput benchmark.",
            "CUDA residency does not prove that every codec stage ran exclusively on the GPU.",
            "cuCIM does not expose the nvImageCodec plugin selected for an individual read.",
            "The public corpus is intentionally small and does not represent clinical diversity.",
            "ICC checks cover profile exposure and associated-image profiles, not color-managed rendering.",
            "Bounds are compared only when exposed through each reader's public metadata API.",
        ],
        "summary": {
            "fixtures": len(fixtures),
            "conformant_fixtures": sum(
                fixture["conformant"] for fixture in fixtures
            ),
            "failed_checks": failed_checks,
        },
        "fixtures": fixtures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT
    )
    parser.add_argument("--implementation-label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--region-size", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--pixel-absolute-tolerance", type=int, default=3)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.region_size < 1 or args.repetitions < 1:
        parser.error("region size and repetitions must be positive")
    if not 0 <= args.pixel_absolute_tolerance <= 255:
        parser.error("pixel tolerance must be between 0 and 255")

    report = run(
        args.manifest,
        args.fixture_root,
        args.implementation_label,
        args.region_size,
        args.repetitions,
        args.pixel_absolute_tolerance,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.strict and report["summary"]["failed_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
