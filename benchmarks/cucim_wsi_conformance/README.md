# cuCIM whole-slide semantic conformance

This opt-in suite compares the public `CuImage`/cuslide2 contract with
[OpenSlide](https://openslide.org/api/python/) on public Aperio SVS, Leica SCN,
and tiled TIFF fixtures. It is a diagnostic benchmark, not a production reader
or a replacement for cuCIM's unit tests. Known semantic differences are written
to a JSON report; they do not fail the command unless `--strict` is requested.

The suite intentionally contains no Aperio or Leica reader fixes. It keeps the
fixture acquisition, observation, comparison, and reporting code under
`benchmarks/` so downstream compatibility evidence does not enter the cuCIM
runtime.

## Public fixtures

[`fixtures.json`](fixtures.json) is the authority for every URL, immutable
revision, license, attribution, byte count, and SHA-256 checksum. Acquisition
verifies both size and digest before a reader is opened.

| Source | Coverage | License |
| --- | --- | --- |
| [WSILabs wsi-fixtures v8](https://github.com/WSILabs/wsi-fixtures/releases/tag/v8), source `3c52c9594a87d51a177836bbcce1350132392bbb` | Aperio JPEG and JPEG 2000; ImageScope JPEG, LZW, and uncompressed tiled TIFF | Per-file `CC0-1.0` or `CC-BY-4.0`; see the manifest and upstream sidecars |
| [OpenSlide Leica test data](https://openslide.cs.cmu.edu/download/openslide-testdata/Leica/) | `Leica-1.scn`, 291,812,870 bytes | Free to use and distribute, with or without modification; credit Yves Sucaet |
| [`generate_anisotropic_tiff.py`](generate_anisotropic_tiff.py) | Generic uncompressed tiled TIFF, including partial edge tiles and 0.25 × 0.5 µm/px | `Apache-2.0` |

The anisotropic TIFF is generated from an asymmetric pixel formula by a
standard-library-only writer. Its exact 295,292-byte output has SHA-256
`c682a1f76d91051563153a20354d3578d46427d7eb6242eea98a1b1163a56fa7`.
No fixture binary is checked into Git.

Acquire all fixtures into the ignored local data directory:

```bash
python benchmarks/cucim_wsi_conformance/fetch_fixtures.py
```

## Run released cuCIM and current main separately

Run each implementation in a clean environment and give it an unambiguous
label. The test dependencies in `python/cucim/pyproject.toml` provide NumPy,
OpenSlide Python, and tifffile; the OpenSlide shared library must also be
installed. Set `ENABLE_CUSLIDE2=1` explicitly.

For the released baseline:

```bash
ENABLE_CUSLIDE2=1 python benchmarks/cucim_wsi_conformance/conformance.py \
  --implementation-label released-v26.10.00 \
  --output benchmarks/cucim_wsi_conformance/reports/released-v26.10.00.json
```

For a source build from the checked-out main commit:

```bash
CUCIM_CONFORMANCE_SOURCE_COMMIT=$(git rev-parse HEAD) \
  ENABLE_CUSLIDE2=1 \
  python benchmarks/cucim_wsi_conformance/conformance.py \
  --implementation-label current-main-$(git rev-parse --short=12 HEAD) \
  --output benchmarks/cucim_wsi_conformance/reports/current-main.json
```

Follow `CONTRIBUTING.md` to build and install the checkout before the second
command. Keep the reports separate: the embedded cuCIM version and source
commit (or the explicit source-build commit above) prevent a wheel result from
being mistaken for a main result. Generated JSON reports are ignored by Git
because they contain machine-specific facts.

Use `--strict` only when a job should enforce complete OpenSlide parity. The
default mode still exits nonzero for fixture corruption, unreadable inputs, or
runner errors.

## Contract

Each fixture report records and checks:

- primary image selection, complete pyramid dimensions, and downsample values;
- MPP X and Y independently, including the generated 0.25 × 0.5 fixture;
- standardized bounds and TIFF orientation/direction behavior;
- slide and associated-image ICC profile exposure;
- explicit versus implicit alpha behavior for interior and out-of-bounds reads;
- associated-image names, dimensions, profiles, and common-image pixels;
- a center region and a bottom-right partial edge tile;
- CUDA array-interface residency for both reads; and
- RGB maximum and mean absolute error against OpenSlide, with an explicit
  default maximum tolerance of 3.

The report also records the host kernel, CPU, logical CPU count, memory, GPU,
compute capability, driver, CUDA runtime, cuCIM source revision, codec package
versions, OpenSlide Python/library versions, tifffile, NumPy, and CuPy.

Timing uses `time.perf_counter`, one warm-up, and the median of five measured
reads. cuCIM's stream is synchronized after every CUDA read. OpenSlide is timed
first, the OS cache is warm, storage is ordinary filesystem I/O, and no GDS path
is used. These numbers are diagnostic single-region latency, not a throughput
claim.

## Limitations

- OpenSlide is the compatibility oracle, not independent truth for every
  vendor tag.
- OpenSlide's generic-TIFF RGBA conversion is also part of the measured
  contract. Inspect the recorded alpha and pixel details before attributing a
  generic RGB TIFF difference solely to cuCIM.
- The small public matrix is not representative of scanner or clinical
  diversity.
- CUDA-resident output proves the copy boundary, but cuCIM does not expose the
  selected nvImageCodec plugin and some codecs can still perform host work.
- Bounds can only be compared when both public reader APIs expose them.
- ICC checks compare profile exposure and bytes; they do not perform a
  color-managed display transform.
- Timings are warm-cache, serial, and machine-specific.

Run the runner's focused contract tests with:

```bash
python -m pytest benchmarks/cucim_wsi_conformance/tests
```
