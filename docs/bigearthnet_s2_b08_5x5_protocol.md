# BigEarthNet-S2 B08 5x5 Mosaic Protocol

## Goal

Derive larger single-channel NIR images from the local BigEarthNet-S2 patch
tree by stitching complete 5x5 neighborhoods of `B08` patches. Each `B08`
patch is 120x120 pixels at 10 m resolution, so one 5x5 mosaic is 600x600
pixels and covers 6 km x 6 km.

## Local Dataset Findings

The local dataset is at:

```text
data/raw/BigEarthNet-S2
```

The downloaded description in `Description_BigEarthNet_v2.pdf` says
BigEarthNet-S2 is organized as one directory per Sentinel-2 source tile and one
directory per patch. Patch names end with `<H-Order>_<V-Order>`, where
`H-Order` is the horizontal grid index and `V-Order` is the vertical grid
index. Those two indices are enough to identify neighboring patches inside one
source-tile directory.

Exploration script:

```bash
python scripts/datasets/explore_bigearthnet_s2_mosaics.py \
  --root data/raw/BigEarthNet-S2 \
  --band B08 \
  --window-size 5 \
  --metadata-parquet data/raw/BigEarthNet-S2/metadata.parquet \
  --max-band-shape-checks 2000 \
  --max-transform-checks 115
```

Output report:

```text
artifacts/analysis/bigearthnet_s2_mosaics/exploration_summary.json
```

Main local results:

| Check | Result |
| --- | ---: |
| Sentinel-2 source-tile directories | 115 |
| Patch directories | 549,488 |
| Missing `B08` files | 0 |
| Sampled `B08` shape/dtype | 2,000 / 2,000 were `120x120`, `uint16` (`I;16`) |
| Sampled `B08` pixel scale | 2,000 / 2,000 had `(10.0, 10.0, 0.0)` |
| Complete sliding 5x5 windows | 430,524 |
| Complete non-overlapping windows with `H % 5 == 0`, `V % 5 == 0` | 18,447 |
| Best global stride-5 offset | offset `(1, 0)`, 18,461 windows |
| Per-scene best stride-5 offsets | 18,678 windows |
| Transform checks | 115 / 115 sampled complete windows passed |

The local `metadata.parquet` was also inspected:

| Official metadata check | Result |
| --- | ---: |
| Metadata path | `data/raw/BigEarthNet-S2/metadata.parquet` |
| Metadata rows | 480,038 |
| Duplicate patch IDs | 0 |
| Null patch IDs / splits | 0 / 0 |
| Train patches | 237,871 |
| Validation patches | 122,342 |
| Test patches | 119,825 |

When a 5x5 window is required to use only metadata-listed patches and to keep
all 25 patches in the same official split, the counts become:

| Window variant | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Sliding 5x5 | 123,406 | 53,565 | 78,837 | 255,808 |
| Non-overlap, global offset `(0, 0)` | 5,628 | 1,941 | 2,854 | 10,423 |
| Non-overlap, best single offset `(3, 3)` | - | - | - | 11,883 |
| Non-overlap, best offset per split | 5,630 | 3,010 | 3,603 | 12,243 |
| Non-overlap, best offset per scene and split | 5,803 | 3,024 | 3,670 | 12,497 |

Among all 430,524 geometrically complete sliding windows, 96,640 include at
least one patch that is absent from `metadata.parquet`, and 78,076 contain
patches from more than one official split. Those windows should not be used in
split-compliant training or evaluation.

The script can also write one stitched proof mosaic:

```text
artifacts/analysis/bigearthnet_s2_mosaics/example/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_71_16_B08_5x5.npy
artifacts/analysis/bigearthnet_s2_mosaics/example/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_71_16_B08_5x5_preview.png
```

## Feasibility

This derivation is feasible. The local `B08` files are consistently 120x120
pixels, and the GeoTIFF tags align with the patch grid: increasing `H-Order`
moves east by 1,200 m, and increasing `V-Order` moves south by 1,200 m. A 5x5
block therefore creates a coherent 600x600 NIR image without resampling.

Do not merge across source-tile directories. Source-tile directories can differ
by acquisition date, MGRS tile, UTM zone, or CRS. Each mosaic should be formed
only from 25 patches under the same immediate scene directory.

## Recommended Dataset Variant

Use non-overlapping stride-5 windows with official split filtering for the
first derived dataset:

```text
anchor is valid when H % 5 == 0 and V % 5 == 0
window contains patches (H + dx, V + dy) for dx, dy in [0, 4]
all 25 patches must exist
all 25 patch IDs must appear in metadata.parquet
all 25 patches must have the same metadata.parquet split
```

This gives 10,423 local split-compliant images: 5,628 train, 1,941 validation,
and 2,854 test. It avoids reusing the same patch in multiple mosaics and avoids
train/validation/test leakage. That is usually better for training/evaluation
than the 255,808 split-compliant sliding windows, which are valid but extremely
correlated.

If maximizing sample count matters more than the simple `(0, 0)` tiling, use a
single global stride offset `(3, 3)` after split filtering. It produces 11,883
total split-compliant non-overlapping images. Avoid choosing different offsets
for train, validation, and test unless the manifest records the policy clearly;
it adds bookkeeping and makes comparisons less tidy.

If dense augmentation is desired, sliding windows can be generated as a
separate experimental variant, not mixed with the non-overlapping baseline.

## Output Layout

Keep derived data out of tracked source:

```text
data/derived/bigearthnet_s2_b08_5x5/
  images/
    <scene>_H<h>_V<v>_B08_5x5.tif
  previews/
    <scene>_H<h>_V<v>_B08_5x5.png
  manifests/
    train.jsonl
    validation.jsonl
    test.jsonl
    all.jsonl
  stats/
    train_b08_stats.json
```

The GeoTIFF should remain single-channel `uint16`. PNG previews are only for
inspection and should use a contrast stretch; they should not replace the raw
training data.

Manifest entries should include:

```json
{
  "sample_id": "<scene>_H<h>_V<v>_B08_5x5",
  "scene": "<Sentinel-2 source tile directory>",
  "anchor_h": 0,
  "anchor_v": 0,
  "band": "B08",
  "image_path": "data/derived/bigearthnet_s2_b08_5x5/images/...",
  "source_patch_ids": ["...", "..."],
  "source_patch_paths": ["...", "..."],
  "height": 600,
  "width": 600,
  "dtype": "uint16",
  "pixel_size_m": 10,
  "crs": "from source GeoTIFF",
  "split": "train"
}
```

## Split And Label Policy

The local extraction now includes the official metadata file:

```text
data/raw/BigEarthNet-S2/metadata.parquet
```

This file has columns `patch_id`, `labels`, `split`, `country`, `s1_name`,
`s2v1_name`, `contains_seasonal_snow`, and `contains_cloud_or_shadow`. The
metadata rows are the recommended patches, excluding patches with seasonal
snow, clouds, or cloud shadows.

Recommended policy:

1. Use `data/raw/BigEarthNet-S2/metadata.parquet` as the default whitelist.
2. For every candidate 5x5 window, require all 25 patch IDs to be present in
   the whitelist.
3. Assign a mosaic split only if all 25 patches share the same BigEarthNet
   split. Drop or quarantine mixed-split windows.
4. For image-only generative training, labels can be omitted from the training
   tensor but should remain in the manifest.
5. If patch-level labels are needed, store the union of the 25 patch label sets
   and optionally per-patch labels.
6. If pixel-level reference maps are downloaded later, stitch the matching 5x5
   reference-map patches with the exact same anchor rules.

## Implementation Notes

Use `rasterio` for the production writer, even though the exploration script
uses Pillow. `rasterio` will preserve CRS and write a correct transform for the
600x600 GeoTIFF.

For each valid window:

```python
rows = []
for dy in range(5):
    row = []
    for dx in range(5):
        patch = patches[(anchor_h + dx, anchor_v + dy)]
        row.append(read_uint16_b08(patch))
    rows.append(row)
mosaic = np.block(rows)  # shape: (600, 600)
```

Production write rules:

1. Read the top-left patch profile and transform.
2. Assert all 25 patches have the same CRS, dtype, shape, and pixel size.
3. Assert neighboring transforms match the expected 1,200 m offsets.
4. Write height `600`, width `600`, count `1`, dtype `uint16`, CRS from the
   top-left patch, and transform from the top-left patch.
5. Use compression such as `deflate` with a predictor for smaller GeoTIFFs.

## Validation Checklist

Before launching the full derivation:

```bash
python scripts/datasets/explore_bigearthnet_s2_mosaics.py \
  --root data/raw/BigEarthNet-S2 \
  --band B08 \
  --window-size 5 \
  --metadata-parquet data/raw/BigEarthNet-S2/metadata.parquet
```

Then verify:

1. `missing_B08_files == 0`.
2. Sampled `B08` files are `120x120` and `uint16`.
3. Transform checks pass for sampled windows.
4. A random set of output mosaics has shape `600x600`.
5. No output sample crosses source-tile directories.
6. Every output window has 25 metadata-listed patches and exactly one split.
7. Dataset statistics are computed on the training split only.

## Open Choices

The main choice is window density:

| Variant | Count | Use when |
| --- | ---: | --- |
| Non-overlap, split-compliant global offset `(0, 0)` | 10,423 | Recommended baseline |
| Non-overlap, split-compliant best single offset `(3, 3)` | 11,883 | Slightly more samples, less natural anchor |
| Non-overlap, split-compliant best offset per split | 12,243 | More samples, split-specific bookkeeping |
| Non-overlap, split-compliant best offset per scene and split | 12,497 | Max non-overlap count, more manifest bookkeeping |
| Sliding 5x5, split-compliant | 255,808 | Dense augmentation, high spatial correlation |

Start with the global `(0, 0)` non-overlapping protocol unless there is a
specific reason to maximize count.
