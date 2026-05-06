#!/usr/bin/env python3
"""
Download missing 90m Copernicus DEM tiles for NW Iran/Armenia/Azerbaijan region.
These will be used as fallback where 30m tiles are not available.
"""

from pathlib import Path

# Missing tiles in NW Iran/Armenia/Azerbaijan region (30m tiles that don't exist)
missing_30m_tiles = [
    "Copernicus_DSM_COG_10_N38_00_E045_00_DEM.tif",
    "Copernicus_DSM_COG_10_N38_00_E046_00_DEM.tif",
    "Copernicus_DSM_COG_10_N38_00_E048_00_DEM.tif",
    "Copernicus_DSM_COG_10_N39_00_E044_00_DEM.tif",
    "Copernicus_DSM_COG_10_N39_00_E045_00_DEM.tif",
    "Copernicus_DSM_COG_10_N39_00_E046_00_DEM.tif",
    "Copernicus_DSM_COG_10_N39_00_E047_00_DEM.tif",
    "Copernicus_DSM_COG_10_N39_00_E048_00_DEM.tif",
    "Copernicus_DSM_COG_10_N40_00_E043_00_DEM.tif",
    "Copernicus_DSM_COG_10_N40_00_E044_00_DEM.tif",
    "Copernicus_DSM_COG_10_N40_00_E045_00_DEM.tif",
    "Copernicus_DSM_COG_10_N40_00_E046_00_DEM.tif",
    "Copernicus_DSM_COG_10_N40_00_E047_00_DEM.tif",
    "Copernicus_DSM_COG_10_N40_00_E048_00_DEM.tif",
    "Copernicus_DSM_COG_10_N41_00_E043_00_DEM.tif",
    "Copernicus_DSM_COG_10_N41_00_E044_00_DEM.tif",
    "Copernicus_DSM_COG_10_N41_00_E045_00_DEM.tif",
    "Copernicus_DSM_COG_10_N41_00_E046_00_DEM.tif",
    "Copernicus_DSM_COG_10_N41_00_E047_00_DEM.tif",
    "Copernicus_DSM_COG_10_N41_00_E048_00_DEM.tif",
]

# Convert to 90m tile names (use COG_30 prefix instead of COG_10)
output_dir = Path("middle_east_central_asia_tiles_90m")
output_dir.mkdir(exist_ok=True)

s5cmd_list = []
for tile_30m in missing_30m_tiles:
    # Convert: Copernicus_DSM_COG_10_N40_00_E045_00_DEM.tif
    # To:      Copernicus_DSM_COG_30_N40_00_E045_00_DEM.tif
    tile_90m = tile_30m.replace("COG_10_", "COG_30_")

    # S3 path for 90m tiles
    tile_name_no_ext = tile_90m.replace(".tif", "")
    s3_path = f"s3://copernicus-dem-90m/{tile_name_no_ext}/{tile_90m}"
    local_path = output_dir / tile_90m

    s5cmd_list.append(f"cp {s3_path} {local_path}")

# Write s5cmd list
with open("missing_90m_s5cmd_list.txt", "w") as f:
    for cmd in s5cmd_list:
        f.write(cmd + "\n")

print(f"Generated download list for {len(s5cmd_list)} 90m tiles")
print(f"Tiles will be downloaded to: {output_dir}/")
print(f"Run: s5cmd --no-sign-request run missing_90m_s5cmd_list.txt")
