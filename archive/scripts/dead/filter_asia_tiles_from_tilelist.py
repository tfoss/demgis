import re

# Asia/Middle East bounds (integer degrees)
# Covers Middle East, Central Asia, South Asia, East Asia, Southeast Asia
# For Middle East specifically: ~12°N to 42°N, 34°E to 63°E
# Extended to cover all of Asia: 0°N to 60°N, 25°E to 150°E
LAT_MIN, LAT_MAX = 0, 60
LON_MIN, LON_MAX = 25, 150

TILELIST_PATH = "tileList.txt"
OUT_PATH = "asia_tile_urls_s3.txt"

# Example ID: Copernicus_DSM_COG_10_S10_00_W060_00_DEM
ID_PATTERN = re.compile(
    r"Copernicus_DSM_COG_10_([NS]\d{2})_00_([EW]\d{3})_00_DEM"
)

def parse_lat_lon_from_id(tile_id: str):
    m = ID_PATTERN.fullmatch(tile_id)
    if not m:
        return None, None

    lat_token, lon_token = m.group(1), m.group(2)

    # Latitude
    lat_sign = 1 if lat_token[0] == "N" else -1
    lat_deg = int(lat_token[1:])
    lat = lat_sign * lat_deg

    # Longitude
    lon_sign = 1 if lon_token[0] == "E" else -1
    lon_deg = int(lon_token[1:])
    lon = lon_sign * lon_deg

    return lat, lon

def main():
    count_total = 0
    count_keep = 0

    with open(TILELIST_PATH, "r") as f_in, open(OUT_PATH, "w") as f_out:
        for line in f_in:
            tile_id = line.strip()
            if not tile_id:
                continue

            count_total += 1
            lat, lon = parse_lat_lon_from_id(tile_id)
            if lat is None:
                continue

            if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
                key = f"{tile_id}/{tile_id}.tif"
                s3_url = f"s3://copernicus-dem-30m/{key}"
                f_out.write(s3_url + "\n")
                count_keep += 1

    print(f"Total IDs in tileList: {count_total}")
    print(f"Asia/Middle East tiles written to {OUT_PATH}: {count_keep}")

if __name__ == "__main__":
    main()
