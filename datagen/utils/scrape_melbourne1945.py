import os
import math
import urllib.request

BASE_URL = 'http://1945.melbourne/tileserver-php-master/s0za1yvi'
OUTPUT_DIR = 'melbourne_1945_tiles'
MIN_ZOOM = 11
MAX_ZOOM = 17
# bounding box: southwest (lat1, lon1) and northeast (lat2, lon2)
LAT1, LON1 = -38.646, 143.645
LAT2, LON2 = -37.065, 146.210


def tile_xy(lat, lon, zoom):
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def download_tile(z, x, y):
    url = f"{BASE_URL}/{z}/{x}/{y}.png"
    local_path = os.path.join(OUTPUT_DIR, str(z), str(x), f"{y}.png")
    if os.path.exists(local_path):
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
            if len(data) > 10:  # skip empty tiles
                with open(local_path, 'wb') as f:
                    f.write(data)
                print('Saved', local_path)
    except Exception as e:
        print('Failed', url, e)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for z in range(MIN_ZOOM, MAX_ZOOM + 1):
        x_min, y_max = tile_xy(LAT1, LON1, z)
        x_max, y_min = tile_xy(LAT2, LON2, z)
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                download_tile(z, x, y)


if __name__ == '__main__':
    main()
