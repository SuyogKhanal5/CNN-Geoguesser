import os
import time
import random
import requests
from shapely.geometry import Point
import geopandas as gpd
import csv

# Your API Key here
API_KEY = "API_KEY"

NUM_IMAGES = 800
MAX_ATTEMPTS = 100000


def generate_candidate_points(country_geom, n_candidates):
    points = []
    minx, miny, maxx, maxy = country_geom.bounds
    attempts = 0

    while len(points) < n_candidates and attempts < MAX_ATTEMPTS:
        lon = random.uniform(minx, maxx)
        lat = random.uniform(miny, maxy)
        pt = Point(lon, lat)
        if country_geom.contains(pt):
            points.append((lat, lon))
        attempts += 1

    return points


def has_streetview(lat, lon):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lon}",
        "key": API_KEY
    }
    response = requests.get(url, params=params).json()
    return response.get("status") == "OK"


def download_image(lat, lon, heading, filename):
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x640",
        "location": f"{lat},{lon}",
        "heading": heading,
        "fov": 90,
        "pitch": 0,
        "key": API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    return False

world = gpd.read_file('ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
countries = ['United States of America', 'United Kingdom', 'India', 'Japan', 'Thailand', 'Spain', 'Israel', 'Switzerland', 'Netherlands', 'Taiwan']

for COUNTRY_NAME in countries:
    OUT_FOLDER = f"streetview_dataset/{COUNTRY_NAME}"
    os.makedirs(OUT_FOLDER, exist_ok=True)

    # ✅ Count existing images
    existing_images = [f for f in os.listdir(OUT_FOLDER) if f.endswith(".jpg")]
    start_index = len(existing_images)
    
    if start_index >= NUM_IMAGES:
        print(f"✅ Completed: {start_index} images already exist in {OUT_FOLDER}")
        continue

    # ✅ Load previous metadata file if exists
    metadata_path = os.path.join(OUT_FOLDER, "metadata.csv")
    metadata_rows = []
    metadata_exists = os.path.exists(metadata_path)

    country = world[world['NAME'] == COUNTRY_NAME]
    geometry = country.geometry.iloc[0]

    print(f"Getting street view images from ", COUNTRY_NAME, ":")
    image_count = 0

    while start_index + image_count < NUM_IMAGES:
        candidate = generate_candidate_points(geometry, 1)

        for lat, lon in candidate:
            if has_streetview(lat, lon):
                global_index = start_index + image_count
                print('Extracting image', global_index, 'from', COUNTRY_NAME)
                heading = random.choice([0, 90, 180, 270])
                img_path = os.path.join(OUT_FOLDER, f"{global_index:05d}.jpg")
                if download_image(lat, lon, heading, img_path):
                    metadata_rows.append([img_path, lat, lon, heading])
                    image_count += 1
                    time.sleep(0.1)

    # ✅ Append metadata instead of overwrite
    with open(metadata_path, "a", newline='') as f:
        writer = csv.writer(f)
        if not metadata_exists:
            writer.writerow(["filename", "latitude", "longitude", "heading"])
        writer.writerows(metadata_rows)

    print(f"✅ Completed: {image_count} new images saved in {OUT_FOLDER}")
