import shapefile
import pandas as pd
import pyproj
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
from tqdm import tqdm

# Load the shapefile
sf = shapefile.Reader("Z:/iiixr-drive/Projects/2024_City_Team/CityBoundaries/500Cities_City_11082016/CityBoundaries.shp")

# Initialize a list to hold the bbox info
bbox_info = []

# Initialize the projection from EPSG:3857 to EPSG:4326
proj = pyproj.Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)

# Nominatim 초기화
geolocator = Nominatim(user_agent="my_agent")

# 주(state) 정보를 얻기 위한 함수
def get_state(city_name, lat, lon):
    try:
        location = geolocator.reverse(f"{lat}, {lon}")
        if location and 'address' in location.raw:
            return location.raw['address'].get('state', 'Unknown')
    except GeocoderTimedOut:
        print(f"Timeout for {city_name}")
    return 'Unknown'

# Loop over each record and extract the city name and its bounding box
for shape_record in tqdm(sf.shapeRecords()):
    city_name = shape_record.record[0]
    shape = shape_record.shape
    x_min, y_min, x_max, y_max = shape.bbox  # Bounding box in EPSG:3857

    # Convert the bounding box to EPSG:4326 (lon/lat)
    lon_min, lat_min = proj.transform(x_min, y_min)
    lon_max, lat_max = proj.transform(x_max, y_max)

    # Calculate the center point
    lat = (lat_min + lat_max) / 2
    lon = (lon_min + lon_max) / 2

    # Get the state information
    state = get_state(city_name, lat, lon)
    state = state.replace(" ", "")
    
    # Append the information to the list
    bbox_info.append({
        "city_name": city_name,
        "state": state,
        "x_min": lon_min,
        "x_max": lon_max,
        "y_min": lat_min,
        "y_max": lat_max
    })

# Convert the list to a DataFrame
bbox_df = pd.DataFrame(bbox_info)

# Save the DataFrame to a CSV file
bbox_df.to_csv("Z:/iiixr-drive/Projects/2024_City_Team/CityBoundaries/500Cities_City_11082016/city_bounding_boxes.csv", index=False)

# Display the first few rows to confirm
print(bbox_df.head())
