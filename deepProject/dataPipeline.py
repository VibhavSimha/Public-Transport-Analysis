import overpass
import pandas as pd
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime

# Define Bengaluru bounding box (south, west, north, east)
BENGALURU_BBOX = (12.8, 77.4, 13.2, 77.8)

def get_osm_data():
    """Collect transport data from OpenStreetMap"""
    api = overpass.API(timeout=600)  # Increased timeout
    
    # Correct Overpass QL query
    query = f"""
    [out:json];
    (
        node["highway"="bus_stop"]({BENGALURU_BBOX[0]},{BENGALURU_BBOX[1]},{BENGALURU_BBOX[2]},{BENGALURU_BBOX[3]});
        node["railway"="station"]["station"~"subway|suburban"]({BENGALURU_BBOX[0]},{BENGALURU_BBOX[1]},{BENGALURU_BBOX[2]},{BENGALURU_BBOX[3]});
    );
    out meta;
    """
    
    return api.get(query)

def process_osm_data(response):
    """Process OSM response into structured DataFrame"""
    data = []
    
    for element in response['elements']:
        if element['type'] != 'node':
            continue
            
        feature = {
            'id': element['id'],
            'lat': element['lat'],
            'lon': element['lon'],
            'timestamp': datetime.strptime(element['timestamp'], '%Y-%m-%dT%H:%M:%SZ'),
            'type': None,
            'name': element.get('tags', {}).get('name', '')
        }
        
        tags = element.get('tags', {})
        if tags.get('highway') == 'bus_stop':
            feature['type'] = 'bus'
        elif tags.get('railway') == 'station':
            station_type = tags.get('station', '').lower()
            if 'subway' in station_type:
                feature['type'] = 'metro'
            elif 'suburban' in station_type:
                feature['type'] = 'suburban'
        
        if feature['type']:
            data.append(feature)
            
    return pd.DataFrame(data)

# Rest of the code remains the same...
# Execute data pipeline
try:
    response = get_osm_data()
    df = process_osm_data(response)
    df.to_csv('bengaluru_transport.csv', index=False)
    print(f"Successfully collected {len(df)} transport nodes")
except Exception as e:
    print(f"Error: {str(e)}")
    exit()

# Rest of the visualization code remains the same...

#Timeline map:

def create_timeline_map(df):
    """Create interactive map with time slider"""
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
    
    features = []
    for _, row in df.iterrows():
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [row['lon'], row['lat']]
            },
            'properties': {
                'time': row['timestamp'].date().isoformat(),
                'style': {'color': get_color(row['type'])},
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': get_color(row['type']),
                    'fillOpacity': 0.8,
                    'radius': 8
                },
                'popup': f"Type: {row['type']}<br>Date: {row['timestamp'].date()}"
            }
        })
    
    TimestampedGeoJson(
        {'type': 'FeatureCollection', 'features': features},
        period='P1Y',
        add_last_point=True,
        auto_play=False,
        transition_time=1000
    ).add_to(m)
    
    return m

def get_color(transport_type):
    return {
        'bus': 'blue',
        'metro': 'green',
        'suburban': 'red'
    }.get(transport_type, 'gray')

# Generate and save map
m = create_timeline_map(df)
m.save('transport_evolution.html')