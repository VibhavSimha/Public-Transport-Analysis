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