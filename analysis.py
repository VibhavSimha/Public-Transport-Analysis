import os
import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
from sklearn.linear_model import LogisticRegression
import folium
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# --- FIX: Moved st.set_page_config() to the top level of the script ---
# This must be the first Streamlit command.
st.set_page_config(layout="wide", page_title="Bengaluru Metro Analysis")

class ComprehensiveMetroAnalyzer:
    """
    Final, optimized metro expansion prediction and analysis system.
    """
    
    def __init__(self, road_network_file='road_network.geojson', data_path='.'):
        self.road_network_file = road_network_file
        self.data_path = data_path
        self.G = None
        self.snaps = {}
        self.ts = None
        self.model = None
        self.predictions = {}
        self.feeder_analysis = []
        
    def build_road_graph(self):
        print("Building road network graph...")
        roads = gpd.read_file(self.road_network_file)
        self.G = nx.Graph()
        
        for r in roads.itertuples():
            geom = getattr(r, 'geometry', None)
            if geom and geom.geom_type == 'LineString':
                prev = None
                for lon, lat in geom.coords:
                    node = (round(lat, 6), round(lon, 6))
                    self.G.add_node(node)
                    if prev:
                        dist = geodesic(prev, node).meters
                        self.G.add_edge(prev, node, weight=dist)
                    prev = node
        
        print(f"Built road graph: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        return self.G

    def load_historical_data(self, years):
        print("Loading historical metro data...")
        self.snaps = {}
        
        for y in years:
            f = os.path.join(self.data_path, f"data_{y}.geojson")
            coords = []
            if os.path.exists(f):
                gdf = gpd.read_file(f)
                for _, row in gdf.iterrows():
                    geom = row.geometry
                    if geom.geom_type == 'Point' and (row.get('network') == 'Namma Metro' or row.get('station') == 'subway'):
                        coords.append((geom.y, geom.x))
            self.snaps[y] = coords
            if coords:
                print(f"Loaded {len(coords)} metro stations for {y}")
        return self.snaps

    def make_tree(self, nodes):
        arr = np.array(nodes)
        tree = BallTree(np.deg2rad(arr), metric='haversine')
        return tree, arr

    def build_time_series(self, years):
        print("Building time series...")
        nodes = list(self.G.nodes)
        idx = pd.MultiIndex.from_product([nodes, years], names=('node','year'))
        self.ts = pd.DataFrame(0, index=idx, columns=['station'])
        
        tree, arr = self.make_tree(nodes)
        
        for y in years:
            coords = self.snaps[y]
            if coords:
                _, ix = tree.query(np.deg2rad(np.array(coords)), k=1)
                for i in ix.flatten():
                    self.ts.at[(tuple(arr[i]), y), 'station'] = 1
        
        return self.ts

    def train_prediction_model(self, years):
        print(f"Training model up to {years[-1]}...")
        rows = []
        for node in self.ts.index.get_level_values('node').unique():
            s = self.ts.xs(node, level=0)['station']
            cum = 0
            for y1, y2 in zip(years, years[1:]):
                cum += s.get(y1, 0)
                rows.append({'past': cum, 'year': y1, 'target': s.get(y2, 0)})
        df = pd.DataFrame(rows)
        if df['target'].nunique() < 2:
            print("Warning: Insufficient training data variance. Using previous model.")
            return None
        
        X = df[['past','year']].values
        y = df['target'].values
        self.model = LogisticRegression(solver='liblinear').fit(X, y)
        
        return self.model

    def predict_metro_lines(self, year, start_year, count, min_dist, global_nodes):
        cand, sel, lines = [], [], []
        
        for node in self.G.nodes:
            if node in global_nodes: continue
            past = self.ts.xs(node, level=0)['station'].loc[start_year:year-1].sum()
            if past > 0: continue
            p = self.model.predict_proba([[past, year]])[0,1] if self.model else 0
            cand.append((node, p))
        
        cand.sort(key=lambda x: -x[1])
        
        rads = [np.deg2rad(n) for n in global_nodes]
        tree = BallTree(np.array(rads), metric='haversine') if rads else None
        
        for node, p in cand:
            if len(sel) >= count: break
            rad = np.deg2rad(node)
            if tree:
                d = tree.query([rad], k=1)[0][0][0] * 6371000
                if d < min_dist: continue
            sel.append((node, p))
            rads.append(rad)
            tree = BallTree(np.array(rads), metric='haversine')
        
        prev = global_nodes[-1] if global_nodes else None
        for node, _ in sel:
            if prev and nx.has_path(self.G, prev, node):
                try: lines.append(nx.shortest_path(self.G, prev, node, weight='weight'))
                except: lines.append([])
            else: lines.append([])
            prev = node
        return sel, lines

    def run_prediction_pipeline(self, history_years, future_years):
        print("Running prediction pipeline...")
        self.build_road_graph()
        self.load_historical_data(history_years + [2025])
        self.build_time_series(history_years)
        self.train_prediction_model(history_years)
        print("Initial model trained successfully.")
        
        preds, lines, global_nodes = [], [], [n for n, s in self.ts.xs(2025, level=1)['station'].items() if s==1]
        
        total_needed = max(1, int(len(self.G.nodes) * 0.02))
        per_year = total_needed // len(future_years)
        remaining = total_needed % len(future_years)
        
        history = list(history_years)
        for i, y in enumerate(future_years):
            count = per_year + (1 if i < remaining else 0)
            sel, ln = self.predict_metro_lines(y, history[0], count, 1000, global_nodes)
            preds.append(sel)
            lines.append(ln)
            self.predictions[y] = sel
            global_nodes.extend([node for node,_ in sel])
            new_data = pd.DataFrame({'station':1}, index=pd.MultiIndex.from_product([[n for n,_ in sel], [y]], names=('node','year')))
            self.ts = pd.concat([self.ts, new_data])
            history.append(y)
            new_model = self.train_prediction_model(history)
            if new_model: self.model = new_model
        return preds, lines, global_nodes

    def analyze_metro_feeder_bus_stops_farthest(self, existing_stations, top_n=100):
        print("\nOptimized Metro Feeder Bus Stop Analysis (Farthest Stops)")
        bus_stops = self.load_bus_stops()
        all_metro_stations = list(existing_stations)
        for year_stations in self.predictions.values():
            all_metro_stations.extend([node for node, _ in year_stations])
        
        bus_coords = np.array([stop['coordinates'] for stop in bus_stops])
        metro_coords = np.array(all_metro_stations)
        
        metro_tree = BallTree(np.deg2rad(metro_coords), metric='haversine')
        
        distances, indices = metro_tree.query(np.deg2rad(bus_coords), k=1)
        min_distances_meters = distances.flatten() * 6371000
        
        for i, bus_stop in enumerate(bus_stops):
            bus_stop['distance_to_metro'] = min_distances_meters[i]
            bus_stop['nearest_metro'] = tuple(metro_coords[indices[i][0]])
        
        sorted_bus_stops = sorted(bus_stops, key=lambda x: x['distance_to_metro'], reverse=True)
        priority_feeders = sorted_bus_stops[:top_n]
        
        self.feeder_analysis = []
        for bus_stop in priority_feeders:
            self.feeder_analysis.append({
                'bus_stop': bus_stop,
                'nearest_metro': bus_stop['nearest_metro'],
                'distance_to_metro': bus_stop['distance_to_metro'],
                'is_priority': True
            })
        
        print(f"Identified {len(self.feeder_analysis)} priority feeder bus stops.")
        return self.feeder_analysis

    def load_bus_stops(self, data_file='data_2025.geojson'):
        gdf = gpd.read_file(data_file)
        bus_stops = []
        for _, row in gdf.iterrows():
            if (row.get('highway') == 'bus_stop' and 
                hasattr(row, 'geometry') and 
                row.geometry.geom_type == 'Point'):
                bus_stops.append({
                    'coordinates': (row.geometry.y, row.geometry.x),
                    'name': row.get('name', 'Unknown Bus Stop')
                })
        return bus_stops

# Use Streamlit's caching to run the expensive computations only once.
@st.cache_data
def run_full_analysis():
    """
    This function runs the entire prediction and analysis pipeline and caches the result.
    It will only run the first time the app is loaded or if the code changes.
    """
    analyzer = ComprehensiveMetroAnalyzer()
    history_years = list(range(2017, 2026))
    future_years = list(range(2026, 2036))
    
    preds, lines, existing_stations = analyzer.run_prediction_pipeline(history_years, future_years)
    
    # Run feeder analysis on both existing (2025) and predicted stations
    analyzer.analyze_metro_feeder_bus_stops_farthest(existing_stations, top_n=100)
    
    return analyzer, existing_stations, future_years

def create_dashboard(analyzer, existing_stations, future_years):
    """
    Creates the Streamlit UI. This part is fast and re-runs on user interaction.
    """
    st.title("Bengaluru Metro Expansion & Feeder Bus Analysis Dashboard")
    
    st.sidebar.header("Dashboard Controls")
    
    years_to_show = st.sidebar.multiselect(
        "Select Prediction Years to Display",
        future_years,
        default=future_years[:3]
    )
    
    show_feeders = st.sidebar.checkbox("Show Priority Feeder Bus Stops", value=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Metro Network Prediction Map")
        
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=10)
        
        # Existing metro layer
        existing_layer = folium.FeatureGroup(name="Existing Metro Stations", show=True)
        for station in existing_stations:
            folium.CircleMarker(
                location=station, 
                radius=4, 
                color='blue', 
                fill=True, 
                popup="Existing Metro Station"
            ).add_to(existing_layer)
        existing_layer.add_to(m)
        
        # Predicted metro layers
        colors = ['red', 'green', 'purple', 'orange', 'darkred', 'darkgreen', 'pink', 'gray', 'brown', 'olive']
        
        for i, year in enumerate(future_years):
            if year in analyzer.predictions and year in years_to_show:
                layer = folium.FeatureGroup(name=f'Predicted {year}')
                color = colors[i % len(colors)]
                
                for node, prob in analyzer.predictions[year]:
                    folium.CircleMarker(
                        location=node,
                        radius=6,
                        color=color,
                        fill=True,
                        popup=f"Predicted {year}: Probability {prob:.3f}"
                    ).add_to(layer)
                
                layer.add_to(m)
        
        # Feeder bus analysis layer
        if show_feeders and analyzer.feeder_analysis:
            priority_feeders_layer = folium.FeatureGroup(name='Priority Feeder Bus Stops')
            
            for analysis in analyzer.feeder_analysis:
                bus_stop = analysis['bus_stop']
                coord = bus_stop['coordinates']
                
                folium.Circle(
                    coord,
                    radius=800,
                    color='green',
                    fill=True,
                    fillOpacity=0.1,
                    popup=f"Feeder Catchment: {bus_stop['name']}"
                ).add_to(priority_feeders_layer)
                
                folium.CircleMarker(
                    coord,
                    radius=8,
                    color='green',
                    fill=True,
                    popup=f"Priority Feeder: {bus_stop['name']}"
                ).add_to(priority_feeders_layer)
            
            priority_feeders_layer.add_to(m)
        
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width=900, height=700)
    
    with col2:
        st.header("Analysis Metrics")
        
        total_predicted = sum(len(analyzer.predictions.get(year, [])) for year in future_years)
        priority_feeders_count = len(analyzer.feeder_analysis)
        
        st.metric("Total Predicted Stations", total_predicted)
        st.metric("Priority Feeder Stops Identified", priority_feeders_count)
        
        st.subheader("Yearly Expansion Rate")
        yearly_growth = [len(analyzer.predictions.get(year, [])) for year in future_years]
        chart_data = pd.DataFrame({'Year': future_years, 'New Stations': yearly_growth})
        st.bar_chart(chart_data.set_index('Year'))
        
        # **FIXED HISTOGRAM**
        st.subheader("Distance to Metro for Priority Feeders")
        feeder_distances = [analysis['distance_to_metro'] for analysis in analyzer.feeder_analysis]
        fig, ax = plt.subplots()
        ax.hist(feeder_distances, bins=20, color='green', edgecolor='black')
        ax.set_xlabel('Distance to Nearest Metro (meters)')
        ax.set_ylabel('Number of Bus Stops')
        ax.set_title('Distribution of Farthest Bus Stops')
        st.pyplot(fig)
        
        st.subheader("AI Prediction Summary")
        st.write("""
            This dashboard visualizes AI-driven metro expansion. The model learns from historical data to predict growth along the road network.
            The feeder analysis identifies the top 100 bus stops **farthest** from the metro network, highlighting critical areas for new feeder services.
        """)

if __name__ == '__main__':
    # Run the heavy computation once and cache the results
    analyzer_instance, existing_stations_data, future_years_data = run_full_analysis()
    
    # Create the Streamlit UI (this part re-runs on interaction)
    create_dashboard(analyzer_instance, existing_stations_data, future_years_data)
