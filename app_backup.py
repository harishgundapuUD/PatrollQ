import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import numpy as np
from folium.plugins import HeatMap
from sklearn.cluster import KMeans

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/dbscan_clustered_crimes.csv')
    return df

# Cache the KMeans clustering to avoid refitting the model on each interaction
@st.cache_resource
def apply_kmeans(df, n_clusters=7):
    coords = df[['Latitude', 'Longitude']].copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['KMeans_Cluster'] = kmeans.fit_predict(coords)
    return df, kmeans

# Load the dataset (cached)
df = load_data()

# Cache the creation of the heatmap data so that it is not recomputed every time
@st.cache_data
def create_heatmap_data(df_subsample):
    return [[row['Latitude'], row['Longitude']] for index, row in df_subsample.iterrows()]

# Streamlit App Header
st.title("Geographic Crime Heatmap with Clusters")

# Sidebar for navigation (new page selection)
page = st.sidebar.radio("Select a Page", ["Crime Heatmap", "KMeans Clustering Image"])

if page == "Crime Heatmap":
    # Optimize by using a subset of data
    subsample_size = 10000  # You can adjust this value based on performance vs accuracy tradeoff
    df_subsample = df.sample(n=subsample_size, random_state=42)

    # If clustering is not available, apply KMeans clustering (only once)
    if 'KMeans_Cluster' not in df.columns:
        df, kmeans = apply_kmeans(df)
    else:
        # If KMeans clustering is already done, use the existing model to get cluster centers
        kmeans = KMeans(n_clusters=7, random_state=42)
        kmeans.cluster_centers_ = np.array([
            df[df['KMeans_Cluster'] == i][['Latitude', 'Longitude']].mean().values for i in range(7)
        ])

    # Create a Folium Map centered around the average coordinates of the dataset
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    crime_map = folium.Map(location=map_center, zoom_start=12)

    # Create a HeatMap layer for subsample (downsampled data)
    heat_data = create_heatmap_data(df_subsample)
    HeatMap(heat_data).add_to(crime_map)

    # Add Cluster Boundaries to the map (only for the downsampled data)
    for cluster_id in df_subsample['KMeans_Cluster'].unique():
        cluster_data = df_subsample[df_subsample['KMeans_Cluster'] == cluster_id]
        for index, row in cluster_data.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6
            ).add_to(crime_map)

    # Add Cluster Centers (Optional)
    for center in kmeans.cluster_centers_:
        folium.Marker(
            location=[center[0], center[1]],
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(crime_map)

    # Display map in Streamlit using st_folium
    st_folium(crime_map, width=700, height=500)

elif page == "KMeans Clustering Image":
    # Display the saved KMeans clustering image directly
    kmeans_image_path = "clustering_plots/kmeans_crime_clusters.png"  # Path to your image
    st.image(kmeans_image_path, caption='KMeans Clusters on Crime Data', use_container_width=True)