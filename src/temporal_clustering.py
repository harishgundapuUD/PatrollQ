import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient

# Directories for storing models and results
MODEL_DIR = "temporal_clustering_models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("clustering_plots", exist_ok=True)  # For saving clustering plots and results

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv('dataset/cleaned_crimes.csv')  # Replace with your path

# Extract temporal features: hour, day, and month
# df['Date_hour'] = pd.to_datetime(df['Date']).dt.hour
# df['Day'] = pd.to_datetime(df['Date']).dt.dayofweek  # 0 = Monday, 6 = Sunday
# df['Month'] = pd.to_datetime(df['Date']).dt.month

# Temporal feature set for clustering
temporal_features = df[['Date_hour', 'Date_day', 'Date_month']]

# ----------------------------
# MLflow Setup
# ----------------------------
mlruns_path = os.path.abspath(os.path.join(MODEL_DIR, "mlruns"))
os.makedirs(mlruns_path, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
mlflow.set_experiment("temporal_crime_clustering")

# ----------------------------
# K-Means Clustering on Temporal Features
# ----------------------------

# Preprocess features (optional: scale the features)
scaler = StandardScaler()
temporal_features_scaled = scaler.fit_transform(temporal_features)

# Use a fixed number of clusters, for example, 10
kmeans = KMeans(n_clusters=10, random_state=42, max_iter=500)

# Fit the KMeans model
kmeans.fit(temporal_features_scaled)
df['Temporal_Cluster'] = kmeans.labels_

# ----------------------------
# Log the model and its parameters with MLflow
# ----------------------------
mlflow.end_run()
# Log the KMeans model and its parameters
with mlflow.start_run(run_name="KMeans Temporal Clustering") as run:
    mlflow.log_param("n_clusters", 10)
    mlflow.log_param("init", "k-means++")  # Default initialization method
    mlflow.log_param("max_iter", 500)
    
    # Log the trained KMeans model
    mlflow.sklearn.log_model(kmeans, "KMeans_Temporal_Model")
    
    # Log the model's cluster centers as an artifact
    cluster_centers_path = 'clustering_plots/kmeans_temporal_centers.png'
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Date_hour'], df['Date_day'], c=df['Temporal_Cluster'], cmap='tab10', s=10)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='x', s=100, label="Centroids")
    plt.xlabel("Hour")
    plt.ylabel("Day of Week")
    plt.title("KMeans Temporal Clusters (Hour vs Day)")
    plt.legend()
    plt.savefig(cluster_centers_path, bbox_inches='tight')
    mlflow.log_artifact(cluster_centers_path)  # Log the plot as an artifact
    plt.close()

    # Save the clustered dataset as an artifact (optional)
    df_clustered_path = 'dataset/kmeans_temporal_clustered_crimes.csv'
    df.to_csv(df_clustered_path, index=False)
    mlflow.log_artifact(df_clustered_path)  # Log the clustered data as an artifact

    # Save the run details into a JSON file
    run_metadata = {
        "run_id": run.info.run_id,
        "params": {
            "n_clusters": 10,
            "init": "k-means++",
            "max_iter": 500
        },
        "model_artifacts": {
            "model": "KMeans_Temporal_Model",
            "cluster_centers_image": cluster_centers_path,
            "clustered_data": df_clustered_path
        }
    }

    # Path to save the metadata JSON file
    metadata_json_path = os.path.join(MODEL_DIR, f"run_{run.info.run_id}_metadata.json")
    
    # Write the run metadata to JSON
    with open(metadata_json_path, "w") as json_file:
        json.dump(run_metadata, json_file, indent=4)

# ----------------------------
# FINAL OUTPUT
# ----------------------------
mlflow.end_run()
print("K-Means Temporal Clustering complete.")