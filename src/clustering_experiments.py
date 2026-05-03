import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from mlflow.tracking import MlflowClient
from sklearn.metrics import silhouette_score, davies_bouldin_score


MODEL_DIR = "clustering_models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("clustering_plots", exist_ok=True)  # For saving clustering plots and results

# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv('dataset/cleaned_crimes.csv')  # Replace with your path

# Extract coordinates for clustering
coords = df[['Latitude', 'Longitude']].copy()

# ----------------------------
# MLflow Setup
# ----------------------------

mlruns_path = os.path.abspath(os.path.join(MODEL_DIR, "mlruns"))
os.makedirs(mlruns_path, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
mlflow.set_experiment("crime_clustering")


# Function to calculate silhouette score with subsampling
def calculate_silhouette_score_subsample(X, labels, sample_size=20000):
    """Calculate silhouette score on a random subsample of the data"""
    if len(X) > sample_size:
        # Randomly choose indices from the dataframe
        subsample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_subsample = X.iloc[subsample_indices]  # Use .iloc to index rows by position
        labels_subsample = labels.iloc[subsample_indices]  # Same for labels
    else:
        X_subsample = X
        labels_subsample = labels
    
    return silhouette_score(X_subsample, labels_subsample)


# ----------------------------
# METRICS LOGGING
# ----------------------------

# Initialize results dictionary for logging metrics into a JSON file
metrics_log = {
    "models": {"clustering": {}}
}

# Path to save metrics JSON file
model_metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")

# Check if there's already an existing metrics file and load it
if os.path.exists(model_metrics_path):
    with open(model_metrics_path, "r") as f:
        existing_metrics = json.load(f)
    metrics_log = {**metrics_log, **existing_metrics}  # Merge with existing metrics

# ----------------------------
# KMeans Clustering
# ----------------------------

# Elbow method to choose optimal K (1 to 15 clusters)
inertia = []
K = range(1, 15)  # Cluster range

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(coords)
    inertia.append(kmeans.inertia_)

# Log KMeans elbow method graph
plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Sum of squared distances)')
plt.title('Elbow Method to choose k')
elbow_method_path = 'clustering_plots/elbow_method.png'
plt.savefig(elbow_method_path, bbox_inches='tight')
mlflow.log_artifact(elbow_method_path)  # Log the elbow method plot
plt.close()

# Choose k=7 based on elbow method
kmeans = KMeans(n_clusters=7, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(coords)

# Evaluate KMeans using Silhouette Score and Davies-Bouldin Index
sil_score_kmeans = calculate_silhouette_score_subsample(coords, df['KMeans_Cluster'])
db_score_kmeans = davies_bouldin_score(coords, df['KMeans_Cluster'])


# Ensure any previous run is ended before starting a new one
mlflow.end_run()

# Log KMeans model and parameters
with mlflow.start_run(run_name="KMeans Clustering") as run:
    mlflow.log_param("k", 10)
    mlflow.sklearn.log_model(kmeans, "KMeans_Model")

    # Log model metrics (in this case, inertia)
    metrics_log["models"]["clustering"]["KMeans"] = {
        "run_id": run.info.run_id,
        "k": 10,
        "silhouette_score": sil_score_kmeans,
        "davies_bouldin_index": db_score_kmeans,
        "inertia": kmeans.inertia_,
        "bestmodel": "yes"
    }
    print(f"The silhouette score for KMeans is: {sil_score_kmeans}")
    print(f"The Davies-Bouldin index for KMeans is: {db_score_kmeans}")

    # Save the KMeans clustered dataset
    df.to_csv('dataset/kmeans_clustered_crimes.csv', index=False)

    # Plot KMeans clusters
    plt.figure(figsize=(10,8))
    plt.scatter(df['Longitude'], df['Latitude'], c=df['KMeans_Cluster'], cmap='tab10', s=10)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Crime Clusters - KMeans')
    kmeans_cluster_path = 'clustering_plots/kmeans_crime_clusters.png'
    plt.savefig(kmeans_cluster_path, bbox_inches='tight')
    mlflow.log_artifact(kmeans_cluster_path)  # Log the KMeans cluster plot
    plt.close()

# ----------------------------
# DBSCAN Clustering
# ----------------------------

# Convert coordinates to radians for Haversine distance (for DBSCAN)
coords_rad = np.radians(coords)

# Set DBSCAN parameters
eps = 0.5 / 6371  # 0.5 km radius (in radians)
min_samples = 10

# Log DBSCAN parameters
with mlflow.start_run(run_name="DBSCAN Clustering") as run:
    mlflow.log_param("DBSCAN_eps", eps)
    mlflow.log_param("DBSCAN_min_samples", min_samples)

    # Run DBSCAN
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='haversine',
        algorithm='ball_tree',
        n_jobs=-1
    )
    df['DBSCAN_Cluster'] = dbscan.fit_predict(coords)

    # Evaluate DBSCAN using Silhouette Score and Davies-Bouldin Index
    if len(np.unique(df['DBSCAN_Cluster'])) > 1:  # DBSCAN might have only one cluster (if no noise)
        sil_score_dbscan = calculate_silhouette_score_subsample(coords, df['DBSCAN_Cluster'])
        db_score_dbscan = davies_bouldin_score(coords, df['DBSCAN_Cluster'])
    else:
        sil_score_dbscan = -1  # Invalid score for a single cluster
        db_score_dbscan = -1

    # Separate noise and clusters
    noise = df[df['DBSCAN_Cluster'] == -1]
    clusters = df[df['DBSCAN_Cluster'] != -1]

    # Plot DBSCAN clusters
    plt.figure(figsize=(10,8))
    plt.scatter(noise['Longitude'], noise['Latitude'], c='lightgrey', s=2, label='Noise')
    plt.scatter(clusters['Longitude'], clusters['Latitude'], c=clusters['DBSCAN_Cluster'], cmap='tab10', s=5)
    plt.legend()
    plt.title("DBSCAN Crime Hotspots")
    dbscan_cluster_path = 'clustering_plots/dbscan_crime_clusters.png'
    plt.savefig(dbscan_cluster_path, bbox_inches='tight')
    mlflow.log_artifact(dbscan_cluster_path)  # Log the DBSCAN cluster plot
    plt.close()

    # Log DBSCAN model
    mlflow.sklearn.log_model(dbscan, "DBSCAN_Model")

    # Log DBSCAN metrics
    metrics_log["models"]["clustering"]["DBSCAN"] = {
        "run_id": run.info.run_id,
        "eps": eps,
        "min_samples": min_samples,
        "silhouette_score": sil_score_dbscan,
        "davies_bouldin_index": db_score_dbscan,
        "bestmodel": "no"
    }
    if sil_score_dbscan > sil_score_kmeans:
        metrics_log["models"]["clustering"]["KMeans"]["bestmodel"] = "no"
        metrics_log["models"]["clustering"]["DBSCAN"]["bestmodel"] = "yes"
    print(f"The silhouette score for DBSCAN is: {sil_score_dbscan}")
    print(f"The Davies-Bouldin index for DBSCAN is: {db_score_dbscan}")

    # Save the DBSCAN clustered dataset
    df.to_csv('dataset/dbscan_clustered_crimes.csv', index=False)

# ----------------------------
# FINAL OUTPUT
# ----------------------------

# Save all metrics to a JSON file
with open(model_metrics_path, "w") as f:
    json.dump(metrics_log, f, indent=4)

print("Clustering complete.")