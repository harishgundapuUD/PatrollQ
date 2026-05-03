import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from mlflow.tracking import MlflowClient

# ----------------------------
# CONFIGURATION & DIRECTORIES
# ----------------------------

MODEL_DIR = "dimensionality_reduction_models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("dimensionality_reduction_plots", exist_ok=True)  # For saving plots

# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv('dataset/dbscan_clustered_crimes.csv')  # Replace with your path

# Extract coordinates for dimensionality reduction
coords = df[['Latitude', 'Longitude']].copy()

# ----------------------------
# MLflow Setup
# ----------------------------

mlruns_path = os.path.abspath(os.path.join(MODEL_DIR, "mlruns"))
os.makedirs(mlruns_path, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")  # MLflow URI
mlflow.set_experiment("crime_dimensionality_reduction")

# ----------------------------
# METRICS LOGGING
# ----------------------------

metrics_log = {
    "models": {
        "dimensionality_reduction": {
            "PCA": {},
            # "UMAP": {}
        }
    }
}

model_metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")

# If metrics file already exists, load it and merge with the new data
if os.path.exists(model_metrics_path):
    with open(model_metrics_path, "r") as f:
        existing_metrics = json.load(f)
    metrics_log = {**metrics_log, **existing_metrics}  # Merge with existing metrics

# ----------------------------
# PCA - Dimensionality Reduction
# ----------------------------

# Initialize PCA with 2 components for visualization
pca = PCA(n_components=2, random_state=42)

# Log PCA parameters and model
with mlflow.start_run(run_name="PCA") as run:
    mlflow.log_param("n_components", 2)
    
    # Fit PCA
    pca.fit(coords)

    # Log PCA model
    mlflow.sklearn.log_model(pca, "PCA_Model")

    # Store model metadata
    metrics_log["models"]["dimensionality_reduction"]["PCA"] = {
        "run_id": run.info.run_id,
        "n_components": 2
    }

    # Save PCA transformed data for plotting
    pca_data = pca.transform(coords)
    df_pca = pd.DataFrame(pca_data, columns=["PCA_Latitude", "PCA_Longitude"])
    df['PCA_Latitude'] = df_pca["PCA_Latitude"]
    df['PCA_Longitude'] = df_pca["PCA_Longitude"]

    # Plot PCA result
    plt.figure(figsize=(10,8))
    plt.scatter(df['PCA_Latitude'], df['PCA_Longitude'], c=df['KMeans_Cluster'], cmap='tab10', s=10)
    plt.xlabel('PCA_Latitude')
    plt.ylabel('PCA_Longitude')
    plt.title('PCA - Crime Clusters')
    pca_plot_path = 'dimensionality_reduction_plots/pca_plot.png'
    plt.savefig(pca_plot_path, bbox_inches='tight')
    mlflow.log_artifact(pca_plot_path)  # Log the PCA plot

# ----------------------------
# UMAP - Dimensionality Reduction
# ----------------------------

# Initialize UMAP with 2 components
# umap_model = umap.UMAP(n_components=2, random_state=42)

# Subsampling to reduce the size of the dataset
# sample_size = 10000  # Adjust the sample size as needed
# subsample_indices = np.random.choice(len(coords), sample_size, replace=False)
# subsampled_coords = coords.iloc[subsample_indices]
# umap_model = umap.UMAP(
#     n_components=2,
#     n_neighbors=15,  # Reduce the number of neighbors if needed
#     metric='euclidean',  # Adjust the metric as necessary
#     random_state=42,
#     n_jobs=-1,  # Use multiple cores if available
#     # method='umap-learn',  # Use the default method (optimize for speed)
#     target_n_neighbors=5  # This helps to speed up computations with approximate neighbors
# )

# umap_model.fit(subsampled_coords)  # Apply UMAP on the subsampled data

# mlflow.end_run()
# # Log UMAP parameters and model
# with mlflow.start_run(run_name="UMAP") as run:
#     mlflow.log_param("n_components", 2)
    
#     # Fit UMAP model
#     umap_model.fit(coords)

#     # Log UMAP model
#     mlflow.sklearn.log_model(umap_model, "UMAP_Model")

#     # Store model metadata
#     metrics_log["models"]["dimensionality_reduction"]["UMAP"] = {
#         "run_id": run.info.run_id,
#         "n_components": 2
#     }

#     # Save UMAP transformed data for plotting
#     umap_data = umap_model.transform(coords)
#     df_umap = pd.DataFrame(umap_data, columns=["UMAP_Latitude", "UMAP_Longitude"])
#     df['UMAP_Latitude'] = df_umap["UMAP_Latitude"]
#     df['UMAP_Longitude'] = df_umap["UMAP_Longitude"]

#     # Plot UMAP result
#     plt.figure(figsize=(10,8))
#     plt.scatter(df['UMAP_Latitude'], df['UMAP_Longitude'], c=df['KMeans_Cluster'], cmap='tab10', s=10)
#     plt.xlabel('UMAP_Latitude')
#     plt.ylabel('UMAP_Longitude')
#     plt.title('UMAP - Crime Clusters')
#     umap_plot_path = 'dimensionality_reduction_plots/umap_plot.png'
#     plt.savefig(umap_plot_path, bbox_inches='tight')
#     mlflow.log_artifact(umap_plot_path)  # Log the UMAP plot

# ----------------------------
# FINAL OUTPUT
# ----------------------------

# Save the updated metrics to the JSON file
with open(model_metrics_path, "w") as f:
    json.dump(metrics_log, f, indent=4)
mlflow.end_run()
print("Dimensionality reduction models saved and metrics logged.")