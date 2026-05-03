
# PatrollQ - Crime Data Clustering and Visualization

PatrollQ is a crime analysis and clustering application built using various machine learning techniques such as KMeans and DBSCAN. The app processes and clusters geographic crime data, visualizes it on an interactive map, and allows the prediction of crime clusters based on user input coordinates. It is built using **Streamlit**, **Scikit-learn**, **Pandas**, **Folium**, and **MLflow**.

## Features

- **Geographic Crime Heatmap**: Visualizes crime data on an interactive map with heatmaps to show crime density and clusters.
- **Clustering**: Uses KMeans and DBSCAN clustering algorithms to categorize crimes based on geographic locations.
- **Cluster Prediction**: Allows users to input geographic coordinates (latitude, longitude) and predict which cluster the crime belongs to.
- **Clustering Model and Metrics Logging**: Logs and tracks models and metrics using MLflow, enabling easy model versioning and monitoring.

## Tech Stack

- **Streamlit**: For creating the interactive user interface.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For applying machine learning models such as KMeans and DBSCAN.
- **Folium**: For creating interactive maps and visualizing crime data on a geographic map.
- **MLflow**: For model tracking, versioning, and logging metrics.

## Getting Started

To get started with PatrollQ, you need to set up the project locally. Follow these steps:

### Prerequisites

Ensure that you have **Python 3.7+** installed on your system. You will also need **pip** (Python package manager).

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/harishgundapuUD/PatrollQ.git
cd PatrollQ
```


### 2. Set up a Virtual Environment (Optional but Recommended)

It is recommended to create a virtual environment to manage dependencies for this project.

#### Using `venv`:

<pre class="overflow-visible! px-0!" data-start="2229" data-end="2326"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>python </span><span class="ͼn">-m</span><span> venv venv</span><br/><span class="ͼl">source</span><span> venv/bin/activate  </span><span class="ͼe"># On Windows, use venv\Scripts\activate</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

### 3. Install Dependencies

Install all the necessary dependencies using `pip`:

<pre class="overflow-visible! px-0!" data-start="2410" data-end="2453"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>pip install </span><span class="ͼn">-r</span><span> requirements.txt</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

If you don’t have the `requirements.txt` file, you can install the dependencies manually:

<pre class="overflow-visible! px-0!" data-start="2546" data-end="2630"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>pip install streamlit pandas numpy scikit-learn folium mlflow matplotlib</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

### 4. Download Dataset

Place the `cleaned_crimes.csv` dataset in the `dataset/` directory. If you don't have the dataset, you can download it from [here]().

### 5. Run the Application

Once the dependencies are installed and the dataset is in place, you can start the Streamlit app by running:

<pre class="overflow-visible! px-0!" data-start="2946" data-end="2978"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>streamlit run app.py</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

This will launch the Streamlit app in your default browser.


## Folder Structure

The folder structure of the project is as follows:

<pre class="overflow-visible! px-0!" data-start="3983" data-end="4600"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>PatrollQ/</span><br/><span>│</span><br/><span>├── dataset/</span><br/><span>│   ├── cleaned_crimes.csv       # Input crime data</span><br/><span>│   ├── dbscan_clustered_crimes.csv # Clustered crime data (if available)</span><br/><span>│</span><br/><span>├── clustering_plots/            # Plots for clustering results</span><br/><span>│   ├── kmeans_crime_clusters.png  # KMeans clustering result plot</span><br/><span>│   ├── dbscan_crime_clusters.png  # DBSCAN clustering result plot</span><br/><span>│   ├── elbow_method.png          # Elbow method plot for determining optimal clusters</span><br/><span>│</span><br/><span>├── app.py                       # Streamlit application code</span><br/><span>├── requirements.txt             # List of dependencies</span><br/><span>└── README.md                    # This README file</span></code></pre></div></div></div></div></div></div></div></div></div></div></div></div></pre>
