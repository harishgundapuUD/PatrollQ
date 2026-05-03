# 🚓 PatrollQ – Smart Patrol & Incident Analytics System

An end-to-end data-driven system designed to **analyze, cluster, and optimize patrol operations** using geographic and temporal insights.

This project combines **data preprocessing, exploratory analysis, clustering, and intelligent insights generation** to support better decision-making in patrol management systems.

---

## 🚀 Features

- 🔹 Data preprocessing pipeline for structured & temporal data
- 🔹 Exploratory Data Analysis (EDA) with visual insights
- 🔹 Geographic clustering for patrol optimization
- 🔹 Pattern discovery in incidents (time, location, frequency)
- 🔹 Scalable modular architecture
- 🔹 Ready-to-extend ML pipeline

---

## 🏗️ Project Structure

```bash
PatrollQ/

├── dataset/
│   ├── crimes.csv
│   ├── cleaned_crimes.csv
│
├── eda/
│   ├── *.png
│   ├── *.csv
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── crimes_clustering.py
│   ├── temporal_clustering.py
│   ├── dimensionality_reduction.py
│
├── utils/
│   ├── config.json/
│
├── app.py
├── requirements.txt
├── README.md
```


## 🧠 System Pipeline

### 1. Data Preprocessing

* Data cleaning & normalization
* Date-time parsing and transformation
* Handling missing/null values
* Feature creation:
* Time-based features (hour, day, trends)
* Location-based encoding
* Incident density metrics

---

### 2. Exploratory Data Analysis (EDA)

* Incident distribution over time
* Geographic heatmaps
* Correlation analysis
* Trend identification

Outputs stored in:

<pre class="overflow-visible! px-0!" data-start="1687" data-end="1705"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="relative"><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>*_plots/</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

### 3. Geographic Clustering

* Algorithms:
* K-Means
* DBSCAN (optional)
* Identifies:
* High-risk zones
* Patrol priority regions

---

### 4. Insights Generation

* Peak incident hours
* High-density locations
* Patrol optimization suggestions

---

## ⚙️ Installation

### 1. Clone the Repository

<pre class="overflow-visible! px-0!" data-start="2078" data-end="2159"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span class="ͼl">git</span><span> clone https://github.com/harishgundapuUD/PatrollQ.git</span><br/><span class="ͼl">cd</span><span> PatrollQ</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

### 2. Create Virtual Environment

<pre class="overflow-visible! px-0!" data-start="2201" data-end="2266"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>python </span><span class="ͼn">-m</span><span> venv venv</span><br/><span>venv\Scripts\activate   </span><span class="ͼe"># Windows</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

### 3. Install Dependencies

<pre class="overflow-visible! px-0!" data-start="2302" data-end="2345"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>pip install </span><span class="ͼn">-r</span><span> requirements.txt</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

## ▶️ Run the Project

<pre class="overflow-visible! px-0!" data-start="2375" data-end="2401"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼd ͼr"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>python -m streamlit app.py</span></code></pre></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

## 📊 Example Use Cases

* 🚓 Patrol route optimization
* 📍 Crime hotspot detection
* ⏱️ Time-based incident analysis
* 📈 Resource allocation planning

---

## 📈 Outputs

* Clustered geographic zones
* Analytical visualizations
* Actionable patrol insights

---

## 🛠️ Technologies Used

* Python
* Pandas / NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Geospatial libraries

---

## 🚀 Future Improvements

* 🌐 Interactive dashboard (Streamlit / Dash)
* 📍 Real-time tracking integration
* 🤖 Predictive modeling
* 🧠 AI-based patrol recommendation system
* ☁️ Cloud deployment

---

## 👨‍💻 Author

**Harish Gundapu**

GitHub: [https://github.com/harishgundapuUD](https://github.com/harishgundapuUD)

---

## 📄 License

This project is licensed under the MIT License.
