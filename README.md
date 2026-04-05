# Spotify-music-recommender-pipeline
End to end ML pipeline using Databricks, Delta Lake, Airflow, GCS
# Spotify Music Recommender Pipeline

A personal project I built to learn data engineering concepts by doing something actually interesting. The idea is simple: take raw Spotify song data, run it through a full medallion architecture pipeline, cluster songs by their audio features, and spit out recommendations based on mood.

The stack is Databricks (free edition), Apache Airflow running locally on WSL2, Google Cloud Storage for raw file storage, and Python for everything else. Total cost: $0.

---

## Why I Built This

I wanted to get hands on with the tools that actually show up in data engineering job postings. Reading about Delta Lake and Airflow is one thing. Building a pipeline that uses them end to end is another. So I picked a dataset I cared about and built the whole thing from scratch.

---

## Architecture

```
GCS Bucket (raw CSV)
    |
    v
Bronze Layer (raw ingestion)
    |
    v
Data Quality (Great Expectations)
    |
    v
Quarantine Layer (bad rows isolated)
    |
    v
Silver Layer (cleaned, typed, deduplicated)
    |
    v
Data Quality Silver (business rule validation)
    |
    v
Gold Layer (ML features + KMeans clustering)
    |
    v
Incremental Merge (Delta MERGE pattern)
    |
    v
Skew Detection (training vs serving comparison)
    |
    v
Monitoring (Evidently AI drift detection)
```

Everything runs on a schedule via Airflow. Each step is a separate Databricks notebook triggered by the DAG.

---

## Stack

| Tool | Purpose |
|---|---|
| Databricks Free Edition | Notebooks, Delta Lake, compute |
| Apache Airflow 3.x | Pipeline orchestration |
| Google Cloud Storage | Raw file storage |
| Delta Lake | ACID table format across all layers |
| Great Expectations 1.x | Data quality validation |
| Evidently AI 0.7.x | Data drift and model monitoring |
| scikit-learn | KMeans clustering for mood detection |
| MLflow | Experiment tracking |
| WSL2 (Ubuntu 24.04) | Local Airflow environment |

---

## The Data

Spotify tracks dataset from Kaggle. About 114,000 rows with audio features like energy, danceability, tempo, valence, acousticness, and so on. Each row is a song. No personal data, no API calls, just a CSV.

After running through the pipeline:
- 41,072 rows quarantined at Bronze (duplicates, bad values, nulls)
- 15 rows quarantined at Silver (DJ mixes and ambient sounds over 60 minutes)
- 89,605 clean songs make it to Gold

---

## Medallion Layers

### Bronze
Raw ingestion from GCS. No transformations. Data lands exactly as it came from the source. Schema is all strings since we trust nothing at this stage.

```python
pandas_df = pd.read_csv(gcs_url)
df_raw = spark.createDataFrame(pandas_df)
df_raw.write.format("delta").saveAsTable("bronze_songs")
```

### Data Quality (Bronze)
Great Expectations runs six validation rules against the raw data before anything moves forward. Rules include checking that track_id is never null, energy is between 0 and 1, tempo is between 0 and 300 BPM, and track_id is unique. Results are logged and bad rows are routed to the quarantine table.

### Quarantine Layer
Bad rows do not get dropped. They get isolated in a separate Delta table with a reason column explaining why they failed. This makes debugging and upstream investigation much easier. At Bronze we quarantine 41k rows, mostly duplicates where the same song appeared across multiple genre categories.

### Silver
Cast all columns to their correct types using `try_cast` instead of `cast` to avoid crashing on malformed values. Deduplicate using a window function that keeps the most popular version of each duplicate track. Drop remaining nulls.

```python
window = Window.partitionBy("track_id").orderBy(col("popularity").desc())
df_deduped = df_casted \
    .withColumn("row_num", row_number().over(window)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")
```

### Data Quality (Silver)
Second round of Great Expectations with tighter business rules. This catches anything that slipped through Silver, like songs with unrealistic durations. In our dataset this caught 15 DJ mixes and ambient sound recordings over 60 minutes long.

### Gold
KMeans clustering on the scaled audio features. Five clusters, each mapped to a mood label. Songs get a cluster assignment and a mood label, then the whole thing gets written as a Delta table ready for querying.

```
Cluster 0: Intense
Cluster 1: Energetic
Cluster 2: Sad/Acoustic
Cluster 3: Chill/Acoustic
Cluster 4: Happy/Dance
```

Silhouette score sits around 0.15 to 0.27 depending on sample. Not bad for unsupervised clustering on music data.

### Incremental Merge
Instead of overwriting 114k rows on every pipeline run, the incremental merge uses Delta's MERGE statement to only process what changed. New songs get inserted, updated songs get updated, unchanged songs get skipped. Delta keeps full history of every operation.

```sql
MERGE INTO bronze_songs AS target
USING new_batch AS source
ON target.track_id = source.track_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
```

### Skew Detection
Compares the mean of each feature in the training data (Silver) against the serving data (new Bronze batch). If any feature shifts by more than 10%, it gets flagged. Instrumentalness came up as skewed in testing, which makes sense given how different the raw and cleaned distributions are.

### Monitoring
Evidently AI generates a full drift report comparing reference data against the current batch using Wasserstein distance. Results get written to a Delta table for historical tracking. An HTML report also gets saved to the GCS volume.

---

## Recommender

Given a song name, find its cluster in the Gold table and return the most popular songs in the same cluster.

```python
recommend_songs("Bohemian Rhapsody")

# Output:
# Song: Bohemian Rhapsody
# Mood: Intense
# Cluster: 0
# Top 5: Blinding Lights, Sweater Weather, Jimmy Cooks...
```

---

## Project Structure

```
Spotify-music-recommender-pipeline/
|-- dags/
|   `-- music_recommender_dag.py
|-- notebooks/
|   |-- Bronze_layer_ingestion.py
|   |-- Data_quality.py
|   |-- Quarantine_layer.py
|   |-- Silver_layer_ingestion.py
|   |-- Data_quality_silver.py
|   |-- Gold_layer_ingestion.py
|   |-- Incremental_merge.py
|   |-- Skew_detection.py
|   `-- Monitoring.py
|-- tests/
|-- .github/workflows/
|-- requirements.txt
`-- README.md
```

---

## Running Locally

### Prerequisites
- WSL2 with Ubuntu 24.04
- Python 3.12
- Databricks Free Edition account
- Google Cloud Storage bucket (public read)
