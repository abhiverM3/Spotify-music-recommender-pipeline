# Databricks notebook source
# Install MLflow (already in Databricks but let's make sure)
# from pyspark.ml.feature import VectorAssembler, StandardScaler
# from pyspark.ml.clustering import KMeans
# from pyspark.ml.evaluation import ClusteringEvaluator
# import mlflow
# import mlflow.spark

# print("✅ Libraries imported!")

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import mlflow
# import mlflow.sklearn
# import os

# os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/default/spotify_recommender_raw"

# print("✅ Libraries imported!")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

print("✅ Libraries imported!")

# COMMAND ----------

# Read Silver table
# df_silver = spark.table("workspace.default.silver_songs")

# print(f"Total rows: {df_silver.count()}")
# display(df_silver.limit(5))


# Read Silver table
df_silver = spark.table("workspace.default.silver_songs").toPandas()

print(f"✅ Loaded {len(df_silver)} rows from Silver table!")

# COMMAND ----------

# ============================================================
# GOLD LAYER — ML Features + K-Means Clustering
# ============================================================

# 1. Select only the audio features we want for clustering
# feature_cols = [
#     "danceability", "energy", "loudness", "speechiness",
#     "acousticness", "instrumentalness", "liveness",
#     "valence", "tempo", "popularity"
# ]

# # 2. Assemble all feature columns into a single vector column
# # (ML algorithms need one single vector, not separate columns)
# assembler = VectorAssembler(
#     inputCols=feature_cols,
#     outputCol="features_raw"
# )

# df_assembled = assembler.transform(df_silver)

# print("✅ Features assembled!")


# ============================================================
# GOLD LAYER — ML Features + K-Means Clustering (sklearn)
# ============================================================

# 1. Select audio features for clustering
feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "popularity"
]

# 2. Extract features
X = df_silver[feature_cols].fillna(0)

# 3. Scale features so no single column dominates
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Features scaled!")

# COMMAND ----------

# 3. Scale features so no single column dominates
# e.g. loudness is in -60 to 0 range, danceability is 0 to 1
# StandardScaler brings them all to the same scale

# scaler = StandardScaler(
#     inputCol="features_raw",
#     outputCol="features",
#     withStd=True,
#     withMean=True
# )

# scaler_model = scaler.fit(df_assembled)
# df_scaled = scaler_model.transform(df_assembled)

# print("✅ Features scaled!")


# ============================================================
# Train K-Means and track with MLflow
# ============================================================

# NUM_CLUSTERS = 5

# with mlflow.start_run(run_name="kmeans_music_recommender_sklearn"):

#     # 1. Train K-Means
#     kmeans = KMeans(
#         n_clusters=NUM_CLUSTERS,
#         random_state=42,
#         n_init=10
#     )
#     kmeans.fit(X_scaled)

#     # 2. Assign clusters
#     df_silver["cluster"] = kmeans.labels_

#     # 3. Calculate silhouette score
#     from sklearn.metrics import silhouette_score
#     silhouette = silhouette_score(X_scaled, kmeans.labels_, sample_size=10000)

#     # 4. Log to MLflow
#     mlflow.log_param("num_clusters", NUM_CLUSTERS)
#     mlflow.log_param("feature_cols", feature_cols)
#     mlflow.log_metric("silhouette_score", silhouette)
#     mlflow.sklearn.log_model(kmeans, "kmeans_model")

#     print(f"✅ Model trained!")
#     print(f"Number of clusters: {NUM_CLUSTERS}")
#     print(f"Silhouette score: {silhouette:.4f}")


NUM_CLUSTERS = 5

# 1. Train K-Means
kmeans = KMeans(
    n_clusters=NUM_CLUSTERS,
    random_state=42,
    n_init=10
)
kmeans.fit(X_scaled)

# 2. Assign clusters
df_silver["cluster"] = kmeans.labels_

# 3. Calculate silhouette score
silhouette = silhouette_score(X_scaled, kmeans.labels_, sample_size=10000)

print(f"✅ Model trained!")
print(f"Number of clusters: {NUM_CLUSTERS}")
print(f"Silhouette score: {silhouette:.4f}")

# 4. Save metrics to a Delta table instead of MLflow
metrics_df = spark.createDataFrame([{
    "run_date": str(pd.Timestamp.now()),
    "num_clusters": NUM_CLUSTERS,
    "silhouette_score": float(silhouette),
    "total_songs": len(df_silver)
}])

metrics_df.write.format("delta") \
    .mode("append") \
    .saveAsTable("workspace.default.ml_metrics")

print("✅ Metrics saved to Delta table!")

# COMMAND ----------

# Tell MLflow to use our Volume for storing models
# mlflow.set_tracking_uri("databricks")

# import os
# os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/default/spotify_recommender_raw"

# print("✅ MLflow storage path set!")


# Map cluster numbers to mood labels
mood_map = {0: "Intense", 1: "Energetic", 2: "Sad/Acoustic", 3: "Chill/Acoustic", 4: "Happy/Dance"}
df_silver["mood"] = df_silver["cluster"].map(mood_map)

print("✅ Mood labels assigned!")
print(df_silver["mood"].value_counts())

# COMMAND ----------

# ============================================================
# Train K-Means model and track with MLflow
# ============================================================

# Number of clusters (mood buckets)
# 5 = chill, energetic, sad, happy, intense
# NUM_CLUSTERS = 5

# # Start MLflow run to track our experiment
# with mlflow.start_run(run_name="kmeans_music_recommender"):
    
#     # 1. Train K-Means model
#     kmeans = KMeans(
#         featuresCol="features",
#         predictionCol="cluster",
#         k=NUM_CLUSTERS,
#         seed=42
#     )
    
#     model = kmeans.fit(df_scaled)
    
#     # 2. Assign clusters to all songs
#     df_clustered = model.transform(df_scaled)
    
#     # 3. Evaluate model using Silhouette score
#     # (measures how well songs fit in their cluster, -1 to 1, higher is better)
#     evaluator = ClusteringEvaluator(
#         featuresCol="features",
#         predictionCol="cluster"
#     )
#     silhouette = evaluator.evaluate(df_clustered)
    
#     # 4. Log metrics and params to MLflow
#     mlflow.log_param("num_clusters", NUM_CLUSTERS)
#     mlflow.log_param("feature_cols", feature_cols)
#     mlflow.log_metric("silhouette_score", silhouette)
    
#     # 5. Log the model itself
#     mlflow.spark.log_model(model, "kmeans_model")
    
#     print(f"✅ Model trained!")
#     print(f"Number of clusters: {NUM_CLUSTERS}")
#     print(f"Silhouette score: {silhouette:.4f}")




# Convert back to Spark and write Gold table
df_gold = spark.createDataFrame(df_silver[[
    "track_id", "artists", "album_name", "track_name",
    "track_genre", "popularity", "danceability", "energy",
    "valence", "tempo", "acousticness", "instrumentalness",
    "cluster", "mood"
]])

df_gold.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.gold_songs")

print("✅ Gold table created!")

# COMMAND ----------

# display(
#     df_clustered.groupBy("cluster")
#     .count()
#     .orderBy("cluster")
# )

# COMMAND ----------

# See average audio features per cluster
# This tells us what mood each cluster represents
# display(
#     df_clustered.groupBy("cluster").avg(
#         "danceability", "energy", "valence",
#         "tempo", "acousticness", "instrumentalness"
#     ).orderBy("cluster")
# )

# COMMAND ----------

# ============================================================
# GOLD LAYER — Add mood labels and write Gold table
# ============================================================
# from pyspark.sql.functions import when
# from pyspark.sql.functions import col, when
# from pyspark.ml.feature import VectorAssembler, StandardScaler
# from pyspark.ml.clustering import KMeans
# from pyspark.ml.evaluation import ClusteringEvaluator
# import mlflow
# import mlflow.spark

# print("✅ Libraries imported!")

# # Map cluster numbers to mood labels
# df_gold = df_clustered.select(
#     "track_id", "artists", "album_name", "track_name",
#     "track_genre", "popularity", "danceability", "energy",
#     "valence", "tempo", "acousticness", "instrumentalness",
#     "cluster"
# ).withColumn("mood",
#     when(col("cluster") == 0, "Intense")
#     .when(col("cluster") == 1, "Energetic")
#     .when(col("cluster") == 2, "Sad/Acoustic")
#     .when(col("cluster") == 3, "Chill/Acoustic")
#     .when(col("cluster") == 4, "Happy/Dance")
# )

# display(df_gold.limit(10))
# print("✅ Mood labels assigned!")

# COMMAND ----------

# Write to Gold Delta table
# df_gold.write.format("delta") \
#     .mode("overwrite") \
#     .option("overwriteSchema", "true") \
#     .saveAsTable("workspace.default.gold_songs")

# print("✅ Gold table created!")

# COMMAND ----------


#============================================================
# RECOMMENDER — Input a song, get recommendations back
# ============================================================

# def recommend_songs(song_name, n=5):
#     # 1. Find the song
#     song = spark.table("workspace.default.gold_songs") \
#         .filter(col("track_name") == song_name) \
#         .limit(1)
    
#     if song.count() == 0:
#         print(f"❌ Song '{song_name}' not found!")
#         return
    
#     # 2. Get its cluster and mood
#     cluster_id = song.collect()[0]["cluster"]
#     mood = song.collect()[0]["mood"]
    
#     print(f"🎵 Song: {song_name}")
#     print(f"🎭 Mood: {mood}")
#     print(f"🔢 Cluster: {cluster_id}")
#     print(f"\n🎯 Top {n} recommendations:\n")
    
#     # 3. Find similar songs in the same cluster
#     recommendations = spark.table("workspace.default.gold_songs") \
#         .filter(col("cluster") == cluster_id) \
#         .filter(col("track_name") != song_name) \
#         .orderBy(col("popularity").desc()) \
#         .limit(n) \
#         .select("track_name", "artists", "mood", "popularity")
    
#     display(recommendations)

# # Try it out!
# recommend_songs("Bohemian Rhapsody")

# COMMAND ----------

# Test the recommender
def recommend_songs(song_name, n=5):
    df = spark.table("workspace.default.gold_songs").toPandas()
    
    song = df[df["track_name"] == song_name]
    
    if len(song) == 0:
        print(f"❌ Song '{song_name}' not found!")
        return
    
    cluster_id = song.iloc[0]["cluster"]
    mood = song.iloc[0]["mood"]
    
    print(f"🎵 Song: {song_name}")
    print(f"🎭 Mood: {mood}")
    print(f"🔢 Cluster: {cluster_id}")
    print(f"\n🎯 Top {n} recommendations:\n")
    
    recommendations = df[
        (df["cluster"] == cluster_id) &
        (df["track_name"] != song_name)
    ].sort_values("popularity", ascending=False).head(n)
    
    display(spark.createDataFrame(recommendations[["track_name", "artists", "mood", "popularity"]]))

recommend_songs("Bohemian Rhapsody")