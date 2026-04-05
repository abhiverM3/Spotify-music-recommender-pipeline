# Databricks notebook source
# skew_detection
import pandas as pd
import numpy as np
from pyspark.sql.functions import col

print("✅ Libraries imported!")

# COMMAND ----------

# ============================================================
# SKEW DETECTION — Compare training vs serving features
# ============================================================

# Feature columns used in ML model
feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "popularity"
]

# Training data = what model was trained on (Silver table)
df_training = spark.table("workspace.default.silver_songs") \
    .select(feature_cols) \
    .toPandas()

# Serving data = new incoming batch (Bronze table - simulates new data)
df_serving = spark.table("workspace.default.bronze_songs") \
    .select([c for c in feature_cols if c in spark.table("workspace.default.bronze_songs").columns]) \
    .toPandas()

# Cast to numeric for comparison
for col_name in feature_cols:
    df_training[col_name] = pd.to_numeric(df_training[col_name], errors="coerce")
    df_serving[col_name] = pd.to_numeric(df_serving[col_name], errors="coerce")

print(f"✅ Training data: {len(df_training)} rows")
print(f"✅ Serving data: {len(df_serving)} rows")

# COMMAND ----------

# ============================================================
# Calculate mean difference between training and serving
# ============================================================

# Threshold — if mean shifts by more than 10% flag it
SKEW_THRESHOLD = 0.10

training_means = df_training[feature_cols].mean()
serving_means = df_serving[feature_cols].fillna(0).mean()

# Calculate percentage difference
skew_results = []

for col_name in feature_cols:
    train_val = training_means[col_name]
    serve_val = serving_means[col_name]
    
    # Avoid division by zero
    if train_val != 0:
        skew_pct = abs((serve_val - train_val) / train_val)
    else:
        skew_pct = 0
    
    status = "🚨 SKEWED" if skew_pct > SKEW_THRESHOLD else "✅ OK"
    
    skew_results.append({
        "feature": col_name,
        "training_mean": round(train_val, 4),
        "serving_mean": round(serve_val, 4),
        "skew_pct": round(skew_pct * 100, 2),
        "status": status
    })

df_skew = pd.DataFrame(skew_results)
display(spark.createDataFrame(df_skew))

# COMMAND ----------

# ============================================================
# Pipeline decision — should we retrain the model?
# ============================================================

skewed_features = df_skew[df_skew["status"] == "🚨 SKEWED"]

if len(skewed_features) > 0:
    print(f"⚠️ WARNING: {len(skewed_features)} features have significant skew!")
    print(f"Skewed features: {list(skewed_features['feature'])}")
    print("💡 Recommendation: Consider retraining the model!")
else:
    print("✅ No significant skew detected!")
    print("💡 Model is still reliable for serving recommendations")

print(f"\nTotal features checked: {len(feature_cols)}")
print(f"Skewed features: {len(skewed_features)}")
print(f"Clean features: {len(feature_cols) - len(skewed_features)}")

# COMMAND ----------

# Save skew report for historical tracking
from pyspark.sql.functions import lit
from datetime import datetime

df_skew["check_date"] = str(datetime.now())
df_skew["threshold_pct"] = SKEW_THRESHOLD * 100

spark.createDataFrame(df_skew) \
    .write.format("delta") \
    .mode("append") \
    .saveAsTable("workspace.default.skew_reports")

print("✅ Skew report saved to Delta table!")