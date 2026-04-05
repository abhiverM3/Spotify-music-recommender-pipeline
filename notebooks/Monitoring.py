# Databricks notebook source
#monitoring
%pip install evidently

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
import evidently
print(evidently.__version__)
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently import Report

print("✅ Evidently 0.7.x imported!")

# COMMAND ----------

# ============================================================
# MONITORING — Evidently AI Drift Detection
# ============================================================

feature_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "popularity"
]

# Reference data = Silver (what model was trained on)
reference_data = spark.table("workspace.default.silver_songs") \
    .select(feature_cols) \
    .limit(5000) \
    .toPandas()

# Current data = Bronze (new incoming batch)
current_data = spark.table("workspace.default.bronze_songs") \
    .select(feature_cols) \
    .limit(5000) \
    .toPandas()

# Cast to numeric
for col_name in feature_cols:
    reference_data[col_name] = pd.to_numeric(reference_data[col_name], errors="coerce")
    current_data[col_name] = pd.to_numeric(current_data[col_name], errors="coerce")

# Drop nulls
reference_data = reference_data.dropna()
current_data = current_data.dropna()

print(f"✅ Reference data: {len(reference_data)} rows")
print(f"✅ Current data: {len(current_data)} rows")

# COMMAND ----------

# Create datasets
reference_dataset = Dataset.from_pandas(
    reference_data,
    data_definition=DataDefinition()
)

current_dataset = Dataset.from_pandas(
    current_data,
    data_definition=DataDefinition()
)

# Create and run report
report = Report([
    DataDriftPreset(),
    DataSummaryPreset(),
])

result = report.run(reference_dataset, current_dataset)

print("✅ Evidently report generated!")

# COMMAND ----------

# Save HTML report to Volume (serverless compatible path)
report_path = "/Volumes/workspace/default/spotify_recommender_raw/drift_report.html"
result.save_html(report_path)

print(f"✅ Report saved!")
print("💡 Download from Databricks Volume to view in browser!")

# COMMAND ----------

from datetime import datetime

report_dict = result.dict()
metrics = report_dict.get("metrics", [])

drift_metrics = []

for metric in metrics:
    metric_name = metric.get("metric_name", "")
    
    # Only grab ValueDrift metrics (metrics 1-10)
    if metric_name.startswith("ValueDrift"):
        config = metric.get("config", {})
        value = metric.get("value", 0.0)
        feature = config.get("column", "unknown")
        threshold = config.get("threshold", 0.1)
        drift_score = float(value) if value else 0.0
        drift_detected = drift_score > threshold
        
        drift_metrics.append({
            "check_date": str(datetime.now()),
            "feature": feature,
            "drift_detected": drift_detected,
            "drift_score": round(drift_score, 4),
            "threshold": threshold,
            "stat_test": "Wasserstein distance"
        })

if drift_metrics:
    df_drift = spark.createDataFrame(pd.DataFrame(drift_metrics))
    df_drift.write.format("delta") \
        .mode("append") \
        .saveAsTable("workspace.default.drift_reports")
    print(f"✅ Drift metrics saved! {len(drift_metrics)} features tracked")
    display(df_drift)
else:
    print("⚠️ No drift metrics extracted")

# COMMAND ----------

if drift_metrics:
    df_metrics = pd.DataFrame(drift_metrics)
    drifted = df_metrics[df_metrics["drift_detected"] == True]

    print("=" * 50)
    print("📊 MONITORING SUMMARY")
    print("=" * 50)
    print(f"Total features monitored: {len(drift_metrics)}")
    print(f"Features with drift:      {len(drifted)}")
    print(f"Clean features:           {len(drift_metrics) - len(drifted)}")

    if len(drifted) > 0:
        print(f"\n🚨 DRIFTED FEATURES:")
        for _, row in drifted.iterrows():
            print(f"  - {row['feature']} (score: {row['drift_score']:.4f}, test: {row['stat_test']})")
        print(f"\n💡 Recommendation: Retrain model!")
    else:
        print("\n✅ No drift detected — model is reliable!")
else:
    print("⚠️ No drift metrics to summarize")
    print("💡 Check the HTML report in your Volume for full results!")