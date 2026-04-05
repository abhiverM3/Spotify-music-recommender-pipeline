# Databricks notebook source
from delta.tables import DeltaTable
from pyspark.sql.functions import col, expr
from datetime import datetime

print("✅ Libraries imported!")

# COMMAND ----------

# ============================================================
# INCREMENTAL MERGE — Simulate new incoming data from GCS
# ============================================================

# In production this would be new files landing in GCS
# For demo we simulate by taking a subset and modifying it

# 1. Read current bronze table
df_existing = spark.table("workspace.default.bronze_songs")

# 2. Simulate new batch:
#    - Take 1000 existing songs (simulates updated records)
#    - Add 5 completely new fake songs (simulates new records)

df_updates = df_existing.limit(1000)  # existing songs with changes

# Create 5 fake new songs
new_songs = spark.createDataFrame([
    ("NEW001", "New Artist 1", "New Album 1", "New Song 1", "75", "200000", "false", "0.8", "0.9", "5", "-5.0", "1", "0.1", "0.1", "0.0", "0.1", "0.8", "120.0", "4", "pop"),
    ("NEW002", "New Artist 2", "New Album 2", "New Song 2", "80", "210000", "false", "0.7", "0.8", "3", "-6.0", "0", "0.2", "0.2", "0.0", "0.2", "0.7", "130.0", "4", "rock"),
    ("NEW003", "New Artist 3", "New Album 3", "New Song 3", "65", "190000", "false", "0.6", "0.7", "7", "-7.0", "1", "0.3", "0.3", "0.0", "0.3", "0.6", "110.0", "4", "jazz"),
    ("NEW004", "New Artist 4", "New Album 4", "New Song 4", "70", "220000", "false", "0.5", "0.6", "2", "-8.0", "0", "0.4", "0.4", "0.0", "0.4", "0.5", "140.0", "4", "classical"),
    ("NEW005", "New Artist 5", "New Album 5", "New Song 5", "85", "230000", "false", "0.9", "0.5", "9", "-4.0", "1", "0.5", "0.5", "0.0", "0.5", "0.4", "150.0", "4", "hip-hop"),
], df_existing.columns)

# Combine updates + new songs
df_new_batch = df_updates.union(new_songs)

print(f"✅ New batch size: {df_new_batch.count()} rows")
print(f"   - {df_updates.count()} updated existing songs")
print(f"   - 5 brand new songs")

# COMMAND ----------

# ============================================================
# DELTA MERGE — Only update what changed
# ============================================================

# Get the existing Delta table
bronze_delta = DeltaTable.forName(spark, "workspace.default.bronze_songs")

# Run MERGE
bronze_delta.alias("target").merge(
    df_new_batch.alias("source"),
    "target.track_id = source.track_id"  # match on track_id
).whenMatchedUpdateAll().whenNotMatchedInsertAll() \
.execute()

print("✅ MERGE complete!")

# COMMAND ----------

# Check new songs were added
print("Checking for new songs:")
display(spark.sql("""
    SELECT track_id, artists, track_name 
    FROM workspace.default.bronze_songs 
    WHERE track_id LIKE 'NEW%'
"""))

# Check total row count
total = spark.table("workspace.default.bronze_songs").count()
print(f"Total rows after merge: {total}")
print(f"Expected: 114,005 (114,000 original + 5 new)")

# COMMAND ----------

# Delta Lake keeps full history of all operations!
display(spark.sql("""
    DESCRIBE HISTORY workspace.default.bronze_songs
    LIMIT 5
"""))