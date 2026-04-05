# Databricks notebook source
from pyspark.sql.functions import col, count, lit

# COMMAND ----------

# ============================================================
# QUARANTINE LAYER — Capture bad rows from Bronze
# ============================================================

df_bronze = spark.table("workspace.default.bronze_songs")

# 1. Duplicate track_ids
duplicate_ids = df_bronze.groupBy("track_id") \
    .count() \
    .filter("count > 1") \
    .select("track_id")

df_duplicates = df_bronze.join(duplicate_ids, on="track_id", how="inner") \
    .withColumn("quarantine_reason", lit("duplicate_track_id"))

# 2. Null track_names
df_null_names = df_bronze.filter(col("track_name").isNull()) \
    .withColumn("quarantine_reason", lit("null_track_name"))

# 3. Bad energy values (not between 0 and 1)
df_bad_energy = df_bronze.filter(
    col("energy").try_cast("float").isNull() |
    (col("energy").try_cast("float") < 0) |
    (col("energy").try_cast("float") > 1)
).withColumn("quarantine_reason", lit("bad_energy_value"))

# 4. Bad danceability values
df_bad_dance = df_bronze.filter(
    col("danceability").try_cast("float").isNull() |
    (col("danceability").try_cast("float") < 0) |
    (col("danceability").try_cast("float") > 1)
).withColumn("quarantine_reason", lit("bad_danceability_value"))

# 5. Bad tempo values
df_bad_tempo = df_bronze.filter(
    col("tempo").try_cast("float").isNull() |
    (col("tempo").try_cast("float") < 0) |
    (col("tempo").try_cast("float") > 300)
).withColumn("quarantine_reason", lit("bad_tempo_value"))

print("✅ Bad rows identified!")

# COMMAND ----------

# Combine all bad rows into one quarantine dataframe
df_quarantine = df_duplicates \
    .unionByName(df_null_names, allowMissingColumns=True) \
    .unionByName(df_bad_energy, allowMissingColumns=True) \
    .unionByName(df_bad_dance, allowMissingColumns=True) \
    .unionByName(df_bad_tempo, allowMissingColumns=True) \
    .dropDuplicates()

print(f"Total quarantined rows: {df_quarantine.count()}")

# COMMAND ----------

# Write to quarantine Delta table
df_quarantine.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.quarantine_songs")

print("✅ Quarantine table created!")

# COMMAND ----------

# See breakdown of quarantine reasons
display(
    spark.table("workspace.default.quarantine_songs")
    .groupBy("quarantine_reason")
    .count()
    .orderBy("count", ascending=False)
)