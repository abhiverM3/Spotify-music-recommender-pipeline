# Databricks notebook source
import pandas as pd
print("✅ Libraries imported!")

# COMMAND ----------

# ============================================================
# BRONZE LAYER — Ingest from GCS public bucket
# ============================================================

# GCS public URL
gcs_url = "https://storage.googleapis.com/music-recommender-raw/dataset.csv"

# Read directly from GCS using pandas
print("📥 Reading from GCS...")
pandas_df = pd.read_csv(gcs_url)

print(f"✅ Loaded {len(pandas_df)} rows from GCS!")
print(f"Columns: {list(pandas_df.columns)}")


# COMMAND ----------

# Drop unnamed index column from CSV
pandas_df = pandas_df.drop(columns=["Unnamed: 0"], errors="ignore")

# Convert to Spark DataFrame
df_raw = spark.createDataFrame(pandas_df)

# Write to Bronze Delta table
df_raw.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.bronze_songs")

print(f"✅ Bronze table created from GCS!")
print(f"Total rows: {df_raw.count()}")

df_raw.printSchema()
display(df_raw)

# COMMAND ----------

display(spark.sql("SELECT * FROM workspace.default.bronze_songs LIMIT 10"))

print("columns: " + str(len(spark.table("workspace.default.bronze_songs").columns))) # counts columns

spark.sql("SELECT COUNT(*) as total_rows FROM workspace.default.bronze_songs").show() # Shows var total_rows

# COMMAND ----------

df_bronze = spark.table("workspace.default.bronze_songs")

# see column names and data types
# for col in df.columns:
#     print(col)

df_bronze.printSchema()