# Databricks notebook source
# MAGIC %pip install great-expectations

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import great_expectations as gx
from pyspark.sql.functions import col, lit

context = gx.get_context()
print(f"✅ GX version: {gx.__version__}")

# COMMAND ----------

# Read silver table as pandas for GX
df_silver_pandas = spark.table("workspace.default.silver_songs").toPandas()

# Delete existing data source if exists
try:
    context.data_sources.delete("silver_songs_source")
    print("🗑️ Deleted existing data source")
except Exception:
    print("ℹ️ No existing data source found")

# Create GX data source
data_source = context.data_sources.add_pandas("silver_songs_source")
data_asset = data_source.add_dataframe_asset(name="silver_songs_asset")
batch_definition = data_asset.add_batch_definition_whole_dataframe("silver_batch")

print(f"✅ Loaded {len(df_silver_pandas)} rows into GX!")

# COMMAND ----------

# Delete existing suite if exists
try:
    context.suites.delete("silver_songs_suite")
    print("🗑️ Deleted existing suite")
except Exception:
    print("ℹ️ No existing suite found")

suite = context.suites.add(gx.ExpectationSuite(name="silver_songs_suite"))

# All ML feature columns must not be null
for col_name in ["danceability", "energy", "valence", "tempo", "acousticness", "instrumentalness"]:
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column=col_name)
    )

# Feature ranges must be valid for ML model
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="danceability", min_value=0, max_value=1
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="energy", min_value=0, max_value=1
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="valence", min_value=0, max_value=1
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="tempo", min_value=0, max_value=300
    )
)

# Unrealistic duration (over 60 mins = 3,600,000 ms)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="duration_ms", min_value=0, max_value=3600000
    )
)

# Popularity must be between 0 and 100
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="popularity", min_value=0, max_value=100
    )
)

print("✅ Silver expectations defined!")

# COMMAND ----------

# Delete existing validation if exists
try:
    context.validation_definitions.delete("silver_validation")
    print("🗑️ Deleted existing validation")
except Exception:
    print("ℹ️ No existing validation found")

validation_definition = context.validation_definitions.add(
    gx.ValidationDefinition(
        name="silver_validation",
        data=batch_definition,
        suite=suite
    )
)

results = validation_definition.run(
    batch_parameters={"dataframe": df_silver_pandas}
)

print(f"✅ Validation complete!")
print(f"Success: {results.success}")

# Print detailed results
for result in results.results:
    status = "✅ PASSED" if result.success else "❌ FAILED"
    print(f"{status} — {result.expectation_config.type}")
    print(f"         kwargs: {result.expectation_config.kwargs}")
    print()

# COMMAND ----------

# ============================================================
# QUARANTINE 2 — Capture bad rows from Silver
# ============================================================
df_silver_spark = spark.table("workspace.default.silver_songs")

# 1. Invalid danceability
df_bad_dance = df_silver_spark.filter(
    (col("danceability") < 0) | (col("danceability") > 1)
).withColumn("quarantine_reason", lit("invalid_danceability"))

# 2. Invalid energy
df_bad_energy = df_silver_spark.filter(
    (col("energy") < 0) | (col("energy") > 1)
).withColumn("quarantine_reason", lit("invalid_energy"))

# 3. Invalid valence
df_bad_valence = df_silver_spark.filter(
    (col("valence") < 0) | (col("valence") > 1)
).withColumn("quarantine_reason", lit("invalid_valence"))

# 4. Invalid tempo
df_bad_tempo = df_silver_spark.filter(
    (col("tempo") < 0) | (col("tempo") > 300)
).withColumn("quarantine_reason", lit("invalid_tempo"))

# 5. Unrealistic duration
df_bad_duration = df_silver_spark.filter(
    col("duration_ms") > 3600000
).withColumn("quarantine_reason", lit("unrealistic_duration"))

# 6. Invalid popularity
df_bad_popularity = df_silver_spark.filter(
    (col("popularity") < 0) | (col("popularity") > 100)
).withColumn("quarantine_reason", lit("invalid_popularity"))

# Combine all bad rows
df_quarantine_silver = df_bad_dance \
    .unionByName(df_bad_energy, allowMissingColumns=True) \
    .unionByName(df_bad_valence, allowMissingColumns=True) \
    .unionByName(df_bad_tempo, allowMissingColumns=True) \
    .unionByName(df_bad_duration, allowMissingColumns=True) \
    .unionByName(df_bad_popularity, allowMissingColumns=True) \
    .dropDuplicates()

print(f"Total Silver quarantined rows: {df_quarantine_silver.count()}")

# COMMAND ----------

df_quarantine_silver.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.quarantine_silver_songs")

print("✅ Silver quarantine table created!")

# COMMAND ----------

display(spark.sql("""
    SELECT quarantine_reason, COUNT(*) as count 
    FROM workspace.default.quarantine_silver_songs
    GROUP BY quarantine_reason
    ORDER BY count DESC
"""))



display(spark.sql("""
    SELECT track_name, artists, duration_ms,
    quarantine_reason
    FROM workspace.default.quarantine_silver_songs
"""))