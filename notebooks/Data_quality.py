# Databricks notebook source
# Data quality checks

%pip install great-expectations

# COMMAND ----------

# MAGIC %pip show great-expectations

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import great_expectations as gx

# create a GX data Context
context = gx.get_context()
#
print("GX version: " + str(gx.__version__))
print("Context created!")

# COMMAND ----------

# read bronze table as panadas dataframe
import pandas as pd

df_bronze = spark.table("workspace.default.bronze_songs").toPandas()

try:
    context.data_sources.delete("bronze_songs_source")
    print("Deleted existing data source")
except Exception:
    print("No existing data source found")
# Tells GX where our data is coming from — in this case a pandas DataFrame.
data_source = context.data_sources.add_pandas("bronze_songs_source")
#Registers the specific dataset we want to validate within that data source.
data_asset = data_source.add_dataframe_asset(name="bronze_songs_asset")
#Tells GX to validate the entire DataFrame at once as one batch.
batch_definition = data_asset.add_batch_definition_whole_dataframe("bronze_batch")

print("Loaded! -> rows in GX: " + str(len(df_bronze)))


# COMMAND ----------

# Delete existing suite if it exists
try:
    context.suites.delete("bronze_songs_suite")
    print("🗑️ Deleted existing suite")
except Exception:
    print("ℹ️ No existing suite found")
#  Creating an expectation suite -> like a container that holds all valdiation rules together
suite = context.suites.add(gx.ExpectationSuite(name="bronze_songs_suite"))
# track id must never be null, every song must have an ID
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="track_id"))
# energy must be between 0 and 1. outside is bad data
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="energy", min_value=0, max_value=1))
# tempo must be between 0 and 300 BPM. Outside of 300 does not make sense
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="tempo", min_value=0, max_value=300))
# danceability must be between 0 and 1, outside of 1 does not exist
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="danceability", min_value=0, max_value=1))
# Track_name must never be null
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="track_name"))
# track_id must be unique
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="track_id"))

print(" Expectations defined!")




# COMMAND ----------

# Delete existing valdiation definitions if it exists

try:
    context.validation_definitions.delete("bronze_songs_validation")
    print("🗑️ Deleted existing validation")
except Exception:
    print("ℹ️ No existing validation found")

# run validation
validation_definition = context.validation_definitions.add(gx.ValidationDefinition(
    name="bronze_songs_validation",
    data=batch_definition,
    suite=suite
    )
)

results = validation_definition.run(
    batch_parameters={"dataframe": df_bronze}
)

print(" Validation complete!")
print("Success: " + str(results.success))

# COMMAND ----------

# See which expectations failed

for result in results.results:
    status = "✅ PASSED" if result.success else "❌ FAILED"
    print(f"{status} — {result.expectation_config.type}")
    print(f"         kwargs: {result.expectation_config.kwargs}")
    print()



# COMMAND ----------

# pass the quarantine
print("Duplicate track_ids:")
df_bronze_spark = spark.table("workspace.default.bronze_songs")
df_bronze_spark.groupBy("track_id") \
    .count() \
    .filter("count > 1") \
    .orderBy("count", ascending=False) \
    .show(10)


# COMMAND ----------

# Check null track_names
print("Null track_names:")
from pyspark.sql.functions import col
df_bronze_spark.filter(col("track_name").isNull()).count()