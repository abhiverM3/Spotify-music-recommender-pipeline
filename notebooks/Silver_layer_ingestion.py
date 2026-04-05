# Databricks notebook source
# intro to silver layer
from pyspark.sql.functions import col, row_number, expr
from pyspark.sql.window import Window

df_bronze = spark.table("workspace.default.bronze_songs")

# COMMAND ----------

# 1. Cast all columns using expr try_cast
df_casted = df_bronze \
    .drop("_c0") \
    .withColumn("popularity",        expr("try_cast(popularity as float)")) \
    .withColumn("duration_ms",       expr("try_cast(duration_ms as float)")) \
    .withColumn("explicit",          expr("try_cast(explicit as boolean)")) \
    .withColumn("danceability",      expr("try_cast(danceability as float)")) \
    .withColumn("energy",            expr("try_cast(energy as float)")) \
    .withColumn("key",               expr("try_cast(key as integer)")) \
    .withColumn("loudness",          expr("try_cast(loudness as float)")) \
    .withColumn("mode",              expr("try_cast(mode as integer)")) \
    .withColumn("speechiness",       expr("try_cast(speechiness as float)")) \
    .withColumn("acousticness",      expr("try_cast(acousticness as float)")) \
    .withColumn("instrumentalness",  expr("try_cast(instrumentalness as float)")) \
    .withColumn("liveness",          expr("try_cast(liveness as float)")) \
    .withColumn("valence",           expr("try_cast(valence as float)")) \
    .withColumn("tempo",             expr("try_cast(tempo as float)")) \
    .withColumn("time_signature",    expr("try_cast(time_signature as integer)"))

# 2. For duplicates — keep the row with highest popularity per track_id
window = Window.partitionBy("track_id").orderBy(col("popularity").desc())

df_deduped = df_casted \
    .withColumn("row_num", row_number().over(window)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

# 3. Drop remaining nulls
df_silver = df_deduped.dropna()

print(f"Bronze rows: {df_bronze.count()}")
print(f"Silver rows after dedup & clean: {df_silver.count()}")
print("✅ Cleaning done!")

# COMMAND ----------

df_silver.printSchema()
print("Total rows: " + str(df_silver.count()))

# COMMAND ----------

# writing to silver delta table

df_silver.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.default.silver_songs")

print("✅ Silver table created!")