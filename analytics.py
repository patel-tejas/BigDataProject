#!/usr/bin/env python3
"""
Analytics script: computes 1-minute OHLC, volume, and VWAP.
Reads processed Parquet, writes results as Parquet and a single CSV.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, sum, min, max, min_by, max_by

# ------------------------------------------------------------
# 1. Initialize Spark
# ------------------------------------------------------------
spark = SparkSession.builder \
    .appName("Binance Analytics") \
    .config("spark.sql.shuffle.partitions", "32") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------
# 2. Load processed data
# ------------------------------------------------------------
input_path = "/app/data/processed"
print(f"📂 Loading processed data from {input_path}")
df = spark.read.parquet(input_path)

# Ensure time is timestamp (it already is, but safe)
df = df.withColumn("time", col("time").cast("timestamp"))

# ------------------------------------------------------------
# 3. Compute 1-minute OHLC + VWAP
# ------------------------------------------------------------
print("📊 Computing 1-minute aggregates...")
ohlc = df.groupBy(window(col("time"), "1 minute")).agg(
    min_by("price", "time").alias("open"),      # first trade price in window
    max("price").alias("high"),
    min("price").alias("low"),
    max_by("price", "time").alias("close"),     # last trade price in window
    sum("qty").alias("volume"),
    (sum("quoteQty") / sum("qty")).alias("vwap")
)

# Flatten the window struct
ohlc = ohlc.select(
    col("window.start").alias("start_time"),
    col("window.end").alias("end_time"),
    "open", "high", "low", "close", "volume", "vwap"
)

# Show sample output
print("📈 Sample OHLC data (first 10 rows):")
ohlc.show(10, truncate=False)

# ------------------------------------------------------------
# 4. Save results
# ------------------------------------------------------------
# Parquet (overwrite)
parquet_out = "/app/data/analytics_parquet"
print(f"💾 Saving OHLC to {parquet_out}")
ohlc.write.mode("overwrite").parquet(parquet_out)

# Single CSV for easy viewing
csv_out = "/app/data/analytics_csv"
os.makedirs(csv_out, exist_ok=True)
print(f"💾 Saving OHLC as CSV to {csv_out}")
ohlc.coalesce(1).write.mode("overwrite").option("header", True).csv(csv_out)

print("✅ Analytics completed successfully.")
spark.stop()