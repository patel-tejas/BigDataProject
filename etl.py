#!/usr/bin/env python3
"""
ETL script for Binance trade data.
Reads raw CSV, casts types, adds date column, and saves as Parquet.
FIXED: time_ms is in MICROSECONDS -> divide by 1,000,000 to get seconds.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

# ------------------------------------------------------------
# 1. Initialize Spark
# ------------------------------------------------------------
spark = SparkSession.builder \
    .appName("Binance ETL") \
    .config("spark.sql.shuffle.partitions", "32") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------
# 2. Read CSV
# ------------------------------------------------------------
csv_path = "/app/BTCUSDT-trades-2026-04-19/BTCUSDT-trades-2026-04-19.csv"
print(f"📥 Reading CSV from {csv_path}")

df_raw = spark.read.csv(csv_path, header=False)

# Rename columns according to Binance trade stream schema
df_raw = df_raw.toDF(
    "tradeId", "price", "qty", "quoteQty", "time_ms", "isBuyerMaker", "isBestMatch"
)

# ------------------------------------------------------------
# 3. Type casting & timestamp creation (MICROSECONDS fix)
# ------------------------------------------------------------
df = df_raw.select(
    col("tradeId"),
    col("price").cast("double"),
    col("qty").cast("double"),
    col("quoteQty").cast("double"),
    (col("time_ms").cast("long") / 1000000).cast("timestamp").alias("time"),  # ← FIXED
    col("isBuyerMaker").cast("boolean"),
    col("isBestMatch").cast("boolean")
)

# Add date column for potential partitioning
df = df.withColumn("date", to_date(col("time")))

# Drop rows with missing essential fields
df = df.dropna(subset=["tradeId", "price", "qty", "quoteQty", "time"])

# Repartition for efficient writing
df = df.repartition(16)

# ------------------------------------------------------------
# 4. Write to Parquet
# ------------------------------------------------------------
output_path = "/app/data/processed"
print(f"💾 Writing cleaned data to {output_path}")
df.write.mode("overwrite").parquet(output_path)

print("✅ ETL completed successfully.")
spark.stop()