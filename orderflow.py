#!/usr/bin/env python3
"""
Order flow analysis: 
  1) Buy/sell volume (USDT) per minute + imbalance (last 200 minutes)
  2) Total BTC volume traded per hour (bar chart)
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, window, sum, desc

# ------------------------------------------------------------
# 1. Initialize Spark
# ------------------------------------------------------------
spark = SparkSession.builder \
    .appName("Order Flow Analysis") \
    .config("spark.sql.shuffle.partitions", "32") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------
# 2. Load processed data
# ------------------------------------------------------------
input_path = "/app/data/processed"
print(f"📂 Loading processed data from {input_path}")
df = spark.read.parquet(input_path)
df = df.withColumn("time", col("time").cast("timestamp"))
df = df.filter(col("time").isNotNull())

if df.count() == 0:
    print("❌ No data found after filtering. Exiting.")
    spark.stop()
    exit(1)

# ------------------------------------------------------------
# 3. Define buy/sell volumes (USDT) per minute (for order flow)
# ------------------------------------------------------------
# isBuyerMaker == false  → buyer initiated (buy volume)
# isBuyerMaker == true   → seller initiated (sell volume)
df_flow = df.withColumn(
    "buy_volume",
    when(col("isBuyerMaker") == False, col("quoteQty")).otherwise(0)
).withColumn(
    "sell_volume",
    when(col("isBuyerMaker") == True, col("quoteQty")).otherwise(0)
)

# Aggregate per minute
flow_agg = df_flow.groupBy(window(col("time"), "1 minute")).agg(
    sum("buy_volume").alias("buy_volume"),
    sum("sell_volume").alias("sell_volume")
)
flow_agg = flow_agg.withColumn("imbalance", col("buy_volume") - col("sell_volume"))

flow_df = flow_agg.select(
    col("window.start").alias("start_time"),
    "buy_volume",
    "sell_volume",
    "imbalance"
).orderBy("start_time")

# ------------------------------------------------------------
# 4. Get last 200 minutes for order flow plot
# ------------------------------------------------------------
total_minutes = flow_df.count()
if total_minutes == 0:
    print("⚠️ No minute buckets created. Skipping order flow plot.")
else:
    take_minutes = min(200, total_minutes)
    print(f"📊 Total minute buckets: {total_minutes}. Plotting last {take_minutes} minutes.")
    last_minutes = flow_df.orderBy(desc("start_time")).limit(take_minutes).orderBy("start_time")
    pdf_flow = last_minutes.toPandas()

    if not pdf_flow.empty:
        # Generate order flow plot
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        plt.figure(figsize=(14, 7))
        plt.plot(pdf_flow["start_time"], pdf_flow["buy_volume"], label="Buy Volume (USDT)", color="green", linewidth=1.5)
        plt.plot(pdf_flow["start_time"], pdf_flow["sell_volume"], label="Sell Volume (USDT)", color="red", linewidth=1.5)
        plt.plot(pdf_flow["start_time"], pdf_flow["imbalance"], label="Imbalance (Buy - Sell)", color="blue", linestyle="--", linewidth=1)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Volume (USDT)", fontsize=12)
        plt.title("Order Flow Analysis (1-min buckets) - Most Recent Data", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_dir = "/app/data/orderflow"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = f"{plot_dir}/orderflow_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Order flow plot saved to {plot_path}")
    else:
        print("⚠️ No data for order flow plot.")

# ------------------------------------------------------------
# 5. Hourly BTC volume (total quantity traded per hour)
# ------------------------------------------------------------
print("\n📊 Computing hourly BTC volume...")
df_hourly = df.groupBy(window(col("time"), "1 hour")).agg(
    sum("qty").alias("btc_volume")
)
df_hourly = df_hourly.select(
    col("window.start").alias("hour_start"),
    "btc_volume"
).orderBy("hour_start")

total_hours = df_hourly.count()
if total_hours == 0:
    print("⚠️ No hourly buckets created. Skipping BTC volume chart.")
else:
    pdf_hourly = df_hourly.toPandas()
    print(f"📊 Total hour buckets: {total_hours}. Plotting all hours.")

    # Generate bar chart
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.figure(figsize=(14, 7))
    plt.bar(pdf_hourly["hour_start"], pdf_hourly["btc_volume"], width=0.02, color="orange", alpha=0.7, label="BTC Volume")
    plt.xlabel("Hour (UTC)", fontsize=12)
    plt.ylabel("BTC Quantity Traded", fontsize=12)
    plt.title("Total BTC Traded per Hour", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plot_dir = "/app/data/orderflow"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path_btc = f"{plot_dir}/hourly_btc_volume.png"
    plt.savefig(plot_path_btc, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Hourly BTC volume chart saved to {plot_path_btc}")

# ------------------------------------------------------------
# 6. Done
# ------------------------------------------------------------
print("\n🎉 Order flow analysis completed.")
spark.stop()