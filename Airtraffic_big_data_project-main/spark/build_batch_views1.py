from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# ======================================================
# CONFIG
# ======================================================

SILVER_DB = "silver"
SERVING_DB = "serving"

FLIGHTS_TABLE = f"{SILVER_DB}.flight_facts"
AIRPORTS_TABLE = f"{SILVER_DB}.airport_dim"
WEATHER_TABLE = f"{SILVER_DB}.weather_airport_events"
AIRCRAFT_TABLE = f"{SILVER_DB}.aircraft_dim"

# HDFS PATHS - Direct reading is more robust than spark.table() for mixed types
# Update these to your actual HDFS paths
WEATHER_PATH = "hdfs://node1/silver_data/weather_airport_events"
AIRCRAFT_PATH = "hdfs://node1/silver_data/aircraft_dim"

# PHYSICAL SCHEMAS - We read problematic columns as Integers first
WEATHER_PHYSICAL_SCHEMA = StructType([
    StructField("AIRPORT_ID", IntegerType(), True),  # Read as Int to avoid decode error
    StructField("EVENT_TYPE", StringType(), True),
    StructField("year_partition", IntegerType(), True),
    StructField("month_partition", IntegerType(), True)
])

# Columns
COL_DEP_AIRPORT = "DepartureAirportID"
COL_ARR_AIRPORT = "ArrivalAirportID"
COL_CARRIER = "CarrierCode"
COL_CANCELLED = "IsCancelledFlag"
COL_DEP_DELAY = "DepartureDelay"
COL_ARR_DELAY = "ArrivalDelay"
COL_YEAR = "year_partition"
COL_MONTH = "month_partition"
COL_TAIL = "TailNumber"

AIRPORT_ID_COL = "AirportID"
AIRPORT_REGION_COL = "AirportStateName"

AIRCRAFT_TAIL = "TailNum"
AIRCRAFT_YEAR = "YearManufactured"

# ======================================================
# HELPERS & LOADERS
# ======================================================

def table_exists(spark, full_name: str) -> bool:
    db, tbl = full_name.split(".", 1)
    return spark._jsparkSession.catalog().tableExists(db, tbl)

def save_hive_table(df, full_name: str):
    df.write.mode("overwrite").format("parquet").saveAsTable(full_name)
    print(f"[OK] Saved Hive table: {full_name}")

def normalize_flights_schema(flights):
    """Force stable types for the core facts table."""
    return flights.select(
        F.col(COL_DEP_AIRPORT).cast("string").alias(COL_DEP_AIRPORT),
        F.col(COL_ARR_AIRPORT).cast("string").alias(COL_ARR_AIRPORT),
        F.col(COL_CARRIER).cast("string").alias(COL_CARRIER),
        F.col(COL_CANCELLED).cast("boolean").alias(COL_CANCELLED),
        F.col(COL_DEP_DELAY).cast("int").alias(COL_DEP_DELAY),
        F.col(COL_ARR_DELAY).cast("int").alias(COL_ARR_DELAY),
        F.col(COL_YEAR).cast("int").alias(COL_YEAR),
        F.col(COL_MONTH).cast("int").alias(COL_MONTH),
        F.col(COL_TAIL).cast("string").alias(COL_TAIL),
    )

def load_weather_robustly(spark):
    """The fix: Read as physical Int, then cast to Logical String."""
    try:
        # Check if path exists or table exists
        if not table_exists(spark, WEATHER_TABLE): return None
        
        return (spark.read.schema(WEATHER_PHYSICAL_SCHEMA)
                .parquet(WEATHER_PATH)
                .withColumn("AIRPORT_ID", F.col("AIRPORT_ID").cast("string")))
    except Exception as e:
        print(f"[WARN] Weather load failed, skipping weather views: {e}")
        return None

# ======================================================
# BATCH VIEWS
# ======================================================

def build_delay_airport_month(flights):
    return (flights.groupBy(COL_DEP_AIRPORT, COL_YEAR, COL_MONTH)
            .agg(F.avg(F.col(COL_DEP_DELAY)).alias("avg_dep_delay"),
                 F.avg(F.col(COL_ARR_DELAY)).alias("avg_arr_delay"),
                 F.count(F.lit(1)).alias("flights_cnt")))

def build_top10_airports_month(delay_airport_month):
    w = Window.partitionBy(COL_YEAR, COL_MONTH).orderBy(F.col("avg_dep_delay").desc_nulls_last())
    ranked = delay_airport_month.withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= 10)
    agg_exprs = []
    for i in range(1, 11):
        idx = f"{i:02d}"
        agg_exprs.append(F.max(F.when(F.col("rank") == i, F.col(COL_DEP_AIRPORT))).alias(f"rank{idx}_airport"))
        agg_exprs.append(F.max(F.when(F.col("rank") == i, F.col("avg_dep_delay"))).alias(f"rank{idx}_delay"))
    return ranked.groupBy(COL_YEAR, COL_MONTH).agg(*agg_exprs)

def build_delay_weather_region_month(spark, flights, airports):
    weather = load_weather_robustly(spark)
    if not weather: return None

    airports_pre = airports.select(
        F.col(AIRPORT_ID_COL).cast("string").alias("AirportID_str"),
        F.col(AIRPORT_REGION_COL).alias("Region")
    )

    weather_pre = (weather.select(
        F.col("AIRPORT_ID").alias("AirportID_str"),
        F.col("EVENT_TYPE").alias("EventType"),
        F.col(COL_YEAR).alias("w_year"),
        F.col(COL_MONTH).alias("w_month")
    ).dropna().dropDuplicates())

    joined = flights.join(airports_pre, flights[COL_DEP_AIRPORT] == airports_pre["AirportID_str"], "inner") \
                    .join(weather_pre, (F.col(COL_DEP_AIRPORT) == weather_pre["AirportID_str"]) & 
                                       (F.col(COL_YEAR) == weather_pre["w_year"]) & 
                                       (F.col(COL_MONTH) == weather_pre["w_month"]), "inner")

    return (joined.groupBy("Region", "EventType", COL_YEAR, COL_MONTH)
            .agg(F.avg(F.col(COL_DEP_DELAY)).alias("avg_dep_delay"),
                 F.count(F.lit(1)).alias("flights_cnt")))

# ======================================================
# MAIN
# ======================================================

def main():
    spark = (SparkSession.builder
             .appName("BuildServingBatchViews")
             .enableHiveSupport()
             .config("spark.sql.parquet.enableVectorizedReader", "false")
             .config("spark.sql.parquet.mergeSchema", "true")
             .getOrCreate())

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {SERVING_DB}")

    if not table_exists(spark, FLIGHTS_TABLE):
        raise RuntimeError(f"Missing table: {FLIGHTS_TABLE}")

    # Directly call the function defined above
    flights = normalize_flights_schema(spark.table(FLIGHTS_TABLE))
    airports = spark.table(AIRPORTS_TABLE)

    batch_views = []

    # 1. Delay per Airport Month
    v1 = build_delay_airport_month(flights)
    save_hive_table(v1, f"{SERVING_DB}.delay_airport_month")
    batch_views.append(("delay_airport_month", v1))

    # 2. Top 10 Airports
    v3 = build_top10_airports_month(v1)
    save_hive_table(v3, f"{SERVING_DB}.top10_airports_delay_month")
    batch_views.append(("top10_airports_delay_month", v3))

    # 3. Weather Delay View (The one causing the error)
    v4 = build_delay_weather_region_month(spark, flights, airports)
    if v4:
        save_hive_table(v4, f"{SERVING_DB}.delay_weather_region_month")
        batch_views.append(("delay_weather_region_month", v4))

    # Preview results
    for name, df in batch_views:
        print(f"\n[VIEW] {name}")
        df.show(10, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()