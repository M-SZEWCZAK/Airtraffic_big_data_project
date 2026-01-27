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

# HDFS PATH for Weather (Update this to your actual path if different)
WEATHER_PATH = "hdfs://node1/silver_data/weather_airport_events"

# PHYSICAL SCHEMA: Defines how data actually looks on disk (AIRPORT_ID as Int)
# This prevents the low-level Parquet decoder from crashing.
WEATHER_PHYSICAL_SCHEMA = StructType([
    StructField("AIRPORT_ID", IntegerType(), True),
    StructField("EVENT_TYPE", StringType(), True),
    StructField("year_partition", IntegerType(), True),
    StructField("month_partition", IntegerType(), True)
])

# flight_facts columns
COL_DEP_AIRPORT = "DepartureAirportID"
COL_ARR_AIRPORT = "ArrivalAirportID"
COL_CARRIER = "CarrierCode"
COL_CANCELLED = "IsCancelledFlag"
COL_DEP_DELAY = "DepartureDelay"
COL_ARR_DELAY = "ArrivalDelay"
COL_YEAR = "year_partition"
COL_MONTH = "month_partition"

# airport_dim columns
AIRPORT_ID_COL = "AirportID"
AIRPORT_REGION_COL = "AirportStateName"

COL_TAIL = "TailNumber"         
AIRCRAFT_TAIL = "TailNum"        
AIRCRAFT_YEAR = "YearManufactured"


# ======================================================
# HELPERS
# ======================================================

def table_exists(spark, full_name: str) -> bool:
    """Check if Hive table exists."""
    db, tbl = full_name.split(".", 1)
    return spark._jsparkSession.catalog().tableExists(db, tbl)


def require_columns(df, required, table_name: str):
    """Raise readable error if DataFrame misses required columns."""
    missing = sorted(list(set(required) - set(df.columns)))
    if missing:
        raise RuntimeError(f"Missing columns in {table_name}: {missing}")


def save_hive_table(df, full_name: str):
    """Save DataFrame as Hive table (Parquet)."""
    df.write.mode("overwrite").format("parquet").saveAsTable(full_name)
    print(f"[OK] Saved Hive table: {full_name}")


def normalize_flights_schema(flights):
    """Force stable types to avoid Parquet dictionary decoding issues."""
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


# ======================================================
# BATCH VIEWS
# ======================================================

def build_delay_airport_month(flights):
    return (
        flights
        .groupBy(COL_DEP_AIRPORT, COL_YEAR, COL_MONTH)
        .agg(
            F.avg(F.col(COL_DEP_DELAY)).alias("avg_dep_delay"),
            F.avg(F.col(COL_ARR_DELAY)).alias("avg_arr_delay"),
            F.count(F.lit(1)).alias("flights_cnt")
        )
    )


def build_cancel_airport_month(flights):
    return (
        flights
        .groupBy(COL_DEP_AIRPORT, COL_YEAR, COL_MONTH)
        .agg(
            F.sum(F.when(F.col(COL_CANCELLED) == True, 1).otherwise(0)).alias("cancel_cnt"),
            F.count(F.lit(1)).alias("flights_cnt")
        )
        .withColumn(
            "cancel_pct",
            F.when(F.col("flights_cnt") == 0, F.lit(None))
             .otherwise(F.round(F.col("cancel_cnt") * 100.0 / F.col("flights_cnt"), 3))
        )
    )


def build_delay_carrier_month(flights):
    return (
        flights
        .groupBy(COL_CARRIER, COL_YEAR, COL_MONTH)
        .agg(
            F.avg(F.col(COL_DEP_DELAY)).alias("avg_dep_delay"),
            F.avg(F.col(COL_ARR_DELAY)).alias("avg_arr_delay"),
            F.count(F.lit(1)).alias("flights_cnt")
        )
    )


def build_top10_airports_month(delay_airport_month):
    w = Window.partitionBy(COL_YEAR, COL_MONTH).orderBy(F.col("avg_dep_delay").desc_nulls_last())

    ranked = (
        delay_airport_month
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= 10)
    )

    agg_exprs = []
    for i in range(1, 11):
        idx = f"{i:02d}"
        agg_exprs.append(F.max(F.when(F.col("rank") == i, F.col(COL_DEP_AIRPORT))).alias(f"rank{idx}_airport"))
        agg_exprs.append(F.max(F.when(F.col("rank") == i, F.col("avg_dep_delay"))).alias(f"rank{idx}_delay"))

    return ranked.groupBy(COL_YEAR, COL_MONTH).agg(*agg_exprs)


def build_delay_weather_region_month(spark, flights, airports):
    if not table_exists(spark, WEATHER_TABLE):
        return None

    # THE FIX: Read from HDFS using the Physical Schema (Int) then cast to String
    weather = (
        spark.read.schema(WEATHER_PHYSICAL_SCHEMA)
        .parquet(WEATHER_PATH)
        .withColumn("AIRPORT_ID", F.col("AIRPORT_ID").cast("string"))
    )

    airports_pre = airports.select(
        F.col(AIRPORT_ID_COL).cast("string").alias("AirportID_str"),
        F.col(AIRPORT_REGION_COL).alias("Region")
    )

    flights_region = flights.join(
        airports_pre,
        flights[COL_DEP_AIRPORT] == airports_pre["AirportID_str"],
        "left"
    )

    weather_pre = (
        weather.select(
            F.col("AIRPORT_ID").alias("AirportID_str"),
            F.col("EVENT_TYPE").cast("string").alias("EventType"),
            F.col(COL_YEAR).cast("int").alias("w_year"),
            F.col(COL_MONTH).cast("int").alias("w_month")
        )
        .dropna(subset=["AirportID_str", "EventType", "w_year", "w_month"])
        .dropDuplicates(["AirportID_str", "EventType", "w_year", "w_month"])
    )

    joined = flights_region.join(
        weather_pre,
        (flights_region[COL_DEP_AIRPORT] == weather_pre["AirportID_str"]) &
        (flights_region[COL_YEAR] == weather_pre["w_year"]) &
        (flights_region[COL_MONTH] == weather_pre["w_month"]),
        "inner"
    )

    return (
        joined
        .groupBy("Region", "EventType", COL_YEAR, COL_MONTH)
        .agg(
            F.avg(F.col(COL_DEP_DELAY)).alias("avg_dep_delay"),
            F.avg(F.col(COL_ARR_DELAY)).alias("avg_arr_delay"),
            F.count(F.lit(1)).alias("flights_cnt")
        )
    )


def build_cancel_weather_region_month(spark, flights, airports):
    if not table_exists(spark, WEATHER_TABLE):
        return None

    # THE FIX: Read from HDFS using the Physical Schema (Int) then cast to String
    weather = (
        spark.read.schema(WEATHER_PHYSICAL_SCHEMA)
        .parquet(WEATHER_PATH)
        .withColumn("AIRPORT_ID", F.col("AIRPORT_ID").cast("string"))
    )

    airports_pre = airports.select(
        F.col(AIRPORT_ID_COL).cast("string").alias("AirportID_str"),
        F.col(AIRPORT_REGION_COL).alias("Region")
    )

    flights_region = flights.join(
        airports_pre,
        flights[COL_DEP_AIRPORT] == airports_pre["AirportID_str"],
        "left"
    )

    weather_pre = (
        weather.select(
            F.col("AIRPORT_ID").alias("AirportID_str"),
            F.col("EVENT_TYPE").cast("string").alias("EventType"),
            F.col(COL_YEAR).cast("int").alias("w_year"),
            F.col(COL_MONTH).cast("int").alias("w_month")
        )
        .dropna(subset=["AirportID_str", "EventType", "w_year", "w_month"])
        .dropDuplicates(["AirportID_str", "EventType", "w_year", "w_month"])
    )

    joined = flights_region.join(
        weather_pre,
        (flights_region[COL_DEP_AIRPORT] == weather_pre["AirportID_str"]) &
        (flights_region[COL_YEAR] == weather_pre["w_year"]) &
        (flights_region[COL_MONTH] == weather_pre["w_month"]),
        "inner"
    )

    return (
        joined
        .groupBy("Region", "EventType", COL_YEAR, COL_MONTH)
        .agg(
            F.avg(F.col(COL_CANCELLED).cast("double")).alias("cancel_rate"),
            F.count(F.lit(1)).alias("flights_cnt")
        )
    )


def build_aircraft_age_bucket_carrier_year(spark, flights):
    if not table_exists(spark, AIRCRAFT_TABLE):
        return None

    aircraft = spark.table(AIRCRAFT_TABLE)
    flights_pre = flights.select(
        F.col(COL_CARRIER), F.col(COL_DEP_DELAY), F.col(COL_ARR_DELAY),
        F.col(COL_YEAR).alias("flight_year"), F.col(COL_TAIL)
    ).dropna(subset=[COL_CARRIER, "flight_year", COL_TAIL])

    joined = flights_pre.join(aircraft, flights_pre[COL_TAIL] == aircraft[AIRCRAFT_TAIL], "inner")

    age = (F.col("flight_year") - F.col(AIRCRAFT_YEAR))
    bucket = (
        F.when(age.between(0, 5), F.lit("0-5"))
         .when(age.between(6, 10), F.lit("6-10"))
         .when(age.between(11, 20), F.lit("11-20"))
         .when(age > 20, F.lit(">20"))
         .otherwise(F.lit("unknown"))
    )

    return (
        joined
        .withColumn("aircraft_age_bucket", bucket)
        .groupBy("flight_year", "aircraft_age_bucket", COL_CARRIER)
        .agg(
            F.avg("DepartureDelay").alias("avg_dep_delay"),
            F.avg("ArrivalDelay").alias("avg_arr_delay"),
            F.count(F.lit(1)).alias("flights_cnt")
        )
    )


# ======================================================
# MAIN
# ======================================================

def main():
    spark = (
        SparkSession.builder
        .appName("BuildServingBatchViews")
        .config("spark.sql.parquet.mergeSchema", "true")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .enableHiveSupport()
        .getOrCreate()
    )

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {SERVING_DB}")

    if not table_exists(spark, FLIGHTS_TABLE) or not table_exists(spark, AIRPORTS_TABLE):
        raise RuntimeError("Required silver tables missing.")

    flights = normalize_flights_schema(spark.table(FLIGHTS_TABLE))
    airports = spark.table(AIRPORTS_TABLE)

    # 1) Delay per airport-month
    v1 = build_delay_airport_month(flights)
    save_hive_table(v1, f"{SERVING_DB}.delay_airport_month")

    # 2) Cancel % per airport-month
    v2 = build_cancel_airport_month(flights)
    save_hive_table(v2, f"{SERVING_DB}.cancel_airport_month")

    # 3) Top10 airports
    v3 = build_top10_airports_month(v1)
    save_hive_table(v3, f"{SERVING_DB}.top10_airports_delay_month")

    # 4) Weather Delay View (Robust Load)
    v4 = build_delay_weather_region_month(spark, flights, airports)
    if v4: save_hive_table(v4, f"{SERVING_DB}.delay_weather_region_month")

    # 5) Delay carrier-month
    v5 = build_delay_carrier_month(flights)
    save_hive_table(v5, f"{SERVING_DB}.delay_carrier_month")

    # 6) Cancel Weather View (Robust Load)
    v6 = build_cancel_weather_region_month(spark, flights, airports)
    if v6: save_hive_table(v6, f"{SERVING_DB}.cancel_weather_region_month")

    # 7) Aircraft Age
    v7 = build_aircraft_age_bucket_carrier_year(spark, flights)
    if v7: save_hive_table(v7, f"{SERVING_DB}.aircraft_age_bucket_carrier_year")

    print("[DONE] Batch views created.")
    spark.stop()


if __name__ == "__main__":
    main()