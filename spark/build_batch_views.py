from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ======================================================
# CONFIG
# ======================================================

SILVER_DB = "silver"
SERVING_DB = "serving"

FLIGHTS_TABLE = f"{SILVER_DB}.flight_facts"
AIRPORTS_TABLE = f"{SILVER_DB}.airport_dim"
WEATHER_TABLE = f"{SILVER_DB}.weather_airport_events"
AIRCRAFT_TABLE = f"{SILVER_DB}.aircraft_dim"

# flight_facts
COL_DEP_AIRPORT = "DepartureAirportID"
COL_ARR_AIRPORT = "ArrivalAirportID"
COL_CARRIER = "CarrierCode"
COL_CANCELLED = "IsCancelledFlag"
COL_DEP_DELAY = "DepartureDelay"
COL_ARR_DELAY = "ArrivalDelay"
COL_YEAR = "year_partition"
COL_MONTH = "month_partition"

# airport_dim
AIRPORT_ID_COL = "AirportID"
AIRPORT_REGION_COL = "AirportStateName"

COL_TAIL = "TailNumber"          # w flights
AIRCRAFT_TAIL = "TailNum"        # w aircraft_dim
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
    """
    Force the expected types to avoid Parquet dictionary / vectorized decoding issues
    and schema drift between runs.
    """
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
    """Average delays per departure airport per month."""
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
    """Cancellation percentage per departure airport per month."""
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
    """Average delays per carrier per month."""
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
    """Top 10 airports by avg departure delay per month (one row per month)."""
    w = Window.partitionBy(COL_YEAR, COL_MONTH).orderBy(F.col("avg_dep_delay").desc_nulls_last())

    ranked = (
        delay_airport_month
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= 10)
    )

    agg_exprs = []
    for i in range(1, 11):
        idx = f"{i:02d}"
        agg_exprs.append(
            F.max(F.when(F.col("rank") == i, F.col(COL_DEP_AIRPORT))).alias(f"rank{idx}_airport")
        )
        agg_exprs.append(
            F.max(F.when(F.col("rank") == i, F.col("avg_dep_delay"))).alias(f"rank{idx}_delay")
        )

    return ranked.groupBy(COL_YEAR, COL_MONTH).agg(*agg_exprs)


def build_delay_weather_region_month(spark, flights, airports):
    """
    Average delays vs weather event type per region (state) per month.

    Joins:
      flights.DepartureAirportID (STRING)
      airports.AirportID (INT -> cast to STRING)
      weather.AIRPORT_ID (STRING)

    Uses partitions:
      year_partition, month_partition
    """
    if not table_exists(spark, WEATHER_TABLE):
        print("[WARN] Weather table not found -> skipping weather view.")
        return None

    weather = spark.table(WEATHER_TABLE)

    require_columns(
        weather,
        ["AIRPORT_ID", "EVENT_TYPE", COL_YEAR, COL_MONTH],
        WEATHER_TABLE
    )
    require_columns(
        airports,
        [AIRPORT_ID_COL, AIRPORT_REGION_COL],
        AIRPORTS_TABLE
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

    # Reduce join explosion: unique (airport, event_type, year, month)
    weather_pre = (
        weather.select(
            F.col("AIRPORT_ID").cast("string").alias("AirportID_str"),
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
    """
    Cancel rate vs weather event type per region (state) per month.

    cancel_rate = avg(IsCancelledFlag)
    Row grain: Region, EventType, year_partition, month_partition
    """
    if not table_exists(spark, WEATHER_TABLE):
        print("[WARN] Weather table not found -> skipping cancel-weather view.")
        return None

    weather = spark.table(WEATHER_TABLE)

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
            F.col("AIRPORT_ID").cast("string").alias("AirportID_str"),
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
    """
    Avg delays by aircraft age bucket and carrier (per year_partition).

    aircraft_age = year_partition - aircraft.YearManufactured
    bucket: 0-5, 6-10, 11-20, >20
    """
    if not table_exists(spark, AIRCRAFT_TABLE):
        print("[WARN] Aircraft table not found -> skipping aircraft age view.")
        return None

    aircraft = spark.table(AIRCRAFT_TABLE)

    require_columns(aircraft, [AIRCRAFT_TAIL, AIRCRAFT_YEAR], AIRCRAFT_TABLE)

    flights_pre = flights.select(
        F.col(COL_CARRIER).cast("string").alias("CarrierCode"),
        F.col(COL_DEP_DELAY).cast("int").alias("DepartureDelay"),
        F.col(COL_ARR_DELAY).cast("int").alias("ArrivalDelay"),
        F.col(COL_YEAR).cast("int").alias("flight_year"),
        F.col(COL_TAIL).cast("string").alias("TailNumber")
    ).dropna(subset=["CarrierCode", "flight_year", "TailNumber"])

    aircraft_pre = aircraft.select(
        F.col(AIRCRAFT_TAIL).cast("string").alias("TailNum"),
        F.col(AIRCRAFT_YEAR).cast("int").alias("YearManufactured")
    ).dropna(subset=["TailNum", "YearManufactured"])

    joined = flights_pre.join(
        aircraft_pre,
        flights_pre["TailNumber"] == aircraft_pre["TailNum"],
        "inner"
    )

    age = (F.col("flight_year") - F.col("YearManufactured"))

    bucket = (
        F.when(age.between(0, 5), F.lit("0-5"))
         .when(age.between(6, 10), F.lit("6-10"))
         .when(age.between(11, 20), F.lit("11-20"))
         .when(age > 20, F.lit(">20"))
         .otherwise(F.lit("unknown"))
    )

    out = (
        joined
        .withColumn("aircraft_age_bucket", bucket)
        .groupBy("flight_year", "aircraft_age_bucket", "CarrierCode")
        .agg(
            F.avg("DepartureDelay").alias("avg_dep_delay"),
            F.avg("ArrivalDelay").alias("avg_arr_delay"),
            F.count(F.lit(1)).alias("flights_cnt")
        )
    )
    return out


# ======================================================
# MAIN
# ======================================================

def main():
    spark = (
        SparkSession.builder
        .appName("BuildServingBatchViews")
        .enableHiveSupport()
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .getOrCreate()
    )

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {SERVING_DB}")

    # Validate source tables exist
    if not table_exists(spark, FLIGHTS_TABLE):
        raise RuntimeError(f"Missing table: {FLIGHTS_TABLE}")
    if not table_exists(spark, AIRPORTS_TABLE):
        raise RuntimeError(f"Missing table: {AIRPORTS_TABLE}")

    flights = spark.table(FLIGHTS_TABLE)
    airports = spark.table(AIRPORTS_TABLE)

    # Validate required columns
    require_columns(
        flights,
        [COL_DEP_AIRPORT, COL_CARRIER, COL_CANCELLED, COL_DEP_DELAY, COL_ARR_DELAY, COL_YEAR, COL_MONTH, COL_TAIL],
        FLIGHTS_TABLE
    )

    # Force schema to stable types
    flights = normalize_flights_schema(flights)

    # Optional: drop rows without key partitions (defensive)
    flights = flights.dropna(subset=[COL_DEP_AIRPORT, COL_YEAR, COL_MONTH])

    # 1) Avg delay per airport-month
    v1 = build_delay_airport_month(flights)
    save_hive_table(v1, f"{SERVING_DB}.delay_airport_month")
    print("\n[BATCH VIEW] serving.delay_airport_month")
    v1.show(20, truncate=False)

    # 2) Cancel % per airport-month
    v2 = build_cancel_airport_month(flights)
    save_hive_table(v2, f"{SERVING_DB}.cancel_airport_month")
    print("\n[BATCH VIEW] serving.cancel_airport_month")
    v2.show(20, truncate=False)

    # 3) Top10 airports by delay per month
    v3 = build_top10_airports_month(v1)
    save_hive_table(v3, f"{SERVING_DB}.top10_airports_delay_month")
    print("\n[BATCH VIEW] serving.top10_airports_delay_month")
    v3.show(10, truncate=False)

    # 4) Delay vs weather event type per region/month (optional)
    v4 = build_delay_weather_region_month(spark, flights, airports)
    if v4 is not None:
        save_hive_table(v4, f"{SERVING_DB}.delay_weather_region_month")
        print("\n[BATCH VIEW] serving.delay_weather_region_month")
        v4.show(20, truncate=False)
    else:
        print("[INFO] Weather view not created (no compatible weather table).")

    # 5) Avg delay per carrier-month
    v5 = build_delay_carrier_month(flights)
    save_hive_table(v5, f"{SERVING_DB}.delay_carrier_month")
    print("\n[BATCH VIEW] serving.delay_carrier_month")
    v5.show(20, truncate=False)

    # 6) Cancel rate vs weather event type per region/month (optional)
    v6 = build_cancel_weather_region_month(spark, flights, airports)
    if v6 is not None:
        save_hive_table(v6, f"{SERVING_DB}.cancel_weather_region_month")
        print("\n[BATCH VIEW] serving.cancel_weather_region_month")
        v6.show(20, truncate=False)
    else:
        print("[INFO] Cancel-weather view not created (no compatible weather table).")

    # 7) Aircraft age bucket vs delays (optional)
    v7 = build_aircraft_age_bucket_carrier_year(spark, flights)
    if v7 is not None:
        save_hive_table(v7, f"{SERVING_DB}.aircraft_age_bucket_carrier_year")
        print("\n[BATCH VIEW] serving.aircraft_age_bucket_carrier_year")
        v7.show(20, truncate=False)
    else:
        print("[INFO] Aircraft-age view not created (no aircraft table).")


    print("[DONE] Batch views created in Hive database 'serving'.")
    spark.stop()


if __name__ == "__main__":
    main()
