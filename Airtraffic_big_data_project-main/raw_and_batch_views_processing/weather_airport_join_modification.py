from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, to_timestamp, concat, lpad, floor, ceil,
    explode, sequence, date_add, expr, sqrt, sin, cos, atan2, radians,
    monotonically_increasing_id, year, month
)
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql import Window
import re
import subprocess

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("WeatherDataSilverLayer") \
    .enableHiveSupport() \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Paths
bronze_path = "hdfs:///original_data/weather"
silver_path = "hdfs:///silver_data/weather_airport_events"
airport_bronze_path = "hdfs:///original_data/airports"


def add_event_category_id(df):
    """Add EVENT_CATEGORY_ID based on EVENT_TYPE"""
    return df.withColumn(
        "EVENT_CATEGORY_ID",
        when(col("EVENT_TYPE").isin("Tornado", "Funnel Cloud"), 0)
        .when(col("EVENT_TYPE").isin("Hail", "Heavy Rain", "Thunderstorm Wind"), 1)
        .when(col("EVENT_TYPE").isin("Hurricane", "Marine Hurricane/Typhoon", "Tropical Storm"), 2)
        .when(col("EVENT_TYPE").isin("Winter Storm", "Blizzard", "Heavy Snow", "Ice Storm"), 3)
        .when(col("EVENT_TYPE").isin("Dust Storm", "Dense Fog"), 4)
        .otherwise(None)
    )


def create_at_least_hour_separated_begin_and_end_datetimes(df):
    """Create BEGIN_DATETIME and END_DATETIME from separate date/time columns"""

    # Create BEGIN_DATETIME
    df = df.withColumn(
        "BEGIN_DATETIME_STR",
        concat(
            col("BEGIN_YEARMONTH").cast("string"),
            lpad(col("BEGIN_DAY").cast("string"), 2, "0"),
            lpad(col("BEGIN_TIME").cast("string"), 4, "0")
        )
    )

    df = df.withColumn(
        "BEGIN_DATETIME_PARSED",
        to_timestamp(col("BEGIN_DATETIME_STR"), "yyyyMMddHHmm")
    )

    # Check if already on the hour (minute == 0)
    df = df.withColumn(
        "on_the_hour",
        (expr("minute(BEGIN_DATETIME_PARSED)") == 0)
    )

    # Subtract 1 hour if already on the hour, then floor to hour
    df = df.withColumn(
        "BEGIN_DATETIME",
        when(col("on_the_hour"),
             expr("date_trunc('hour', BEGIN_DATETIME_PARSED - INTERVAL 1 HOUR)"))
        .otherwise(expr("date_trunc('hour', BEGIN_DATETIME_PARSED)"))
    )

    # Create END_DATETIME
    df = df.withColumn(
        "END_DATETIME_STR",
        concat(
            col("END_YEARMONTH").cast("string"),
            lpad(col("END_DAY").cast("string"), 2, "0"),
            lpad(col("END_TIME").cast("string"), 4, "0")
        )
    )

    df = df.withColumn(
        "END_DATETIME_PARSED",
        to_timestamp(col("END_DATETIME_STR"), "yyyyMMddHHmm")
    )

    # Ceil to next hour
    df = df.withColumn(
        "END_DATETIME",
        expr("date_trunc('hour', END_DATETIME_PARSED + INTERVAL 1 HOUR)")
    )

    # Drop temporary columns
    df = df.drop("BEGIN_DATETIME_STR", "BEGIN_DATETIME_PARSED", "on_the_hour",
                 "END_DATETIME_STR", "END_DATETIME_PARSED")

    return df


def preprocess_weather_data(weather_events_df):
    """Filter and preprocess weather events data"""
    event_types = [
        'Heavy Snow', 'Thunderstorm Wind', 'Hail', 'Funnel Cloud', 'Heavy Rain',
        'Tornado', 'Winter Storm', 'Blizzard', 'Dust Storm', 'Dense Fog',
        'Ice Storm', 'Tropical Storm', 'Hurricane', 'Marine Hurricane/Typhoon'
    ]

    # Filter by event types
    weather_events_df = weather_events_df.filter(col("EVENT_TYPE").isin(event_types))

    # Filter out rows with null coordinates
    weather_events_df = weather_events_df.filter(
        col("BEGIN_LAT").isNotNull() &
        col("BEGIN_LON").isNotNull() &
        col("END_LAT").isNotNull() &
        col("END_LON").isNotNull()
    )

    # Create datetime columns
    weather_events_df = create_at_least_hour_separated_begin_and_end_datetimes(weather_events_df)

    # Add event category ID
    weather_events_df = add_event_category_id(weather_events_df)

    # Select final columns
    columns_to_select = [
        'BEGIN_DATETIME', 'END_DATETIME', 'BEGIN_LAT', 'BEGIN_LON',
        'END_LAT', 'END_LON', 'EVENT_TYPE', 'EVENT_CATEGORY_ID'
    ]
    weather_events_df = weather_events_df.select(*columns_to_select)

    return weather_events_df


def create_hourly_weather_points(weather_df):
    """Explode weather events into hourly data points with interpolated coordinates"""

    # Add unique ID for tracking original events
    weather_df = weather_df.withColumn("ORIGINAL_INDEX", monotonically_increasing_id())

    # Calculate duration in hours
    weather_df = weather_df.withColumn(
        "duration_hours",
        expr("cast((unix_timestamp(END_DATETIME) - unix_timestamp(BEGIN_DATETIME)) / 3600 as int)")
    )

    # Create array of hour offsets
    weather_df = weather_df.withColumn(
        "hour_offsets",
        sequence(lit(0), col("duration_hours"))
    )

    # Explode to create one row per hour
    weather_points = weather_df.withColumn("HOUR_OFFSET", explode(col("hour_offsets")))

    # Calculate interpolation factor
    weather_points = weather_points.withColumn(
        "INTERPOLATION_FACTOR",
        when(col("duration_hours") == 0, lit(0.0))
        .otherwise(col("HOUR_OFFSET").cast("double") / col("duration_hours").cast("double"))
    )

    # Calculate current datetime
    weather_points = weather_points.withColumn(
        "DATETIME",
        expr("BEGIN_DATETIME + make_interval(0, 0, 0, 0, HOUR_OFFSET, 0, 0)")
    )

    # Interpolate latitude and longitude
    weather_points = weather_points.withColumn(
        "LATITUDE",
        col("BEGIN_LAT") + col("INTERPOLATION_FACTOR") * (col("END_LAT") - col("BEGIN_LAT"))
    )

    weather_points = weather_points.withColumn(
        "LONGITUDE",
        col("BEGIN_LON") + col("INTERPOLATION_FACTOR") * (col("END_LON") - col("BEGIN_LON"))
    )

    # Select final columns
    weather_points = weather_points.select(
        "ORIGINAL_INDEX",
        "DATETIME",
        "LATITUDE",
        "LONGITUDE",
        "EVENT_TYPE",
        "EVENT_CATEGORY_ID",
        "BEGIN_DATETIME",
        "END_DATETIME",
        "HOUR_OFFSET",
        "INTERPOLATION_FACTOR"
    )

    return weather_points


def haversine_distance_udf():
    """Create Haversine distance calculation expression"""
    distance_expr = """
        6371 * 2 * asin(sqrt(
            power(sin(radians((AIRPORT_LATITUDE - WEATHER_LATITUDE) / 2)), 2) +
            cos(radians(WEATHER_LATITUDE)) * cos(radians(AIRPORT_LATITUDE)) *
            power(sin(radians((AIRPORT_LONGITUDE - WEATHER_LONGITUDE) / 2)), 2)
        ))
    """
    return distance_expr


def weather_airport_join(weather_df, airport_df, max_distance_km):
    """
    Perform complex join between weather events and airports based on distance.
    """

    print("Creating hourly weather data points...")
    weather_points = create_hourly_weather_points(weather_df)

    print(f"Created {weather_points.count()} weather data points")

    print("Cross joining weather points with airports...")

    # Rename airport columns to avoid conflicts
    airport_df = airport_df.select(
        col("AIRPORT_ID"),
        col("DISPLAY_AIRPORT_NAME"),
        col("AIRPORT_COUNTRY_NAME"),
        col("LATITUDE").alias("AIRPORT_LATITUDE"),
        col("LONGITUDE").alias("AIRPORT_LONGITUDE")
    )

    # Rename weather point coordinates
    weather_points = weather_points.withColumnRenamed("LATITUDE", "WEATHER_LATITUDE") \
        .withColumnRenamed("LONGITUDE", "WEATHER_LONGITUDE")

    # Cross join
    joined = weather_points.crossJoin(airport_df)

    print("Calculating distances...")

    # Calculate Haversine distance
    joined = joined.withColumn("DISTANCE_KM", expr(haversine_distance_udf()))

    # Filter by max distance
    joined = joined.filter(col("DISTANCE_KM") <= max_distance_km)

    print("Sorting results...")

    # Sort by original weather index, hour offset, and distance
    result = joined.orderBy("ORIGINAL_INDEX", "HOUR_OFFSET", "DISTANCE_KM")

    # Select and rename final columns
    result = result.select(
        col("ORIGINAL_INDEX").alias("ORIGINAL_WEATHER_INDEX"),
        "DATETIME",
        "WEATHER_LATITUDE",
        "WEATHER_LONGITUDE",
        "EVENT_TYPE",
        "EVENT_CATEGORY_ID",
        "BEGIN_DATETIME",
        "END_DATETIME",
        "HOUR_OFFSET",
        "INTERPOLATION_FACTOR",
        "AIRPORT_ID",
        "DISPLAY_AIRPORT_NAME",
        "AIRPORT_COUNTRY_NAME",
        "AIRPORT_LATITUDE",
        "AIRPORT_LONGITUDE",
        "DISTANCE_KM"
    )

    return result


def get_bronze_files():
    """
    Get list of all parquet files in bronze weather layer
    Returns dict: {(year, month): file_path}
    """
    result = subprocess.run(
        ['hdfs', 'dfs', '-ls', bronze_path],
        capture_output=True,
        text=True
    )

    bronze_files = {}
    pattern = re.compile(r'storm_events_(\d{4})_(\d{2}).*\.parquet')

    for line in result.stdout.split('\n'):
        if '.parquet' in line:
            parts = line.split()
            if len(parts) >= 8:
                file_path = parts[-1]
                file_name = file_path.split('/')[-1]

                match = pattern.search(file_name)
                if match:
                    yr = int(match.group(1))
                    mo = int(match.group(2))
                    bronze_files[(yr, mo)] = file_path

    return bronze_files


def get_existing_partitions():
    """
    Get list of existing partitions in silver layer
    Returns set of (year, month) tuples
    """
    try:
        existing_partitions = spark.sql(
            "SHOW PARTITIONS silver.weather_airport_events"
        ).collect()

        partitions = set()
        for row in existing_partitions:
            parts = row.partition.split('/')
            yr = int(parts[0].split('=')[1])
            mo = int(parts[1].split('=')[1])
            partitions.add((yr, mo))

        return partitions
    except:
        return set()


def load_airport_data():
    """
    Load airport dimension data from bronze layer - US airports only
    """
    result = subprocess.run(
        ['hdfs', 'dfs', '-ls', airport_bronze_path],
        capture_output=True,
        text=True
    )

    for line in result.stdout.split('\n'):
        if 'airport' in line.lower() and '.parquet' in line:
            parts = line.split()
            if len(parts) >= 8:
                file_path = parts[-1]
                print(f"Loading airports from: {file_path}")
                airports_df = spark.read.parquet(file_path)

                # Filter for US airports only
                us_airports = airports_df.filter(
                    col("AIRPORT_COUNTRY_NAME") == "United States"
                )

                us_count = us_airports.count()
                total_count = airports_df.count()
                print(f"Filtered to {us_count} US airports (from {total_count} total)")

                return us_airports

    raise FileNotFoundError("No airport parquet file found in bronze layer")


def process_partitions(partition_list=None, force_reprocess=False, max_distance_km=50):
    """
    Process specific partitions or new partitions only
    partition_list: list of (year, month) tuples
    """

    # Load airport data once (dimension table)
    print("Loading airport data...")
    airports = load_airport_data()
    print(f"Loaded {airports.count()} airports")

    # Get available bronze files
    bronze_files = get_bronze_files()
    print(f"Found {len(bronze_files)} weather files in bronze layer")

    # Determine which partitions to process
    if partition_list:
        partitions_to_process = set(partition_list)
        print(f"Processing specified partitions: {partitions_to_process}")
    else:
        bronze_partitions = set(bronze_files.keys())
        existing_partitions = get_existing_partitions()

        if force_reprocess:
            partitions_to_process = bronze_partitions
            print(f"Force reprocessing all {len(partitions_to_process)} partitions")
        else:
            partitions_to_process = bronze_partitions - existing_partitions
            print(f"Found {len(partitions_to_process)} new partitions to process")
            print(f"Existing partitions: {len(existing_partitions)}")

    if not partitions_to_process:
        print("No new partitions to process")
        return

    # Process each partition
    for yr, mo in sorted(partitions_to_process):
        if (yr, mo) not in bronze_files:
            print(f"Warning: No bronze file found for year={yr}, month={mo}")
            continue

        file_path = bronze_files[(yr, mo)]
        print(f"\n{'=' * 80}")
        print(f"Processing: {file_path}")
        print(f"Partition: year={yr}, month={mo}")
        print(f"{'=' * 80}")

        try:
            # Read weather data
            weather_raw = spark.read.parquet(file_path)
            record_count = weather_raw.count()
            print(f"Records in file: {record_count}")

            if record_count == 0:
                print(f"Skipping empty file")
                continue

            # Preprocess weather data
            print("Preprocessing weather events...")
            weather_processed = preprocess_weather_data(weather_raw)
            processed_count = weather_processed.count()
            print(f"After preprocessing: {processed_count} weather events")

            if processed_count == 0:
                print("No events remaining after preprocessing")
                continue

            # Perform weather-airport join
            print(f"Joining with airports (max distance: {max_distance_km}km)...")
            result = weather_airport_join(weather_processed, airports, max_distance_km)

            joined_count = result.count()
            print(f"Created {joined_count} weather-airport event records")

            if joined_count == 0:
                print("No weather-airport matches found")
                continue

            # Add partition columns
            result = result.withColumn("year_partition", lit(yr)) \
                           .withColumn("month_partition", lit(mo))

            # Write partition
            partition_output = f"{silver_path}/year_partition={yr}/month_partition={mo}"
            print(f"Writing to: {partition_output}")

            result.select([c for c in result.columns if c not in ['year_partition', 'month_partition']]) \
                .write \
                .mode("overwrite") \
                .parquet(partition_output)

            print(f"Successfully written partition")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Repair partitions
    print("\n" + "=" * 80)
    print("Repairing table partitions...")
    spark.sql("MSCK REPAIR TABLE silver.weather_airport_events")

    print("Processing complete!")


def initialize_silver_table():
    """
    Create Hive table structure if it doesn't exist
    """
    print("Initializing silver layer table...")
    spark.sql(f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS silver.weather_airport_events (
            ORIGINAL_WEATHER_INDEX BIGINT,
            DATETIME TIMESTAMP,
            WEATHER_LATITUDE DOUBLE,
            WEATHER_LONGITUDE DOUBLE,
            EVENT_TYPE STRING,
            EVENT_CATEGORY_ID INT,
            BEGIN_DATETIME TIMESTAMP,
            END_DATETIME TIMESTAMP,
            HOUR_OFFSET INT,
            INTERPOLATION_FACTOR DOUBLE,
            AIRPORT_ID STRING,
            DISPLAY_AIRPORT_NAME STRING,
            AIRPORT_COUNTRY_NAME STRING,
            AIRPORT_LATITUDE DOUBLE,
            AIRPORT_LONGITUDE DOUBLE,
            DISTANCE_KM DOUBLE
        )
        PARTITIONED BY (year_partition INT, month_partition INT)
        STORED AS PARQUET
        LOCATION '{silver_path}'
    """)
    print("Table initialized")


if __name__ == "__main__":
    import sys

    # Create database if it doesn't exist
    spark.sql("CREATE DATABASE IF NOT EXISTS silver")

    # Initialize table structure
    initialize_silver_table()

    # Configuration
    MAX_DISTANCE_KM = 50

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--full":
            process_partitions(force_reprocess=True, max_distance_km=MAX_DISTANCE_KM)
        elif sys.argv[1] == "--partition":
            partitions = []
            i = 2
            while i < len(sys.argv):
                if i + 1 < len(sys.argv):
                    yr = int(sys.argv[i])
                    mo = int(sys.argv[i + 1])
                    partitions.append((yr, mo))
                    i += 2
                else:
                    print("Error: --partition requires year month pairs")
                    sys.exit(1)
            process_partitions(partition_list=partitions, max_distance_km=MAX_DISTANCE_KM)
        else:
            print("Usage:")
            print("  python script.py                           # Process new partitions only")
            print("  python script.py --full                    # Reprocess all partitions")
            print("  python script.py --partition 2024 1 2024 2  # Process specific partitions")
    else:
        process_partitions(max_distance_km=MAX_DISTANCE_KM)

    # Show summary
    print("\n" + "=" * 80)
    print("Summary:")
    partition_count = spark.sql("SHOW PARTITIONS silver.weather_airport_events").count()
    print(f"Total partitions: {partition_count}")

    print("\nSample data:")
    spark.sql("SELECT * FROM silver.weather_airport_events LIMIT 5").show(truncate=False)

    spark.stop()