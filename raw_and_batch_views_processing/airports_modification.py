from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim
from pyspark.sql.types import *

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("AirportDataSilverLayer") \
    .enableHiveSupport() \
    .getOrCreate()

# Bronze layer path
bronze_path = "hdfs:///original_data/airports"
# Silver layer path
silver_path = "hdfs:///silver_data/airports"


def transform_airport_data(df):
    """
    Transform bronze layer airport data to silver layer format
    """
    # AirportID: mapping from AIRPORT_ID
    df = df.withColumn("AirportID", col("AIRPORT_ID").cast("int"))

    # AirportStateName: mapping from AIRPORT_STATE_NAME
    df = df.withColumn("AirportStateName", trim(col("AIRPORT_STATE_NAME")))

    # AirportName: mapping from DISPLAY_AIRPORT_NAME
    df = df.withColumn("AirportName", trim(col("DISPLAY_AIRPORT_NAME")))

    # AirportCityName: mapping from DISPLAY_AIRPORT_CITY_NAME_FULL
    df = df.withColumn("AirportCityName", trim(col("DISPLAY_AIRPORT_CITY_NAME_FULL")))

    # Latitude & Longitude (These match your original code)
    df = df.withColumn("Latitude", col("LATITUDE").cast("float"))
    df = df.withColumn("Longitude", col("LONGITUDE").cast("float"))

    # Select final columns
    df = df.select(
        "AirportID",
        "AirportStateName",
        "AirportName",
        "AirportCityName",
        "Latitude",
        "Longitude"
    )
    return df


def get_bronze_files(file_pattern):
    """
    Get list of parquet files matching pattern in bronze layer
    """
    import re
    import subprocess

    # Use HDFS command to list files
    result = subprocess.run(
        ['hdfs', 'dfs', '-ls', bronze_path],
        capture_output=True,
        text=True
    )

    bronze_files = []
    pattern = re.compile(file_pattern)

    for line in result.stdout.split('\n'):
        if '.parquet' in line:
            parts = line.split()
            if len(parts) >= 8:
                file_path = parts[-1]
                file_name = file_path.split('/')[-1]

                if pattern.search(file_name):
                    bronze_files.append(file_path)

    return bronze_files


def get_existing_airport_count():
    """
    Get count of airports already in silver layer
    """
    try:
        count = spark.sql("SELECT COUNT(*) as cnt FROM silver.airport_dim").collect()[0].cnt
        return count
    except:
        return 0


def process_airport_data(force_reprocess=False):
    """
    Process airport data from bronze to silver layer

    Args:
        force_reprocess: If True, reprocess all data (full refresh)
                        If False, only add new airports (incremental)
    """

    # Find airport files in bronze layer
    # Assuming files named like: airports.parquet, airport_lookup.parquet, L_AIRPORT_ID.parquet
    bronze_files = get_bronze_files(r'.*airport.*\.parquet|L_AIRPORT_ID.*\.parquet')

    if not bronze_files:
        print("No airport files found in bronze layer")
        return

    print(f"Found {len(bronze_files)} airport file(s) in bronze layer")

    # Read all airport files
    all_data = None
    for file_path in bronze_files:
        print(f"Reading: {file_path}")
        try:
            df = spark.read.parquet(file_path)
            record_count = df.count()
            print(f"Records in file: {record_count}")

            if all_data is None:
                all_data = df
            else:
                all_data = all_data.union(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if all_data is None:
        print("No data to process")
        return

    # Remove duplicates based on AirportId
    print("Removing duplicates...")
    all_data = all_data.dropDuplicates(["AIRPORT_ID"])

    total_bronze_records = all_data.count()
    print(f"Total unique airport records in bronze: {total_bronze_records}")

    # Transform data
    print("Transforming data...")
    silver_df = transform_airport_data(all_data)

    if force_reprocess:
        # Full refresh - replace all data
        print("Full refresh mode - replacing all data")
        write_mode = "overwrite"
        records_to_write = silver_df.count()
        print(f"Writing {records_to_write} records")
    else:
        # Incremental mode - only add new airports
        existing_count = get_existing_airport_count()
        print(f"Existing airports in silver: {existing_count}")

        if existing_count > 0:
            # Get existing airport IDs
            existing_ids = spark.sql("SELECT AirportID FROM silver.airport_dim").collect()
            existing_id_set = {row.AirportID for row in existing_ids}

            # Filter out existing airports
            silver_df = silver_df.filter(~col("AirportID").isin(existing_id_set))
            new_records = silver_df.count()
            print(f"New airports to add: {new_records}")

            if new_records == 0:
                print("No new airports to add")
                return

            write_mode = "append"
        else:
            # First load
            print("First load - no existing data")
            write_mode = "overwrite"
            records_to_write = silver_df.count()
            print(f"Writing {records_to_write} records")

    # Write to silver layer
    print(f"Writing to silver layer: {silver_path} (mode: {write_mode})")
    silver_df.write \
        .mode(write_mode) \
        .parquet(silver_path)

    print(f"Data written to {silver_path}")

    # Create/update Hive table (only if first time or full refresh)
    if write_mode == "overwrite":
        print("Creating Hive table...")
        spark.sql("DROP TABLE IF EXISTS silver.airport_dim")

        spark.sql(f"""
            CREATE EXTERNAL TABLE silver.airport_dim (
                AirportID INT,
                AirportStateName STRING,
                AirportName STRING,
                AirportCityName STRING,
                Latitude FLOAT,
                Longitude FLOAT
            )
            STORED AS PARQUET
            LOCATION '{silver_path}'
        """)
    else:
        # For append mode, just refresh table metadata
        print("Refreshing table metadata...")
        spark.sql("REFRESH TABLE silver.airport_dim")

    print("Silver layer processing complete!")

    # Show sample data
    print("\nSample of newly processed data:")
    silver_df.show(10, truncate=False)

    # Show statistics
    print("\nCurrent data statistics:")
    stats_df = spark.sql("""
                         SELECT COUNT(*)                         as total_airports,
                                COUNT(DISTINCT AirportStateName) as unique_states,
                                COUNT(DISTINCT AirportCityName)  as unique_cities,
                                MIN(Latitude)                    as min_latitude,
                                MAX(Latitude)                    as max_latitude,
                                MIN(Longitude)                   as min_longitude,
                                MAX(Longitude)                   as max_longitude
                         FROM silver.airport_dim
                         """)
    stats_df.show(truncate=False)

    # Show top 10 states by airport count
    print("\nTop 10 states by number of airports:")
    spark.sql("""
              SELECT AirportStateName,
                     COUNT(*) as airport_count
              FROM silver.airport_dim
              GROUP BY AirportStateName
              ORDER BY airport_count DESC LIMIT 10
              """).show(truncate=False)


def initialize_silver_table():
    """
    Create Hive table structure if it doesn't exist
    """
    try:
        spark.sql("SELECT 1 FROM silver.airport_dim LIMIT 1")
        print("Airport table already exists")
    except:
        print("Initializing silver layer table...")
        spark.sql(f"""
            CREATE EXTERNAL TABLE IF NOT EXISTS silver.airport_dim (
                AirportID INT,
                AirportStateName STRING,
                AirportName STRING,
                AirportCityName STRING,
                Latitude FLOAT,
                Longitude FLOAT
            )
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

    # Parse command line arguments
    force_reprocess = False
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        force_reprocess = True
        print("Force reprocessing enabled")

    # Process airport data
    process_airport_data(force_reprocess=force_reprocess)

    print("\nProcessing complete!")

    spark.stop()