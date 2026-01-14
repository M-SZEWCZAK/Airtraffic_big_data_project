from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, concat, lit, trim, upper
)
from pyspark.sql.types import *


# Initialize Spark Session
spark = SparkSession.builder \
    .appName("AircraftDataSilverLayer") \
    .enableHiveSupport() \
    .getOrCreate()

# Bronze layer path
bronze_path = "hdfs:///original_data/planes"
# Silver layer path
silver_path = "hdfs:///silver_data/aircraft"

# Code lookup dictionaries based on FAA data
AIRCRAFT_TYPE_CODES = {
    "1": "Glider",
    "2": "Balloon",
    "3": "Blimp/Dirigible",
    "4": "Fixed wing single engine",
    "5": "Fixed wing multi engine",
    "6": "Rotorcraft",
    "7": "Weight-shift-control",
    "8": "Powered Parachute",
    "9": "Gyroplane"
}

ENGINE_TYPE_CODES = {
    "0": "None",
    "1": "Reciprocating",
    "2": "Turbo-prop",
    "3": "Turbo-shaft",
    "4": "Turbo-jet",
    "5": "Turbo-fan",
    "6": "Ramjet",
    "7": "2 Cycle",
    "8": "4 Cycle",
    "9": "Unknown",
    "10": "Electric",
    "11": "Rotary"
}


def transform_aircraft_data(df):
    """
    Transform bronze layer aircraft data to silver layer format
    """

    # TailNum: prepend 'N' to N_NUMBER
    df = df.withColumn(
        "TailNum",
        concat(lit("N"), col("N-NUMBER"))
    )

    # ManufacturerName: direct mapping with trimming
    df = df.withColumn(
        "ManufacturerName",
        trim(col("MFR"))
    )

    # ModelName: direct mapping with trimming
    df = df.withColumn(
        "ModelName",
        trim(col("MODEL"))
    )

    # AircraftTypeName: code lookup
    aircraft_type_mapping = when(col("TYPE AIRCRAFT").isNull(), lit(None))
    for code, name in AIRCRAFT_TYPE_CODES.items():
        aircraft_type_mapping = aircraft_type_mapping.when(
            col("TYPE AIRCRAFT") == code, lit(name)
        )
    aircraft_type_mapping = aircraft_type_mapping.otherwise(lit(None))

    df = df.withColumn("AircraftTypeName", aircraft_type_mapping)

    # EngineTypeName: code lookup
    engine_type_mapping = when(col("TYPE ENGINE").isNull(), lit(None))
    for code, name in ENGINE_TYPE_CODES.items():
        engine_type_mapping = engine_type_mapping.when(
            col("TYPE ENGINE") == code, lit(name)
        )
    engine_type_mapping = engine_type_mapping.otherwise(lit(None))

    df = df.withColumn("EngineTypeName", engine_type_mapping)

    # YearManufactured: type conversion to int
    df = df.withColumn(
        "YearManufactured",
        col("YEAR MFR").cast("int")
    )

    # Select final columns
    df = df.select(
        "TailNum",
        "ManufacturerName",
        "ModelName",
        "AircraftTypeName",
        "EngineTypeName",
        "YearManufactured"
    )

    return df


# def get_bronze_files(file_pattern):
#     """
#     Get list of parquet files matching pattern in bronze layer
#     """
#     import re
#     import subprocess
#
#     # Use HDFS command to list files
#     result = subprocess.run(
#         ['hdfs', 'dfs', '-ls', bronze_path],
#         capture_output=True,
#         text=True
#     )
#
#     bronze_files = []
#     pattern = re.compile(file_pattern)
#
#     for line in result.stdout.split('\n'):
#         if '.parquet' in line:
#             parts = line.split()
#             if len(parts) >= 8:
#                 file_path = parts[-1]
#                 file_name = file_path.split('/')[-1]
#
#                 if pattern.search(file_name):
#                     bronze_files.append(file_path)
#
#     return bronze_files


def get_bronze_files(file_pattern):
    """
    Get list of parquet files matching pattern in bronze layer
    """
    import re
    import subprocess
    from urllib.parse import urlparse

    # Use HDFS command to list files
    result = subprocess.run(
        ['hdfs', 'dfs', '-ls', bronze_path],
        capture_output=True,
        text=True
    )

    bronze_files = []
    pattern = re.compile(file_pattern)

    # for line in result.stdout.split('\n'):
    #     if '.parquet' in line:
    #         parts = line.split()
    #         if len(parts) >= 8:
    #             # 1. Get the raw path from HDFS output
    #             raw_path = parts[-1]
    #
    #             # 2. STRIP THE URI: Convert 'hdfs://node1/path' to '/path'
    #             # This prevents the 'Path does not exist' error
    #             clean_path = urlparse(raw_path).path
    #
    #             # 3. Extract filename for regex matching
    #             file_name = clean_path.split('/')[-1]
    #
    #             if pattern.search(file_name):
    #                 bronze_files.append(clean_path)

    for line in result.stdout.split('\n'):
        line = line.strip()
        if not line:
            continue
        if not line.endswith(".parquet"):
            continue

        parts = line.split()
        if len(parts) >= 8:
            # 1. Get the raw path from HDFS output
            raw_path = parts[-1]

            # 2. Strip the URI
            clean_path = urlparse(raw_path).path if "://" in raw_path else raw_path

            # 3. Extract filename for regex matching
            file_name = clean_path.split('/')[-1]

            if pattern.search(file_name):
                # Return consistently hdfs:///...
                bronze_files.append("hdfs:///" + clean_path.lstrip("/"))

    return bronze_files

def get_existing_tail_numbers():
    """
    Get set of tail numbers already in silver layer
    """
    try:
        existing = spark.sql("SELECT TailNum FROM silver.aircraft_dim").collect()
        return {row.TailNum for row in existing}
    except:
        # Table doesn't exist yet
        return set()


def process_aircraft_data(force_reprocess=False):
    """
    Process aircraft data from bronze to silver layer

    Args:
        force_reprocess: If True, reprocess all data (full refresh)
                        If False, only add new tail numbers (incremental)
    """

    # Find aircraft files in bronze layer
    bronze_files = get_bronze_files(r'plane_registry_\d{4}-\d{2}-\d{2}\.parquet')

    if not bronze_files:
        print("No aircraft files found in bronze layer")
        return

    print(f"Found {len(bronze_files)} aircraft file(s) in bronze layer")

    # Read all aircraft files
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

    # Remove duplicates based on TailNum (N_NUMBER) within bronze data
    print("Removing duplicates in bronze data...")
    all_data = all_data.dropDuplicates(["N-NUMBER"])

    total_bronze_records = all_data.count()
    print(f"Total unique aircraft records in bronze: {total_bronze_records}")

    # Transform data
    print("Transforming data...")
    silver_df = transform_aircraft_data(all_data)

    if force_reprocess:
        # Full refresh - replace all data
        print("Full refresh mode - replacing all data")
        write_mode = "overwrite"
        records_to_write = silver_df.count()
        print(f"Writing {records_to_write} records")
    else:
        # Incremental mode - only add new tail numbers
        existing_tail_numbers = get_existing_tail_numbers()
        print(f"Existing aircraft in silver: {len(existing_tail_numbers)}")

        if existing_tail_numbers:
            # Filter out existing tail numbers
            silver_df = silver_df.filter(~col("TailNum").isin(existing_tail_numbers))
            new_records = silver_df.count()
            print(f"New aircraft to add: {new_records}")

            if new_records == 0:
                print("No new aircraft to add")
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
        spark.sql("DROP TABLE IF EXISTS silver.aircraft_dim")

        spark.sql(f"""
            CREATE EXTERNAL TABLE silver.aircraft_dim (
                TailNum STRING,
                ManufacturerName STRING,
                ModelName STRING,
                AircraftTypeName STRING,
                EngineTypeName STRING,
                YearManufactured INT
            )
            STORED AS PARQUET
            LOCATION '{silver_path}'
        """)
    else:
        # For append mode, just refresh table metadata
        print("Refreshing table metadata...")
        spark.sql("REFRESH TABLE silver.aircraft_dim")

    print("Silver layer processing complete!")

    # Show sample data
    print("\nSample of newly processed data:")
    silver_df.show(10, truncate=False)

    # Show statistics
    print("\nCurrent data statistics:")
    stats_df = spark.sql("""
                         SELECT COUNT(*)                         as total_aircraft,
                                COUNT(DISTINCT ManufacturerName) as unique_manufacturers,
                                COUNT(DISTINCT AircraftTypeName) as aircraft_types,
                                COUNT(DISTINCT EngineTypeName)   as engine_types,
                                MIN(YearManufactured)            as oldest_year,
                                MAX(YearManufactured)            as newest_year
                         FROM silver.aircraft_dim
                         """)
    stats_df.show(truncate=False)


def initialize_silver_table():
    """
    Create Hive table structure if it doesn't exist
    """
    try:
        spark.sql("SELECT 1 FROM silver.aircraft_dim LIMIT 1")
        print("Aircraft table already exists")
    except:
        print("Initializing silver layer table...")
        spark.sql(f"""
            CREATE EXTERNAL TABLE IF NOT EXISTS silver.aircraft_dim (
                TailNum STRING,
                ManufacturerName STRING,
                ModelName STRING,
                AircraftTypeName STRING,
                EngineTypeName STRING,
                YearManufactured INT
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

    # Process aircraft data
    process_aircraft_data(force_reprocess=force_reprocess)

    print("\nProcessing complete!")

    spark.stop()