from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, concat, lpad, to_timestamp,
    date_format, monotonically_increasing_id, lit
)
from pyspark.sql.types import *

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("FlightDataSilverLayer") \
    .enableHiveSupport() \
    .getOrCreate()

# Bronze layer path
bronze_path = "hdfs:///original_data/flights_data"
# Silver layer path
silver_path = "hdfs:///silver_data/flights"


def transform_flight_data(df):
    """
    Transform bronze layer flight data to silver layer format
    """


    # Keep original airport IDs (IATA codes)
    df = df.withColumn("DepartureAirportID", col("ORIGIN_AIRPORT_ID"))
    df = df.withColumn("ArrivalAirportID", col("DEST_AIRPORT_ID"))

    # Keep original carrier code instead of FK lookup
    df = df.withColumn("CarrierCode", col("OP_CARRIER"))

    # IsCancelledFlag: direct mapping
    df = df.withColumn("IsCancelledFlag", col("CANCELLED").cast("boolean"))

    # CancellationReasonName: map cancellation codes
    cancellation_map = {
        "A": "Carrier",
        "B": "Weather",
        "C": "National Air System",
        "D": "Security"
    }
    df = df.withColumn(
        "CancellationReasonName",
        when(col("CANCELLATION_CODE").isNull(), lit(None))
        .when(col("CANCELLATION_CODE") == "A", lit("Carrier"))
        .when(col("CANCELLATION_CODE") == "B", lit("Weather"))
        .when(col("CANCELLATION_CODE") == "C", lit("National Air System"))
        .when(col("CANCELLATION_CODE") == "D", lit("Security"))
        .otherwise(lit(None))
    )

    # ScheduledDepartureTime: convert HHMM to time
    df = df.withColumn(
        "ScheduledDepartureTime",
        when(col("CRS_DEP_TIME").isNotNull(),
             date_format(
                 to_timestamp(
                     lpad(col("CRS_DEP_TIME").cast("string"), 4, "0"),
                     "HHmm"
                 ),
                 "HH:mm:ss"
             )
             )
    )

    # DepartureDelay: set to 0 if negative or null
    df = df.withColumn(
        "DepartureDelay",
        when(col("DEP_DELAY").isNull(), lit(0))
        .when(col("DEP_DELAY") < 0, lit(0))
        .otherwise(col("DEP_DELAY"))
        .cast("int")
    )

    # DepartureTimeBlock: keep original time block name
    df = df.withColumn("DepartureTimeBlock", col("DEP_TIME_BLK"))

    # ScheduledArrivalTime: convert HHMM to time
    df = df.withColumn(
        "ScheduledArrivalTime",
        when(col("CRS_ARR_TIME").isNotNull(),
             date_format(
                 to_timestamp(
                     lpad(col("CRS_ARR_TIME").cast("string"), 4, "0"),
                     "HHmm"
                 ),
                 "HH:mm:ss"
             )
             )
    )

    # ArrivalDelay: set to 0 if negative, keep null if null
    df = df.withColumn(
        "ArrivalDelay",
        when(col("ARR_DELAY").isNull(), lit(None))
        .when(col("ARR_DELAY") < 0, lit(0))
        .otherwise(col("ARR_DELAY"))
        .cast("int")
    )

    # ArrivalTimeBlock: keep original time block name
    df = df.withColumn("ArrivalTimeBlock", col("ARR_TIME_BLK"))

    # IsDivertedFlag: direct mapping
    df = df.withColumn("IsDivertedFlag", col("DIVERTED").cast("boolean"))

    # PlaneId: keep original tail number
    df = df.withColumn("TailNumber", col("TAIL_NUM"))

    # Add auto-increment FactID
    df = df.withColumn("FactID", monotonically_increasing_id())

    # Add partition columns for Hive
    df = df.withColumn("year_partition", col("YEAR"))
    df = df.withColumn("month_partition", col("MONTH"))

    # Select final columns in order
    df = df.select(
        "FactID",
        "DepartureAirportID",
        "ArrivalAirportID",
        "CarrierCode",
        "IsCancelledFlag",
        "CancellationReasonName",
        "ScheduledDepartureTime",
        "DepartureDelay",
        "DepartureTimeBlock",
        "ScheduledArrivalTime",
        "ArrivalDelay",
        "ArrivalTimeBlock",
        "IsDivertedFlag",
        "TailNumber",
        "year_partition",
        "month_partition"
    )

    return df


def get_bronze_files():
    """
    Get list of all parquet files in bronze layer
    Returns dict: {(year, month): file_path}
    """
    import re
    import subprocess

    # Use HDFS command to list files
    result = subprocess.run(
        ['hdfs', 'dfs', '-ls', bronze_path],
        capture_output=True,
        text=True
    )

    bronze_files = {}
    pattern = re.compile(r'(\d{4})_(\d{1,2})_aotd\.parquet')

    for line in result.stdout.split('\n'):
        if '.parquet' in line:
            # Parse HDFS ls output: permissions, replication, user, group, size, date, time, path
            parts = line.split()
            if len(parts) >= 8:
                file_path = parts[-1]
                file_name = file_path.split('/')[-1]

                match = pattern.search(file_name)
                if match:
                    year = int(match.group(1))
                    month = int(match.group(2))
                    bronze_files[(year, month)] = file_path

    return bronze_files


def get_existing_partitions():
    """
    Get list of existing partitions in silver layer
    """
    try:
        existing_partitions = spark.sql(
            "SHOW PARTITIONS silver.flight_facts"
        ).collect()

        partitions = set()
        for row in existing_partitions:
            # Parse partition string like "year_partition=2024/month_partition=1"
            parts = row.partition.split('/')
            year = int(parts[0].split('=')[1])
            month = int(parts[1].split('=')[1])
            partitions.add((year, month))

        return partitions
    except:
        # Table doesn't exist yet
        return set()


def process_partitions(year_month_list=None, force_reprocess=False):
    """
    Process specific partitions or new partitions only

    Args:
        year_month_list: List of tuples [(year, month), ...] to process
                        If None, process only new partitions
        force_reprocess: If True, reprocess even if partition exists
    """

    # Get available bronze files
    bronze_files = get_bronze_files()
    print(f"Found {len(bronze_files)} files in bronze layer")

    # Determine which partitions to process
    if year_month_list:
        partitions_to_process = set(year_month_list)
        print(f"Processing specified partitions: {partitions_to_process}")
    else:
        # Find new partitions
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
    for year, month in sorted(partitions_to_process):
        if (year, month) not in bronze_files:
            print(f"Warning: No bronze file found for year={year}, month={month}")
            continue

        file_path = bronze_files[(year, month)]
        print(f"\nProcessing: {file_path}")
        print(f"Partition: year={year}, month={month}")

        # Read specific file from bronze layer
        try:
            bronze_df = spark.read.parquet(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        record_count = bronze_df.count()
        print(f"Records in file: {record_count}")

        if record_count == 0:
            print(f"Skipping empty file: {file_path}")
            continue

        # Transform data
        silver_df = transform_flight_data(bronze_df)

        # Write partition (overwrite mode for idempotency)
        partition_output = f"{silver_path}/year_partition={year}/month_partition={month}"
        silver_df.select([c for c in silver_df.columns if c not in ['year_partition', 'month_partition']]) \
            .write \
            .mode("overwrite") \
            .parquet(partition_output)

        print(f"Written to: {partition_output}")

    # Repair partitions to register new ones with Hive
    print("\nRepairing table partitions...")
    spark.sql("MSCK REPAIR TABLE silver.flight_facts")

    print("Processing complete!")


def initialize_silver_table():
    """
    Create Hive table structure if it doesn't exist
    """
    print("Initializing silver layer table...")
    spark.sql(f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS silver.flight_facts (
            FactID BIGINT,
            DepartureAirportID INT,
            ArrivalAirportID INT,
            CarrierCode STRING,
            IsCancelledFlag BOOLEAN,
            CancellationReasonName STRING,
            ScheduledDepartureTime STRING,
            DepartureDelay INT,
            DepartureTimeBlock STRING,
            ScheduledArrivalTime STRING,
            ArrivalDelay INT,
            ArrivalTimeBlock STRING,
            IsDivertedFlag BOOLEAN,
            TailNumber STRING
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

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--full":
            # Full reprocess
            process_partitions(force_reprocess=True)
        elif sys.argv[1] == "--partition":
            # Process specific partitions
            # Example: --partition 2024,1 2024,2
            partitions = []
            for i in range(2, len(sys.argv)):
                year, month = sys.argv[i].split(',')
                partitions.append((int(year), int(month)))
            process_partitions(year_month_list=partitions)
        else:
            print("Usage:")
            print("  python script.py                    # Process new partitions only")
            print("  python script.py --full             # Reprocess all partitions")
            print("  python script.py --partition 2024,1 2024,2  # Process specific partitions")
    else:
        # Default: process only new partitions
        process_partitions()

    # Show summary
    print("\nSummary:")
    partition_count = spark.sql("SHOW PARTITIONS silver.flight_facts").count()
    print(f"Total partitions: {partition_count}")

    spark.stop()