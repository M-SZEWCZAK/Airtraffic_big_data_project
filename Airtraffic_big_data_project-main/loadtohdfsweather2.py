from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CSV to Parquet Converter") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .getOrCreate()

# Define schema for better type inference and performance
schema = StructType([
    StructField("BEGIN_YEARMONTH", IntegerType(), True),
    StructField("BEGIN_DAY", IntegerType(), True),
    StructField("BEGIN_TIME", IntegerType(), True),
    StructField("END_YEARMONTH", IntegerType(), True),
    StructField("END_DAY", IntegerType(), True),
    StructField("END_TIME", IntegerType(), True),
    StructField("EPISODE_ID", IntegerType(), True),
    StructField("EVENT_ID", IntegerType(), True),
    StructField("STATE", StringType(), True),
    StructField("STATE_FIPS", IntegerType(), True),
    StructField("YEAR", IntegerType(), True),
    StructField("MONTH_NAME", StringType(), True),
    StructField("EVENT_TYPE", StringType(), True),
    StructField("CZ_TYPE", StringType(), True),
    StructField("CZ_FIPS", IntegerType(), True),
    StructField("CZ_NAME", StringType(), True),
    StructField("WFO", StringType(), True),
    StructField("BEGIN_DATE_TIME", StringType(), True),
    StructField("CZ_TIMEZONE", StringType(), True),
    StructField("END_DATE_TIME", StringType(), True),
    StructField("INJURIES_DIRECT", IntegerType(), True),
    StructField("INJURIES_INDIRECT", IntegerType(), True),
    StructField("DEATHS_DIRECT", IntegerType(), True),
    StructField("DEATHS_INDIRECT", IntegerType(), True),
    StructField("DAMAGE_PROPERTY", StringType(), True),
    StructField("DAMAGE_CROPS", StringType(), True),
    StructField("SOURCE", StringType(), True),
    StructField("MAGNITUDE", DoubleType(), True),
    StructField("MAGNITUDE_TYPE", StringType(), True),
    StructField("FLOOD_CAUSE", StringType(), True),
    StructField("CATEGORY", StringType(), True),
    StructField("TOR_F_SCALE", StringType(), True),
    StructField("TOR_LENGTH", DoubleType(), True),
    StructField("TOR_WIDTH", IntegerType(), True),
    StructField("TOR_OTHER_WFO", StringType(), True),
    StructField("TOR_OTHER_CZ_STATE", StringType(), True),
    StructField("TOR_OTHER_CZ_FIPS", IntegerType(), True),
    StructField("TOR_OTHER_CZ_NAME", StringType(), True),
    StructField("BEGIN_RANGE", IntegerType(), True),
    StructField("BEGIN_AZIMUTH", StringType(), True),
    StructField("BEGIN_LOCATION", StringType(), True),
    StructField("END_RANGE", IntegerType(), True),
    StructField("END_AZIMUTH", StringType(), True),
    StructField("END_LOCATION", StringType(), True),
    StructField("BEGIN_LAT", DoubleType(), True),
    StructField("BEGIN_LON", DoubleType(), True),
    StructField("END_LAT", DoubleType(), True),
    StructField("END_LON", DoubleType(), True),
    StructField("EPISODE_NARRATIVE", StringType(), True),
    StructField("EVENT_NARRATIVE", StringType(), True),
    StructField("DATA_SOURCE", StringType(), True)
])

# Define paths
local_input_dir = "/home/vagrant/weather2"
hdfs_output_dir = "hdfs:///original_data/weather"

# Get list of CSV files from local directory
import os
import glob

csv_files = glob.glob(os.path.join(local_input_dir, "*.csv"))

if not csv_files:
    print(f"No CSV files found in {local_input_dir}")
    spark.stop()
    exit(1)

print(f"Found {len(csv_files)} CSV file(s) to process:")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")

print(f"\n{'='*60}")
print(f"Reading all CSV files")
print(f"{'='*60}")

try:
    # Read all CSV files from local filesystem
    local_file_path = f"file://{local_input_dir}/*.csv"
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "false") \
        .schema(schema) \
        .csv(local_file_path)
    
    total_records = df.count()
    print(f"Total records read: {total_records}")
    
    # Get distinct year-month combinations
    year_months = df.select("YEAR", "BEGIN_YEARMONTH").distinct().collect()
    
    print(f"\nFound {len(year_months)} distinct year-month combinations")
    print(f"\n{'='*60}")
    print("Processing each month separately")
    print(f"{'='*60}")
    
    total_written = 0
    
    # Process each year-month combination
    for row in sorted(year_months, key=lambda x: (x.YEAR, x.BEGIN_YEARMONTH)):
        year = row.YEAR
        year_month = row.BEGIN_YEARMONTH
        
        # Extract month from BEGIN_YEARMONTH (last 2 digits)
        month = year_month % 100
        
        # Create filename: storm_events_YYYY_MM
        filename = f"storm_events_{year}_{month:02d}"
        output_path = f"{hdfs_output_dir}/{filename}.parquet"
        
        # Filter data for this specific year-month
        month_df = df.filter(
            (col("YEAR") == year) & (col("BEGIN_YEARMONTH") == year_month)
        )
        
        record_count = month_df.count()
        
        print(f"\nProcessing: {filename}")
        print(f"  Records: {record_count}")
        
        # Write to HDFS as single Parquet file
        month_df.coalesce(1).write \
            .mode("overwrite") \
            .parquet(output_path)
        
        print(f"  Written to: {output_path}")
        
        # Verify
        verification_count = spark.read.parquet(output_path).count()
        if record_count == verification_count:
            print(f"  ✓ Verified: {verification_count} records")
            total_written += verification_count
        else:
            print(f"  ✗ WARNING: Verification mismatch!")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total records read: {total_records}")
    print(f"  Total records written: {total_written}")
    print(f"  Files created: {len(year_months)}")
    if total_records == total_written:
        print(f"  ✓ All records successfully written")
    else:
        print(f"  ✗ WARNING: Record count mismatch!")
    print(f"{'='*60}")
        
except Exception as e:
    print(f"✗ Error processing files: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nProcessing complete!")

spark.stop()