from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, current_date, date_format
from pyspark.sql.types import *
import sys
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("PlanesDataBronzeLoader") \
    .config("spark.hadoop.dfs.nameservices", "node1") \
    .enableHiveSupport() \
    .getOrCreate()

# Paths
local_input_dir = "/home/vagrant/planes_update"
bronze_path = "hdfs://node1/original_data/planes"

# Aircraft type code mappings (from NiFi UpdateRecord)
AIRCRAFT_TYPE_CODES = {
    "1": "Glider",
    "2": "Balloon",
    "3": "Blimp/Dirigible",
    "4": "Fixed wing single engine",
    "5": "Fixed wing multi engine",
    "6": "Rotorcraft",
    "7": "Weight-shift-control",
    "8": "Powered Parachute",
    "9": "Gyroplane",
    "H": "Hybrid Lift",
    "O": "Other"
}

# Engine type code mappings (from NiFi UpdateRecord)
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


def get_local_files(directory, pattern="MASTER"):
    """
    Get list of files matching pattern from local directory
    """
    import glob
    
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return []
    
    file_pattern = os.path.join(directory, f"{pattern}*")
    files = glob.glob(file_pattern)
    
    return files


def add_aircraft_type_column(df):
    """
    Add aircraft_type column based on TYPE AIRCRAFT code
    Replicates NiFi UpdateRecord logic
    """
    aircraft_type_expr = when(col("TYPE AIRCRAFT").isNull(), lit("Unknown"))
    
    for code, name in AIRCRAFT_TYPE_CODES.items():
        aircraft_type_expr = aircraft_type_expr.when(
            col("TYPE AIRCRAFT") == code, lit(name)
        )
    
    aircraft_type_expr = aircraft_type_expr.otherwise(lit("Unknown"))
    
    return df.withColumn("aircraft_type", aircraft_type_expr)


def add_engine_type_column(df):
    """
    Add engine_type column based on TYPE ENGINE code
    Replicates NiFi UpdateRecord logic
    """
    engine_type_expr = when(col("TYPE ENGINE").isNull(), lit("Invalid"))
    
    for code, name in ENGINE_TYPE_CODES.items():
        engine_type_expr = engine_type_expr.when(
            col("TYPE ENGINE") == code, lit(name)
        )
    
    engine_type_expr = engine_type_expr.otherwise(lit("Invalid"))
    
    return df.withColumn("engine_type", engine_type_expr)


def process_planes_file(file_path):
    """
    Process a single planes CSV file
    """
    print(f"Processing file: {file_path}")
    
    try:
        # Add file:// prefix for local files
        local_file_uri = f"file://{file_path}"
        print(f"Reading from: {local_file_uri}")
        
        # Read CSV file
        df = spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .option("delimiter", ",") \
            .option("quote", '"') \
            .option("escape", "\\") \
            .csv(local_file_uri)
        
        print(f"Records read: {df.count()}")
        
        # Add aircraft_type and engine_type columns
        df = add_aircraft_type_column(df)
        df = add_engine_type_column(df)
        
        return df
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def clean_column_names(df):
    """
    Replace spaces and special characters in column names for Parquet compatibility
    """
    for col_name in df.columns:
        # Replace spaces with underscores, remove other invalid characters
        clean_name = col_name.replace(" ", "_").replace(",", "").replace(";", "").replace("{", "").replace("}", "").replace("(", "").replace(")", "").replace("\n", "").replace("\t", "").replace("=", "")
        if clean_name != col_name:
            df = df.withColumnRenamed(col_name, clean_name)
            print(f"Renamed column: '{col_name}' -> '{clean_name}'")
    return df


def write_to_bronze(df, output_path):
    """
    Write DataFrame to bronze layer as Parquet
    """
    if df is None or df.count() == 0:
        print("No data to write")
        return False
    
    try:
        # Clean column names for Parquet compatibility
        print("\nCleaning column names...")
        df = clean_column_names(df)
        
        # Generate filename with current date
        current_date_str = date_format(current_date(), "yyyy-MM-dd").alias("date")
        filename = f"plane_registry_{current_date_str}"
        
        # Full output path
        full_output_path = f"{output_path}/plane_registry_{spark.sql('SELECT current_date()').collect()[0][0]}.parquet"
        
        print(f"Writing to: {full_output_path}")
        
        # Write as Parquet with replace mode
        df.write \
            .mode("overwrite") \
            .parquet(full_output_path)
        
        print(f"Successfully wrote {df.count()} records to {full_output_path}")
        
        # Verify the file was written
        verification = spark.read.parquet(full_output_path)
        verified_count = verification.count()
        print(f"Verification: {verified_count} records in HDFS")
        
        return True
        
    except Exception as e:
        print(f"Error writing to bronze layer: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_processed_files(files):
    """
    Remove processed files from local directory
    """
    for file_path in files:
        try:
            os.remove(file_path)
            print(f"Removed processed file: {file_path}")
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")


def main():
    """
    Main processing function
    """
    print("=" * 80)
    print("Planes Data Bronze Layer Loader")
    print("=" * 80)
    
    # Get list of MASTER files
    local_files = get_local_files(local_input_dir, "MASTER")
    
    if not local_files:
        print(f"No MASTER files found in {local_input_dir}")
        return
    
    print(f"Found {len(local_files)} file(s) to process:")
    for f in local_files:
        print(f"  - {f}")
    
    # Process all files
    all_data = None
    
    for file_path in local_files:
        df = process_planes_file(file_path)
        
        if df is not None:
            if all_data is None:
                all_data = df
            else:
                all_data = all_data.union(df)
    
    if all_data is None:
        print("No data processed successfully")
        return
    
    # Show sample data
    print("\nSample of processed data:")
    all_data.show(10, truncate=False)
    
    print("\nSchema:")
    all_data.printSchema()
    
    # Write to bronze layer
    success = write_to_bronze(all_data, bronze_path)
    
    if success:
        # Cleanup processed files
        cleanup_processed_files(local_files)
        print("\nProcessing complete!")
    else:
        print("\nProcessing failed - files not removed")
    
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()