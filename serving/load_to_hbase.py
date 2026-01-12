import os
import happybase
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ======================================================
# CONFIGURATION
# ======================================================

SERVING_DB = "serving"

# HBase Thrift connection for HappyBase
HBASE_HOST = os.getenv("HBASE_THRIFT_HOST", "node1")
HBASE_PORT = int(os.getenv("HBASE_THRIFT_PORT", "9090"))

COLUMN_FAMILY = "m"

HBASE_TABLES = {
    "delay_airport_month": "serving:delay_airport_month",
    "cancel_airport_month": "serving:cancel_airport_month",
    "delay_carrier_month": "serving:delay_carrier_month",
    "top10_airports_delay_month": "serving:top10_airports_delay_month",
    "delay_weather_region_month": "serving:delay_weather_region_month",
}

COL_YEAR = "year_partition"
COL_MONTH = "month_partition"


# ======================================================
# HBASE HELPERS
# ======================================================

def ensure_table(connection, table_name):
    """Create an HBase table if it does not exist."""
    existing = {
        t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else t
        for t in connection.tables()
    }
    if table_name not in existing:
        connection.create_table(table_name, {COLUMN_FAMILY: dict()})
        print(f"[INFO] Created HBase table: {table_name}")
    else:
        print(f"[INFO] HBase table exists: {table_name}")


def write_df_to_hbase(df, table, rowkey_col, mappings, batch_size=5000):
    """
    Write a Spark DataFrame to HBase using HappyBase batch.
    mappings: list of (df_column, hbase_qualifier)
    """

    def write_partition(rows):
        with table.batch(batch_size=batch_size) as b:
            for r in rows:
                rk = r[rowkey_col]
                if rk is None:
                    continue

                data = {}
                for c, q in mappings:
                    val = r[c]
                    if val is None:
                        continue
                    data[f"{COLUMN_FAMILY}:{q}".encode("utf-8")] = str(val).encode("utf-8")

                if data:
                    b.put(rk.encode("utf-8"), data)

    df.select([rowkey_col] + [c for c, _ in mappings]).rdd.foreachPartition(write_partition)


def hive_table_exists(spark, db: str, tbl: str) -> bool:
    """Check Hive table existence."""
    return spark._jsparkSession.catalog().tableExists(db, tbl)


# ======================================================
# MAIN
# ======================================================

def main():
    spark = (
        SparkSession.builder
        .appName("LoadServingViewsToHBase")
        .enableHiveSupport()
        .getOrCreate()
    )

    print(f"[INFO] Connecting to HBase Thrift: {HBASE_HOST}:{HBASE_PORT}")
    conn = happybase.Connection(HBASE_HOST, HBASE_PORT)
    conn.open()

    # --------------------------------------------------
    # 1) delay_airport_month
    # --------------------------------------------------
    tname = "delay_airport_month"
    hive_tbl = f"{SERVING_DB}.{tname}"
    if not hive_table_exists(spark, SERVING_DB, tname):
        raise RuntimeError(f"Missing Hive staging table: {hive_tbl} (run build_batch_views.py first)")

    ensure_table(conn, HBASE_TABLES[tname])
    table = conn.table(HBASE_TABLES[tname])

    df = (
        spark.table(hive_tbl)
        .withColumn(
            "rowkey",
            F.concat_ws(
                "#",
                F.col("DepartureAirportID").cast("string"),
                F.format_string("%04d%02d", F.col(COL_YEAR), F.col(COL_MONTH))
            )
        )
    )

    write_df_to_hbase(
        df, table, "rowkey",
        [
            ("avg_dep_delay", "avg_dep_delay"),
            ("avg_arr_delay", "avg_arr_delay"),
            ("flights_cnt", "flights_cnt"),
        ]
    )
    print("[OK] Loaded delay_airport_month.")

    # --------------------------------------------------
    # 2) cancel_airport_month
    # --------------------------------------------------
    tname = "cancel_airport_month"
    hive_tbl = f"{SERVING_DB}.{tname}"
    ensure_table(conn, HBASE_TABLES[tname])
    table = conn.table(HBASE_TABLES[tname])

    df = (
        spark.table(hive_tbl)
        .withColumn(
            "rowkey",
            F.concat_ws(
                "#",
                F.col("DepartureAirportID").cast("string"),
                F.format_string("%04d%02d", F.col(COL_YEAR), F.col(COL_MONTH))
            )
        )
    )

    write_df_to_hbase(
        df, table, "rowkey",
        [
            ("cancel_pct", "cancel_pct"),
            ("cancel_cnt", "cancel_cnt"),
            ("flights_cnt", "flights_cnt"),
        ]
    )
    print("[OK] Loaded cancel_airport_month.")

    # --------------------------------------------------
    # 5) delay_carrier_month
    # --------------------------------------------------
    tname = "delay_carrier_month"
    hive_tbl = f"{SERVING_DB}.{tname}"
    ensure_table(conn, HBASE_TABLES[tname])
    table = conn.table(HBASE_TABLES[tname])

    df = (
        spark.table(hive_tbl)
        .withColumn(
            "rowkey",
            F.concat_ws(
                "#",
                F.col("CarrierCode").cast("string"),
                F.format_string("%04d%02d", F.col(COL_YEAR), F.col(COL_MONTH))
            )
        )
    )

    write_df_to_hbase(
        df, table, "rowkey",
        [
            ("avg_dep_delay", "avg_dep_delay"),
            ("avg_arr_delay", "avg_arr_delay"),
            ("flights_cnt", "flights_cnt"),
        ]
    )
    print("[OK] Loaded delay_carrier_month.")

    # --------------------------------------------------
    # 3) top10_airports_delay_month
    # --------------------------------------------------
    tname = "top10_airports_delay_month"
    hive_tbl = f"{SERVING_DB}.{tname}"
    ensure_table(conn, HBASE_TABLES[tname])
    table = conn.table(HBASE_TABLES[tname])

    df = (
        spark.table(hive_tbl)
        .withColumn("rowkey", F.format_string("%04d%02d", F.col(COL_YEAR), F.col(COL_MONTH)))
    )

    # Store all rank columns as qualifiers with identical names
    rank_cols = [c for c in df.columns if c.startswith("rank")]
    mappings = [(c, c) for c in rank_cols]

    write_df_to_hbase(df, table, "rowkey", mappings)
    print("[OK] Loaded top10_airports_delay_month.")

    # --------------------------------------------------
    # 4) delay_weather_region_month (optional)
    # --------------------------------------------------
    tname = "delay_weather_region_month"
    hive_tbl = f"{SERVING_DB}.{tname}"
    if hive_table_exists(spark, SERVING_DB, tname):
        ensure_table(conn, HBASE_TABLES[tname])
        table = conn.table(HBASE_TABLES[tname])

        df = (
            spark.table(hive_tbl)
            .withColumn(
                "rowkey",
                F.concat_ws(
                    "#",
                    F.col("Region").cast("string"),
                    F.col("EventType").cast("string"),
                    F.format_string("%04d%02d", F.col(COL_YEAR), F.col(COL_MONTH))
                )
            )
        )

        write_df_to_hbase(
            df, table, "rowkey",
            [
                ("avg_dep_delay", "avg_dep_delay"),
                ("avg_arr_delay", "avg_arr_delay"),
                ("flights_cnt", "flights_cnt"),
            ]
        )
        print("[OK] Loaded delay_weather_region_month.")
    else:
        print("[WARN] serving.delay_weather_region_month not found – skipping weather load.")

    print("[DONE] All available serving views loaded into HBase.")
    conn.close()
    spark.stop()


if __name__ == "__main__":
    main()
