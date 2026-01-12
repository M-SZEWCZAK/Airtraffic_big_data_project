import os
import happybase
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# CONFIG
# ======================================================

HBASE_HOST = os.getenv("HBASE_THRIFT_HOST", "node1")
HBASE_PORT = int(os.getenv("HBASE_THRIFT_PORT", "9090"))

CF = b"m:"  # column family prefix in HBase

# HBase tables
T_DELAY_AIRPORT_MONTH = "serving:delay_airport_month"
T_CANCEL_AIRPORT_MONTH = "serving:cancel_airport_month"
T_DELAY_CARRIER_MONTH = "serving:delay_carrier_month"
T_TOP10_MONTH = "serving:top10_airports_delay_month"
T_WEATHER = "serving:delay_weather_region_month"  # optional

# Demo parameters
AIRPORT_ID = "12478"
CARRIER_CODE = "AA"
YEAR = 2024
MONTH_FROM = 1
MONTH_TO = 12

# demo Top10 for a specific month
TOP10_YEAR = 2024
TOP10_MONTH = 1


# ======================================================
# HELPERS
# ======================================================

def rk_airport_month(airport_id: str, year: int, month: int) -> bytes:
    return f"{airport_id}#{year:04d}{month:02d}".encode("utf-8")

def rk_carrier_month(carrier: str, year: int, month: int) -> bytes:
    return f"{carrier}#{year:04d}{month:02d}".encode("utf-8")

def rk_month(year: int, month: int) -> bytes:
    return f"{year:04d}{month:02d}".encode("utf-8")

def decode_row(row: dict) -> dict:
    """
    Decode HappyBase row dict: {b'm:qualifier': b'value'} -> {'qualifier': 'value'}
    """
    out = {}
    for k, v in row.items():
        # k example: b'm:avg_dep_delay'
        if k.startswith(CF):
            kk = k[len(CF):].decode("utf-8")
        else:
            kk = k.decode("utf-8")
        out[kk] = v.decode("utf-8")
    return out

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def safe_int(x):
    try:
        return int(float(x))
    except:
        return None


# ======================================================
# READERS (HBase -> pandas)
# ======================================================

def fetch_airport_month_series(table, airport_id: str, year: int, m_from: int, m_to: int) -> pd.DataFrame:
    """
    Fetch monthly metrics for one airport using random reads (rowkey GET).
    """
    rows = []
    for m in range(m_from, m_to + 1):
        rk = rk_airport_month(airport_id, year, m)
        row = table.row(rk)
        if not row:
            continue
        d = decode_row(row)
        d["year"] = year
        d["month"] = m
        rows.append(d)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Convert types
    if "avg_dep_delay" in df.columns:
        df["avg_dep_delay"] = df["avg_dep_delay"].apply(safe_float)
    if "avg_arr_delay" in df.columns:
        df["avg_arr_delay"] = df["avg_arr_delay"].apply(safe_float)
    if "flights_cnt" in df.columns:
        df["flights_cnt"] = df["flights_cnt"].apply(safe_int)

    return df.sort_values(["year", "month"])


def fetch_carrier_month_series(table, carrier: str, year: int, m_from: int, m_to: int) -> pd.DataFrame:
    """
    Fetch monthly metrics for one carrier using random reads.
    """
    rows = []
    for m in range(m_from, m_to + 1):
        rk = rk_carrier_month(carrier, year, m)
        row = table.row(rk)
        if not row:
            continue
        d = decode_row(row)
        d["year"] = year
        d["month"] = m
        rows.append(d)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "avg_dep_delay" in df.columns:
        df["avg_dep_delay"] = df["avg_dep_delay"].apply(safe_float)
    if "avg_arr_delay" in df.columns:
        df["avg_arr_delay"] = df["avg_arr_delay"].apply(safe_float)
    if "flights_cnt" in df.columns:
        df["flights_cnt"] = df["flights_cnt"].apply(safe_int)

    return df.sort_values(["year", "month"])


def fetch_top10_for_month(table, year: int, month: int) -> pd.DataFrame:
    """
    Fetch Top10 airports for a given month (one rowkey = YYYYMM).
    """
    row = table.row(rk_month(year, month))
    if not row:
        return pd.DataFrame()

    d = decode_row(row)

    # Build a table with rank, airport_id, delay
    out = []
    for i in range(1, 11):
        idx = f"{i:02d}"
        a = d.get(f"rank{idx}_airport")
        delay = d.get(f"rank{idx}_delay")
        if a is None and delay is None:
            continue
        out.append({
            "rank": i,
            "airport_id": a,
            "avg_dep_delay": safe_float(delay),
        })

    return pd.DataFrame(out)


# ======================================================
# CHARTS
# ======================================================

def plot_airport_delays(df: pd.DataFrame, airport_id: str):
    plt.figure()
    plt.plot(df["month"], df["avg_dep_delay"], marker="o")
    plt.title(f"Average departure delay by month (airport {airport_id}, year {df['year'].iloc[0]})")
    plt.xlabel("Month")
    plt.ylabel("Avg departure delay [min]")
    plt.grid(True)
    plt.show()


def plot_airport_cancellations(df: pd.DataFrame, airport_id: str):
    if "cancel_pct" not in df.columns:
        return

    df["cancel_pct"] = df["cancel_pct"].apply(safe_float)

    plt.figure()
    plt.plot(df["month"], df["cancel_pct"], marker="o")
    plt.title(f"Cancellation percentage by month (airport {airport_id}, year {df['year'].iloc[0]})")
    plt.xlabel("Month")
    plt.ylabel("Cancellation %")
    plt.grid(True)
    plt.show()


def plot_carrier_delays(df: pd.DataFrame, carrier: str):
    plt.figure()
    plt.plot(df["month"], df["avg_dep_delay"], marker="o")
    plt.title(f"Average departure delay by month (carrier {carrier}, year {df['year'].iloc[0]})")
    plt.xlabel("Month")
    plt.ylabel("Avg departure delay [min]")
    plt.grid(True)
    plt.show()


def plot_top10(df: pd.DataFrame, year: int, month: int):
    plt.figure()
    plt.bar(df["airport_id"], df["avg_dep_delay"])
    plt.title(f"Top 10 airports by avg departure delay ({year}-{month:02d})")
    plt.xlabel("AirportID")
    plt.ylabel("Avg departure delay [min]")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ======================================================
# MAIN
# ======================================================

def main():
    print(f"Connecting to HBase Thrift at {HBASE_HOST}:{HBASE_PORT} ...")
    conn = happybase.Connection(HBASE_HOST, HBASE_PORT)
    conn.open()

    # --- 1) Airport delay chart
    t_delay = conn.table(T_DELAY_AIRPORT_MONTH)
    df_delay = fetch_airport_month_series(t_delay, AIRPORT_ID, YEAR, MONTH_FROM, MONTH_TO)

    if df_delay.empty:
        print(f"No delay data found for airport={AIRPORT_ID} year={YEAR}")
    else:
        print(df_delay.head())
        plot_airport_delays(df_delay, AIRPORT_ID)

    # --- 2) Airport cancellation chart
    t_cancel = conn.table(T_CANCEL_AIRPORT_MONTH)
    df_cancel = fetch_airport_month_series(t_cancel, AIRPORT_ID, YEAR, MONTH_FROM, MONTH_TO)

    if df_cancel.empty:
        print(f"No cancellation data found for airport={AIRPORT_ID} year={YEAR}")
    else:
        plot_airport_cancellations(df_cancel, AIRPORT_ID)

    # --- 3) Carrier delay chart
    t_carrier = conn.table(T_DELAY_CARRIER_MONTH)
    df_carrier = fetch_carrier_month_series(t_carrier, CARRIER_CODE, YEAR, MONTH_FROM, MONTH_TO)

    if df_carrier.empty:
        print(f"No carrier data found for carrier={CARRIER_CODE} year={YEAR}")
    else:
        plot_carrier_delays(df_carrier, CARRIER_CODE)

    # --- 4) Top10 chart for one month
    t_top10 = conn.table(T_TOP10_MONTH)
    df_top10 = fetch_top10_for_month(t_top10, TOP10_YEAR, TOP10_MONTH)

    if df_top10.empty:
        print(f"No Top10 data found for {TOP10_YEAR}-{TOP10_MONTH:02d}")
    else:
        print(df_top10)
        plot_top10(df_top10, TOP10_YEAR, TOP10_MONTH)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
