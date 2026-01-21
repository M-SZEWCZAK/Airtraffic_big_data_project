import os
import happybase
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


# # --- GUI (scrollable window) ---
# import tkinter as tk
# from tkinter import ttk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
T_WEATHER_CANCEL = "serving:cancel_weather_region_month"  # optional
T_AIRCRAFT_AGE = "serving:aircraft_age_bucket_carrier_year"

# # Demo parameters
# AIRPORT_ID = "12478"
# CARRIER_CODE = "AA"
# YEAR = 2024
# MONTH_FROM = 1
# MONTH_TO = 12
#
# # demo Top10 for a specific month
# TOP10_YEAR = 2024
# TOP10_MONTH = 1


_AIRPORT_LOOKUP = None

def load_airport_lookup(csv_path=None):
    """
    Zwraca dict: {AIRPORT_ID (int) -> AIRPORT (str)} wczytany z L_AIRPORT_ID.csv.
    """
    global _AIRPORT_LOOKUP
    if _AIRPORT_LOOKUP is not None:
        return _AIRPORT_LOOKUP

    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "L_AIRPORT_ID.csv"
    else:
        csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    df["AIRPORT_ID"] = df["AIRPORT_ID"].astype(int)
    df["AIRPORT"] = df["AIRPORT"].astype(str)

    _AIRPORT_LOOKUP = dict(zip(df["AIRPORT_ID"], df["AIRPORT"]))
    return _AIRPORT_LOOKUP


# ======================================================
# PARSER
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser("HBase Serving Layer Demo")

    parser.add_argument("--airport-id", default="12478")
    parser.add_argument("--carrier", default="AA")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month-from", type=int, default=1)
    parser.add_argument("--month-to", type=int, default=1)
    parser.add_argument("--top10-month", type=int, default=1)
    parser.add_argument("--top10-year", type=int, default=2025)
    parser.add_argument("--region", default="California")

    parser.add_argument("--out-dir", default="plots")

    return parser.parse_args()


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

def save_fig(fig, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


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

def fetch_weather_heatmap(
    table,
    region: str,
    year: int,
    m_from: int,
    m_to: int,
    value_col: str = "avg_dep_delay"
) -> pd.DataFrame:
    """
    Generic fetcher for weather x region x month heatmaps.

    value_col examples:
      - 'avg_dep_delay'  (delay heatmap)
      - 'cancel_rate'    (cancel heatmap)

    Rowkey format:
      Region#EventType#YYYYMM
    """
    prefix = f"{region}#".encode("utf-8")
    rows = []

    for rk, row in table.scan(row_prefix=prefix):
        rk_str = rk.decode("utf-8", errors="ignore")
        parts = rk_str.split("#", 2)
        if len(parts) != 3:
            continue

        _region, event_type, yyyymm = parts
        try:
            y = int(yyyymm[:4])
            m = int(yyyymm[4:6])
        except:
            continue

        if y != year or m < m_from or m > m_to:
            continue

        d = decode_row(row)
        rows.append({
            "region": _region,
            "EventType": event_type,
            "year": y,
            "month": m,
            value_col: safe_float(d.get(value_col)),
            "flights_cnt": safe_int(d.get("flights_cnt")),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values(["month", "EventType"])


def fetch_aircraft_age_bucket(table, year: int, top_carriers: int = 5) -> pd.DataFrame:
    """
    Scan HBase rows for a given year:
      rowkey = YYYY#bucket#carrier
    returns columns: aircraft_age_bucket, carrier, avg_dep_delay, avg_arr_delay, flights_cnt
    """
    prefix = f"{year:04d}#".encode("utf-8")

    rows = []
    for rk, row in table.scan(row_prefix=prefix):
        rk_str = rk.decode("utf-8", errors="ignore")
        parts = rk_str.split("#")
        if len(parts) != 3:
            continue

        y_str, bucket, carrier = parts
        d = decode_row(row)

        rows.append({
            "year": int(y_str),
            "aircraft_age_bucket": bucket,
            "carrier": carrier,
            "avg_dep_delay": safe_float(d.get("avg_dep_delay")),
            "avg_arr_delay": safe_float(d.get("avg_arr_delay")),
            "flights_cnt": safe_int(d.get("flights_cnt")),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # choose top carriers by volume (optional)
    if top_carriers is not None and top_carriers > 0:
        top = (
            df.groupby("carrier")["flights_cnt"]
            .sum()
            .sort_values(ascending=False)
            .head(top_carriers)
            .index
        )
        df = df[df["carrier"].isin(top)]

    order = ["0-5", "6-10", "11-20", ">20", "unknown"]
    df["aircraft_age_bucket"] = pd.Categorical(df["aircraft_age_bucket"], categories=order, ordered=True)

    return df.sort_values(["aircraft_age_bucket", "carrier"])


# ======================================================
# DASHBOARD / PLOTS
# ======================================================

def save_charts(
    df_delay: pd.DataFrame,
    df_cancel: pd.DataFrame,
    df_carrier: pd.DataFrame,
    df_top10: pd.DataFrame,
    airport_id: str,
    carrier: str,
    year: int,
    month_from: int,
    month_to: int,
    top10_year: int,
    top10_month: int,
    df_weather_delay: pd.DataFrame,
    df_weather_cancel: pd.DataFrame,
    region: str,
    df_age: pd.DataFrame,
    out_dir: str,
):
    lookup = load_airport_lookup()
    airport_code = lookup.get(int(airport_id), airport_id)
    months_range = list(range(month_from, month_to + 1))

    # 1) Airport delays
    fig, ax = plt.subplots(figsize=(8, 4))
    if not df_delay.empty and "avg_dep_delay" in df_delay.columns:
        df_d = df_delay.copy()
        df_d["month"] = df_d["month"].astype(int)
        s = df_d.set_index("month")["avg_dep_delay"].reindex(months_range)

        ax.plot(months_range, s.values, marker="o")
        ax.set_xticks(months_range)
        ax.set_xlim(month_from - 0.5, month_to + 0.5)
        ax.set_xlabel("Month")
        ax.set_ylabel("Delay [min]")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title(f"Avg dep delay (airport {airport_id}, {year})")
    save_fig(fig, out_dir, f"01_airport_{airport_code}_{year}_dep_delay.png")

    # 2) Airport cancellations
    fig, ax = plt.subplots(figsize=(8, 4))
    if not df_cancel.empty and "cancel_pct" in df_cancel.columns:
        df_c = df_cancel.copy()
        df_c["month"] = df_c["month"].astype(int)
        df_c["cancel_pct"] = df_c["cancel_pct"].apply(safe_float)

        s = df_c.set_index("month")["cancel_pct"].reindex(months_range)

        ax.plot(months_range, s.values, marker="o")
        ax.set_xticks(months_range)
        ax.set_xlim(month_from - 0.5, month_to + 0.5)
        ax.set_xlabel("Month")
        ax.set_ylabel("Cancel %")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title(f"Cancel % (airport {airport_id}, {year})")
    save_fig(fig, out_dir, f"02_airport_{airport_code}_{year}_cancel_pct.png")

    # 3) Carrier delays
    fig, ax = plt.subplots(figsize=(8, 4))
    if not df_carrier.empty and "avg_dep_delay" in df_carrier.columns:
        df_cd = df_carrier.copy()
        df_cd["month"] = df_cd["month"].astype(int)

        s = df_cd.set_index("month")["avg_dep_delay"].reindex(months_range)

        ax.plot(months_range, s.values, marker="o")
        ax.set_xticks(months_range)
        ax.set_xlim(month_from - 0.5, month_to + 0.5)
        ax.set_xlabel("Month")
        ax.set_ylabel("Delay [min]")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title(f"Avg dep delay (carrier {carrier}, {year})")
    save_fig(fig, out_dir, f"03_carrier_{carrier}_{year}_dep_delay.png")

    # 4) Top10 airports
    fig, ax = plt.subplots(figsize=(10, 5))
    if not df_top10.empty and "airport_id" in df_top10.columns and "avg_dep_delay" in df_top10.columns:
        lookup = load_airport_lookup()

        df_top10["airport_id"] = df_top10["airport_id"].astype(int)
        df_top10["AirportCode"] = df_top10["airport_id"].map(lookup)
        df_top10["AirportCode"] = df_top10["AirportCode"].fillna(df_top10["airport_id"].astype(str))

        ax.bar(df_top10["AirportCode"], df_top10["avg_dep_delay"])

        ax.set_xlabel("Airport")
        ax.set_ylabel("Delay [min]")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title(f"Top10 airports avg dep delay ({top10_year}-{top10_month:02d})")
    save_fig(fig, out_dir, f"04_top10_{top10_year}-{top10_month:02d}_dep_delay.png")

    # 5) Weather vs delay heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    if df_weather_delay is not None and not df_weather_delay.empty:
        pivot = df_weather_delay.pivot_table(
            index="month", columns="EventType", values="avg_dep_delay", aggfunc="mean"
        )
        pivot = pivot.reindex(months_range)

        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_title(f"Avg dep delay vs weather ({region}, {year})")
        ax.set_xlabel("EventType")
        ax.set_ylabel("Month")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Delay [min]")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(f"Avg dep delay vs weather ({region}, {year})")
    save_fig(fig, out_dir, f"05_weather_{region}_{year}_delay_heatmap.png")

    # 6) Weather vs cancel rate heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    if df_weather_cancel is not None and not df_weather_cancel.empty:
        pivot = df_weather_cancel.pivot_table(
            index="month", columns="EventType", values="cancel_rate", aggfunc="mean"
        )
        pivot = pivot.reindex(months_range)

        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_title(f"Cancel rate vs weather ({region}, {year})")
        ax.set_xlabel("EventType")
        ax.set_ylabel("Month")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cancel rate")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title(f"Cancel rate vs weather ({region}, {year})")
    save_fig(fig, out_dir, f"06_weather_{region}_{year}_cancel_heatmap.png")

    # 7) Aircraft age bucket vs avg dep delay
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Aircraft age bucket vs avg dep delay ({year})")
    if df_age is not None and not df_age.empty:
        pivot = df_age.pivot_table(
            index="aircraft_age_bucket", columns="carrier", values="avg_dep_delay", aggfunc="mean"
        )
        pivot.plot(kind="bar", ax=ax)
        ax.set_xlabel("Aircraft age bucket [years]")
        ax.set_ylabel("Avg dep delay [min]")
        for label in ax.get_xticklabels():
            label.set_rotation(0)
        ax.grid(True, axis="y")
        ax.legend(title="Carrier", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    save_fig(fig, out_dir, f"07_aircraft_age_{year}_dep_delay.png")


# def show_dashboard(
#     df_delay: pd.DataFrame,
#     df_cancel: pd.DataFrame,
#     df_carrier: pd.DataFrame,
#     df_top10: pd.DataFrame,
#     airport_id: str,
#     carrier: str,
#     year: int,
#     month_from: int,
#     month_to: int,
#     top10_year: int,
#     top10_month: int,
#     df_weather_delay: pd.DataFrame,
#     df_weather_cancel: pd.DataFrame,
#     region: str,
#     df_age: pd.DataFrame,
# ):
#     fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))
#     ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()
#
#     # 1) Airport delays
#     if not df_delay.empty and "avg_dep_delay" in df_delay.columns:
#         ax1.plot(df_delay["month"], df_delay["avg_dep_delay"], marker="o")
#         ax1.set_title(f"Avg dep delay (airport {airport_id}, {year})")
#         ax1.set_xlabel("Month")
#         ax1.set_ylabel("Delay [min]")
#         ax1.grid(True)
#     else:
#         ax1.set_title(f"Avg dep delay (airport {airport_id}, {year})")
#         ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)
#         ax1.set_axis_off()
#
#     # 2) Airport cancellations
#     if not df_cancel.empty and "cancel_pct" in df_cancel.columns:
#         df_cancel = df_cancel.copy()
#         df_cancel["cancel_pct"] = df_cancel["cancel_pct"].apply(safe_float)
#
#         ax2.plot(df_cancel["month"], df_cancel["cancel_pct"], marker="o")
#         ax2.set_title(f"Cancel % (airport {airport_id}, {year})")
#         ax2.set_xlabel("Month")
#         ax2.set_ylabel("Cancel %")
#         ax2.grid(True)
#     else:
#         ax2.set_title(f"Cancel % (airport {airport_id}, {year})")
#         ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
#         ax2.set_axis_off()
#
#     # 3) Carrier delays
#     if not df_carrier.empty and "avg_dep_delay" in df_carrier.columns:
#         ax3.plot(df_carrier["month"], df_carrier["avg_dep_delay"], marker="o")
#         ax3.set_title(f"Avg dep delay (carrier {carrier}, {year})")
#         ax3.set_xlabel("Month")
#         ax3.set_ylabel("Delay [min]")
#         ax3.grid(True)
#     else:
#         ax3.set_title(f"Avg dep delay (carrier {carrier}, {year})")
#         ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)
#         ax3.set_axis_off()
#
#     # 4) Top10 airports
#     if not df_top10.empty and "airport_id" in df_top10.columns and "avg_dep_delay" in df_top10.columns:
#         ax4.bar(df_top10["airport_id"], df_top10["avg_dep_delay"])
#         ax4.set_title(f"Top10 airports avg dep delay ({top10_year}-{top10_month:02d})")
#         ax4.set_xlabel("AirportID")
#         ax4.set_ylabel("Delay [min]")
#         for label in ax4.get_xticklabels():
#             label.set_rotation(45)
#             label.set_ha("right")
#         ax4.grid(False)
#     else:
#         ax4.set_title(f"Top10 airports avg dep delay ({top10_year}-{top10_month:02d})")
#         ax4.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax4.transAxes)
#         ax4.set_axis_off()
#
#     # 5) Weather vs delay (heatmap: month x EventType)
#     # Expect df_weather columns: month, EventType, avg_dep_delay
#     if df_weather_delay is not None and not df_weather_delay.empty:
#         pivot = df_weather_delay.pivot_table(
#             index="month",
#             columns="EventType",
#             values="avg_dep_delay",
#             aggfunc="mean"
#         )
#
#         im = ax5.imshow(pivot.values, aspect="auto")  # default colormap
#         ax5.set_title(f"Avg dep delay vs weather ({region}, {year})")
#         ax5.set_xlabel("EventType")
#         ax5.set_ylabel("Month")
#
#         ax5.set_yticks(range(len(pivot.index)))
#         ax5.set_yticklabels(pivot.index.tolist())
#
#         ax5.set_xticks(range(len(pivot.columns)))
#         ax5.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right")
#
#         fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04, label="Delay [min]")
#     else:
#         ax5.set_title(f"Avg dep delay vs weather ({region}, {year})")
#         ax5.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax5.transAxes)
#         ax5.set_axis_off()
#
#     # 6) Weather vs cancel rate (heatmap: month x EventType)
#     # Expect df_weather_cancel columns: month, EventType, cancel_rate
#     if df_weather_cancel is not None and not df_weather_cancel.empty:
#         pivot = df_weather_cancel.pivot_table(
#             index="month",
#             columns="EventType",
#             values="cancel_rate",
#             aggfunc="mean"
#         )
#
#         im = ax6.imshow(pivot.values, aspect="auto")  # default colormap
#         ax6.set_title(f"Cancel rate vs weather ({region}, {year})")
#         ax6.set_xlabel("EventType")
#         ax6.set_ylabel("Month")
#
#         ax6.set_yticks(range(len(pivot.index)))
#         ax6.set_yticklabels(pivot.index.tolist())
#
#         ax6.set_xticks(range(len(pivot.columns)))
#         ax6.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right")
#
#         fig.colorbar(im, ax=ax6, fraction=0.046, pad=0.04, label="Cancel rate")
#     else:
#         ax6.set_title(f"Cancel rate vs weather ({region}, {year})")
#         ax6.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax6.transAxes)
#         ax6.set_axis_off()
#
#     # 7) Aircraft age bucket vs avg dep delay (grouped bar)
#     ax7.set_title(f"Aircraft age bucket vs avg dep delay ({year})")
#     if df_age is not None and not df_age.empty:
#         pivot = df_age.pivot_table(
#             index="aircraft_age_bucket",
#             columns="carrier",
#             values="avg_dep_delay",
#             aggfunc="mean"
#         )
#
#         pivot.plot(kind="bar", ax=ax7)
#         ax7.set_xlabel("Aircraft age bucket [years]")
#         ax7.set_ylabel("Avg dep delay [min]")
#         for label in ax7.get_xticklabels():
#             label.set_rotation(0)
#         ax7.grid(True, axis="y")
#         ax7.legend(title="Carrier", fontsize=8)
#     else:
#         ax7.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax7.transAxes)
#         ax7.set_axis_off()
#
#     # 8) empty plot
#     ax8.set_axis_off()
#
#
#     fig.suptitle(
#         f"HBase Serving Demo | airport={airport_id} carrier={carrier} "
#         f"range={year}-{month_from:02d}..{month_to:02d} | top10={top10_year}-{top10_month:02d}",
#         fontsize=12
#     )
#     fig.tight_layout(rect=[0, 0, 1, 0.95])
#
#     out_dir = "plots"
#     os.makedirs(out_dir, exist_ok=True)
#
#     fname = (
#         f"dashboard_airport_{airport_id}_carrier_{carrier}_"
#         f"{year}_{month_from:02d}-{month_to:02d}_"
#         f"top10_{top10_year}-{top10_month:02d}.png"
#     )
#
#     out_path = os.path.join(out_dir, fname)
#
#     fig.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#
#     print(f"[OK] Saved dashboard to {out_path}")
#
#     # root = tk.Tk()
#     # root.title("HBase Serving Layer Demo - Dashboard")
#     #
#     # container = ttk.Frame(root)
#     # container.pack(fill="both", expand=True)
#     #
#     # canvas = tk.Canvas(container)
#     # vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
#     # canvas.configure(yscrollcommand=vscroll.set)
#     #
#     # vscroll.pack(side="right", fill="y")
#     # canvas.pack(side="left", fill="both", expand=True)
#     #
#     # inner = ttk.Frame(canvas)
#     # inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")
#     #
#     # mpl_canvas = FigureCanvasTkAgg(fig, master=inner)
#     # mpl_widget = mpl_canvas.get_tk_widget()
#     # mpl_widget.pack(fill="both", expand=True)
#     #
#     # mpl_canvas.draw()
#     #
#     # def _on_configure(event):
#     #     canvas.configure(scrollregion=canvas.bbox("all"))
#     #
#     # inner.bind("<Configure>", _on_configure)
#     #
#     # def _on_canvas_configure(event):
#     #     canvas.itemconfig(inner_id, width=event.width)
#     #
#     # canvas.bind("<Configure>", _on_canvas_configure)
#     #
#     # def _on_mousewheel(event):
#     #     canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
#     #
#     # canvas.bind_all("<MouseWheel>", _on_mousewheel)
#     #
#     # root.minsize(900, 600)
#     # root.mainloop()


# ======================================================
# MAIN
# ======================================================

def main():
    args = parse_args()
    out_dir = args.out_dir

    region = args.region
    airport_id = args.airport_id
    carrier = args.carrier
    year = args.year
    month_from = args.month_from
    month_to = args.month_to
    top10_month = args.top10_month
    top10_year = args.top10_year


    print(f"Connecting to HBase Thrift at {HBASE_HOST}:{HBASE_PORT} ...")
    conn = happybase.Connection(HBASE_HOST, HBASE_PORT)
    conn.open()

    t_delay = conn.table(T_DELAY_AIRPORT_MONTH)
    df_delay = fetch_airport_month_series(t_delay, airport_id, year, month_from, month_to)

    t_cancel = conn.table(T_CANCEL_AIRPORT_MONTH)
    df_cancel = fetch_airport_month_series(t_cancel, airport_id, year, month_from, month_to)

    t_carrier = conn.table(T_DELAY_CARRIER_MONTH)
    df_carrier = fetch_carrier_month_series(t_carrier, carrier, year, month_from, month_to)

    t_top10 = conn.table(T_TOP10_MONTH)
    df_top10 = fetch_top10_for_month(t_top10, top10_year, top10_month)

    df_weather_delay = pd.DataFrame()
    df_weather_cancel = pd.DataFrame()
    df_age = pd.DataFrame()

    try:
        t_age = conn.table(T_AIRCRAFT_AGE)
        df_age = fetch_aircraft_age_bucket(t_age, year, top_carriers=5)
    except Exception as e:
        print(f"[WARN] Aircraft-age chart skipped: {e}")

    # delay heatmap (avg_dep_delay)
    try:
        t_weather_delay = conn.table(T_WEATHER)  # serving:delay_weather_region_month
        df_weather_delay = fetch_weather_heatmap(
            t_weather_delay, region, year, month_from, month_to, value_col="avg_dep_delay"
        )
    except Exception as e:
        print(f"[WARN] Weather delay chart skipped: {e}")

    # cancel heatmap (cancel_rate)
    try:
        t_weather_cancel = conn.table(T_WEATHER_CANCEL)  # serving:cancel_weather_region_month
        df_weather_cancel = fetch_weather_heatmap(
            t_weather_cancel, region, year, month_from, month_to, value_col="cancel_rate"
        )
    except Exception as e:
        print(f"[WARN] Weather cancel chart skipped: {e}")

    conn.close()

    save_charts(
        df_delay=df_delay,
        df_cancel=df_cancel,
        df_carrier=df_carrier,
        df_top10=df_top10,
        airport_id=airport_id,
        carrier=carrier,
        year=year,
        month_from=month_from,
        month_to=month_to,
        top10_year=top10_year,
        top10_month=top10_month,
        df_weather_delay=df_weather_delay,
        df_weather_cancel=df_weather_cancel,
        region=region,
        df_age=df_age,
        out_dir=out_dir,
    )

    # # --- One window with all charts ---
    # show_dashboard(
    #     df_delay=df_delay,
    #     df_cancel=df_cancel,
    #     df_carrier=df_carrier,
    #     df_top10=df_top10,
    #     airport_id=airport_id,
    #     carrier=carrier,
    #     year=year,
    #     month_from=month_from,
    #     month_to=month_to,
    #     top10_year=top10_year,
    #     top10_month=top10_month,
    #     df_weather_delay=df_weather_delay,
    #     df_weather_cancel=df_weather_cancel,
    #     region=region,
    #     df_age=df_age,
    # )

    print("Done.")


if __name__ == "__main__":
    main()