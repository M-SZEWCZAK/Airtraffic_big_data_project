import os
import happybase
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# --- GUI (scrollable window) ---
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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


# ======================================================
# PARSER
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser("HBase Serving Layer Demo")

    parser.add_argument("--airport-id", default="12478")
    parser.add_argument("--carrier", default="AA")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month-from", type=int, default=1)
    parser.add_argument("--month-to", type=int, default=12)
    parser.add_argument("--top10-month", type=int, default=1)
    parser.add_argument("--top10-year", type=int, default=2025)

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
# DASHBOARD
# ======================================================

def show_dashboard(
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
):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # 1) Airport delays
    if not df_delay.empty and "avg_dep_delay" in df_delay.columns:
        ax1.plot(df_delay["month"], df_delay["avg_dep_delay"], marker="o")
        ax1.set_title(f"Avg dep delay (airport {airport_id}, {year})")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Delay [min]")
        ax1.grid(True)
    else:
        ax1.set_title(f"Avg dep delay (airport {airport_id}, {year})")
        ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_axis_off()

    # 2) Airport cancellations
    if not df_cancel.empty and "cancel_pct" in df_cancel.columns:
        df_cancel = df_cancel.copy()
        df_cancel["cancel_pct"] = df_cancel["cancel_pct"].apply(safe_float)

        ax2.plot(df_cancel["month"], df_cancel["cancel_pct"], marker="o")
        ax2.set_title(f"Cancel % (airport {airport_id}, {year})")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Cancel %")
        ax2.grid(True)
    else:
        ax2.set_title(f"Cancel % (airport {airport_id}, {year})")
        ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()

    # 3) Carrier delays
    if not df_carrier.empty and "avg_dep_delay" in df_carrier.columns:
        ax3.plot(df_carrier["month"], df_carrier["avg_dep_delay"], marker="o")
        ax3.set_title(f"Avg dep delay (carrier {carrier}, {year})")
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Delay [min]")
        ax3.grid(True)
    else:
        ax3.set_title(f"Avg dep delay (carrier {carrier}, {year})")
        ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_axis_off()

    # 4) Top10 airports
    if not df_top10.empty and "airport_id" in df_top10.columns and "avg_dep_delay" in df_top10.columns:
        ax4.bar(df_top10["airport_id"], df_top10["avg_dep_delay"])
        ax4.set_title(f"Top10 airports avg dep delay ({top10_year}-{top10_month:02d})")
        ax4.set_xlabel("AirportID")
        ax4.set_ylabel("Delay [min]")
        for label in ax4.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
        ax4.grid(False)
    else:
        ax4.set_title(f"Top10 airports avg dep delay ({top10_year}-{top10_month:02d})")
        ax4.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_axis_off()

    fig.suptitle(
        f"HBase Serving Demo | airport={airport_id} carrier={carrier} "
        f"range={year}-{month_from:02d}..{month_to:02d} | top10={top10_year}-{top10_month:02d}",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    root = tk.Tk()
    root.title("HBase Serving Layer Demo - Dashboard")

    container = ttk.Frame(root)
    container.pack(fill="both", expand=True)

    canvas = tk.Canvas(container)
    vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)

    vscroll.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = ttk.Frame(canvas)
    inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    mpl_canvas = FigureCanvasTkAgg(fig, master=inner)
    mpl_widget = mpl_canvas.get_tk_widget()
    mpl_widget.pack(fill="both", expand=True)

    mpl_canvas.draw()

    def _on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    inner.bind("<Configure>", _on_configure)

    def _on_canvas_configure(event):
        canvas.itemconfig(inner_id, width=event.width)

    canvas.bind("<Configure>", _on_canvas_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    root.minsize(900, 600)
    root.mainloop()


# ======================================================
# MAIN
# ======================================================

def main():
    args = parse_args()

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

    conn.close()

    # --- One window with all charts ---
    show_dashboard(
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
    )

    print("Done.")


if __name__ == "__main__":
    main()