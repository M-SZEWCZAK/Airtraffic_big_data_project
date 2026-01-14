import os
import sys
import time
import subprocess
from pathlib import Path
import argparse

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    """Run a command and stop pipeline on non-zero exit."""
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    if cwd:
        print("CWD:", str(cwd))
    print("=" * 100)

    start = time.time()
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    dt = time.time() - start

    if p.returncode != 0:
        raise RuntimeError(f"FAILED (exit={p.returncode}) after {dt:.1f}s: {' '.join(cmd)}")

    print(f"OK ({dt:.1f}s)")

def which_or_fail(exe: str) -> str:
    """Find executable in PATH (simple check)."""
    from shutil import which
    path = which(exe)
    if not path:
        raise RuntimeError(f"Executable not found in PATH: {exe}")
    return path

def project_root() -> Path:
    return Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main():
    root = project_root()

    parser = argparse.ArgumentParser(
        description="Run end-to-end pipeline (bronze->silver->batch->hbase->demo)."
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Run pipeline without launching the GUI demo."
    )
    parser.add_argument(
        "--skip-etl",
        action="store_true",
        help="Skip Spark ETL jobs (run only demo)."
    )
    parser.add_argument(
        "demo_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to demo_hbase_plots.py. Use: -- <demo args>. "
             "Example: -- --region California --year 2023 --airport_id 12478"
    )

    args = parser.parse_args()

    demo_args = args.demo_args
    if demo_args and demo_args[0] == "--":
        demo_args = demo_args[1:]

    spark_submit = which_or_fail("spark-submit")
    python_exe = sys.executable

    raw_dir = root / "raw_and_batch_views_processing"
    spark_dir = root / "spark"
    serving_dir = root / "serving"

    if not args.skip_etl:

        # --- 1) Bronze -> Silver (Spark jobs)
        silver_jobs = [
            raw_dir / "airports_modification.py",
            raw_dir / "planes_modification.py",
            raw_dir / "flights_modification.py",
            raw_dir / "weather_airport_join_modification.py",
        ]

        for job in silver_jobs:
            if not job.exists():
                raise FileNotFoundError(f"Missing file: {job}")
            run_cmd([spark_submit, str(job)])

        # --- 2) Silver -> Batch views in Hive (Spark job)
        batch_views_job = spark_dir / "build_batch_views.py"
        if not batch_views_job.exists():
            raise FileNotFoundError(f"Missing file: {batch_views_job}")
        run_cmd([spark_submit, str(batch_views_job)])

        # --- 3) Hive serving tables -> HBase (Spark job with HappyBase)
        load_hbase_job = serving_dir / "load_to_hbase.py"
        if not load_hbase_job.exists():
            raise FileNotFoundError(f"Missing file: {load_hbase_job}")
        run_cmd([spark_submit, str(load_hbase_job)])

    # --- 4) Demo (Python GUI / plots)
    demo_job = serving_dir / "demo_hbase_plots.py"
    if not demo_job.exists():
        raise FileNotFoundError(f"Missing file: {demo_job}")

    # old version
    # Pass demo args for random reads
    # e.g. ["--airport-id", "12478", "--year", "2024", "--carrier", "AA"]
    #demo_args: list[str] = []

    if demo_args:
        print("Forwarding demo args:", " ".join(demo_args))
    else:
        print("No demo args provided (demo will use defaults).")

    if not args.skip_demo:
        run_cmd([python_exe, str(demo_job), *demo_args])

    print("\nEnd-to-end pipeline finished successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nPipeline failed:", e)
        sys.exit(1)
