import os
import sys
import time
import subprocess
from pathlib import Path

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

    spark_submit = which_or_fail("spark-submit")
    python_exe = sys.executable

    raw_dir = root / "raw_and_batch_views_processing"
    spark_dir = root / "spark"
    serving_dir = root / "serving"

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

    # Pass demo args for random reads
    # e.g. ["--airport-id", "12478", "--year", "2024", "--carrier", "AA"]
    demo_args: list[str] = []

    run_cmd([python_exe, str(demo_job), *demo_args])

    print("\nEnd-to-end pipeline finished successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nPipeline failed:", e)
        sys.exit(1)
