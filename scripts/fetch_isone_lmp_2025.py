from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

BASE_URL = "https://www.iso-ne.com/static-transform/csv/histRpts/rt-lmp/lmp_rt_final_{yyyymmdd}.csv"
TARGET_NODE = "LD.E_CAMBRG13.8"


@dataclass(frozen=True)
class Job:
    day: date
    url: str
    out_path: Path


def daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def make_jobs(start: date, end: date, raw_dir: Path) -> list[Job]:
    jobs: list[Job] = []
    for day in daterange(start, end):
        stamp = day.strftime("%Y%m%d")
        jobs.append(Job(day=day, url=BASE_URL.format(yyyymmdd=stamp), out_path=raw_dir / f"lmp_rt_final_{stamp}.csv"))
    return jobs


def download_one(session: requests.Session, job: Job, timeout: int) -> tuple[Job, bool, str | None]:
    if job.out_path.exists() and job.out_path.stat().st_size > 0:
        return job, False, None

    try:
        response = session.get(job.url, timeout=timeout)
        response.raise_for_status()
        job.out_path.write_bytes(response.content)
        return job, True, None
    except Exception as exc:  # noqa: BLE001
        return job, False, str(exc)


def download_all(jobs: list[Job], max_workers: int, timeout: int) -> None:
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(download_one, session, job, timeout) for job in jobs]
            failures: list[tuple[Job, str]] = []
            downloaded = 0
            skipped = 0
            for future in as_completed(futures):
                job, did_download, error = future.result()
                if error:
                    failures.append((job, error))
                elif did_download:
                    downloaded += 1
                else:
                    skipped += 1

    if failures:
        sample = "\n".join(f"  {j.day.isoformat()}: {err}" for j, err in failures[:10])
        raise RuntimeError(
            f"Failed downloads: {len(failures)}\n{sample}\n"
            "Inspect connectivity and rerun; successful files are cached."
        )

    print(f"Download complete: {downloaded} downloaded, {skipped} already present.")


def parse_he_to_hour_index(he_raw: str) -> tuple[int, bool]:
    he = he_raw.strip().upper()
    if he.endswith("X"):
        return int(he[:-1]) - 1, True
    return int(he) - 1, False


def parse_daily_file(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for record in reader:
            if not record or record[0] != "D":
                continue

            day = datetime.strptime(record[1], "%m/%d/%Y").date()
            hour_idx, is_extra_dst_hour = parse_he_to_hour_index(record[2])
            interval_start = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour_idx)
            interval_end = interval_start + timedelta(hours=1)

            rows.append(
                {
                    "date": day.isoformat(),
                    "hour_ending": record[2],
                    "interval_start_local": interval_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "interval_end_local": interval_end.strftime("%Y-%m-%d %H:%M:%S"),
                    "is_extra_dst_hour": is_extra_dst_hour,
                    "location_id": int(record[3]),
                    "location_name": record[4].strip(),
                    "location_type": record[5].strip(),
                    "lmp": float(record[6]),
                    "energy_component": float(record[7]),
                    "congestion_component": float(record[8]),
                    "marginal_loss_component": float(record[9]),
                }
            )

    if not rows:
        raise RuntimeError(f"No data rows parsed from {path}")

    return pd.DataFrame(rows)


def build_dataset(raw_dir: Path, pattern: str) -> pd.DataFrame:
    csv_files = sorted(raw_dir.glob(pattern))
    if not csv_files:
        raise RuntimeError(f"No files found in {raw_dir} matching {pattern}")

    daily_frames = [parse_daily_file(path) for path in csv_files]
    df = pd.concat(daily_frames, ignore_index=True)
    df.sort_values(["date", "hour_ending", "location_name"], inplace=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and compile ISO-NE RT hourly final LMP data.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/isone_rt_lmp_2025"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    start = date(args.year, 1, 1)
    end = date(args.year, 12, 31)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    jobs = make_jobs(start, end, args.raw_dir)
    download_all(jobs, max_workers=args.workers, timeout=args.timeout)

    full_df = build_dataset(args.raw_dir, "lmp_rt_final_*.csv")
    full_out = args.processed_dir / f"isone_rt_lmp_hourly_{args.year}.csv.gz"
    full_df.to_csv(full_out, index=False, compression="gzip")

    node_df = full_df[full_df["location_name"] == TARGET_NODE].copy()
    if node_df.empty:
        raise RuntimeError(f"Target node {TARGET_NODE} not found in compiled dataset.")

    node_out = args.processed_dir / f"isone_rt_lmp_hourly_{args.year}_{TARGET_NODE}.csv"
    node_df.to_csv(node_out, index=False)

    expected_hours = 8760
    observed_hours = node_df.shape[0]
    print(f"Wrote full dataset: {full_out} ({len(full_df):,} rows)")
    print(f"Wrote node dataset: {node_out} ({observed_hours:,} rows)")
    if observed_hours != expected_hours:
        print(
            f"Warning: expected ~{expected_hours} rows for a single node in {args.year}, "
            f"observed {observed_hours}. Check DST or missing source files."
        )


if __name__ == "__main__":
    main()
