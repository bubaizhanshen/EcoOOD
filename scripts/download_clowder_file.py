from __future__ import annotations

import argparse
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CONTENT_RANGE_RE = re.compile(r"Content-Range:\s*bytes\s+\d+-\d+/(\d+)", re.IGNORECASE)


def probe_total_size(url: str) -> int:
    command = ["curl", "--http1.1", "-L", "-r", "0-0", "-D", "-", "-o", "/dev/null", url]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    headers = f"{result.stdout}\n{result.stderr}"
    match = CONTENT_RANGE_RE.search(headers)
    if match is None:
        raise RuntimeError(f"Could not infer file size from headers:\n{headers}")
    return int(match.group(1))


def download_chunk(url: str, output: Path, start: int, end: int) -> None:
    command = [
        "curl",
        "--http1.1",
        "-L",
        "-r",
        f"{start}-{end}",
        "-o",
        str(output),
        url,
    ]
    subprocess.run(command, check=True)


def download_chunk_with_retry(
    url: str,
    output: Path,
    start: int,
    end: int,
    *,
    retries: int,
    sleep_seconds: float,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            download_chunk(url, output, start, end)
            return
        except Exception as exc:
            last_error = exc
            if output.exists():
                output.unlink()
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"Failed to download bytes {start}-{end} after {retries} attempts") from last_error


def append_file(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("ab") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)


def expected_chunk_path(part_dir: Path, start: int, end: int) -> Path:
    return part_dir / f"{start:012d}-{end:012d}.part"


def ensure_chunk(
    url: str,
    part_dir: Path,
    start: int,
    end: int,
    *,
    retries: int,
    sleep_seconds: float,
) -> tuple[int, int, Path]:
    expected_size = end - start + 1
    part_path = expected_chunk_path(part_dir, start, end)
    if part_path.exists() and part_path.stat().st_size == expected_size:
        return start, end, part_path
    if part_path.exists():
        part_path.unlink()
    download_chunk_with_retry(
        url,
        part_path,
        start,
        end,
        retries=retries,
        sleep_seconds=sleep_seconds,
    )
    actual_size = part_path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"Chunk size mismatch for {start}-{end}: expected {expected_size}, got {actual_size}")
    return start, end, part_path


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download large public Clowder files with resumable range requests.")
    parser.add_argument("--url", required=True, help="Public file blob URL, e.g. https://clowder.../files/<id>/blob")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size-mb", type=int, default=1, help="Chunk size in MB. Keep small if the server is flaky.")
    parser.add_argument("--max-bytes", type=int, default=None, help="Optional cap for a partial download.")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--jobs", type=int, default=1, help="Number of chunk downloads to run in parallel.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_size = probe_total_size(args.url)
    if args.max_bytes is not None:
        total_size = min(total_size, args.max_bytes)
    chunk_size = max(1, args.chunk_size_mb) * 1024 * 1024
    downloaded = args.output.stat().st_size if args.output.exists() else 0
    if downloaded > total_size:
        raise RuntimeError(f"Existing output is larger than target size: {downloaded} > {total_size}")

    part_path = args.output.with_suffix(args.output.suffix + ".part")
    ranges = [
        (start, min(start + chunk_size - 1, total_size - 1))
        for start in range(downloaded, total_size, chunk_size)
    ]
    retries = max(1, args.retries)
    retry_sleep = max(0.1, args.retry_sleep)
    jobs = max(1, args.jobs)

    if jobs == 1:
        for start, end in ranges:
            expected_size = end - start + 1
            download_chunk_with_retry(
                args.url,
                part_path,
                start,
                end,
                retries=retries,
                sleep_seconds=retry_sleep,
            )
            actual_size = part_path.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(f"Chunk size mismatch for {start}-{end}: expected {expected_size}, got {actual_size}")
            append_file(part_path, args.output)
            part_path.unlink()
            done = end + 1
            print(f"Downloaded {human_bytes(done)} / {human_bytes(total_size)}")
    else:
        part_dir = args.output.with_suffix(args.output.suffix + ".parts")
        part_dir.mkdir(parents=True, exist_ok=True)
        completed = downloaded
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            future_map = {
                executor.submit(
                    ensure_chunk,
                    args.url,
                    part_dir,
                    start,
                    end,
                    retries=retries,
                    sleep_seconds=retry_sleep,
                ): (start, end)
                for start, end in ranges
            }
            for idx, future in enumerate(as_completed(future_map), start=1):
                start, end, _ = future.result()
                completed += end - start + 1
                if idx % max(1, jobs) == 0 or idx == len(future_map):
                    print(f"Fetched chunk {idx}/{len(future_map)}; buffered {human_bytes(completed)} / {human_bytes(total_size)}")
        for start, end in ranges:
            ordered_part = expected_chunk_path(part_dir, start, end)
            append_file(ordered_part, args.output)
            ordered_part.unlink()
            done = end + 1
            print(f"Merged {human_bytes(done)} / {human_bytes(total_size)}")
        part_dir.rmdir()

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
