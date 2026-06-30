from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the EcoOOD Streamlit dashboard.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8501, help="Port for the Streamlit server.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a browser.")
    args = parser.parse_args()

    app_path = Path(__file__).resolve().parents[1] / "app" / "ecoood_explorer.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        cmd.extend(["--server.headless", "true"])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
