#!/usr/bin/env python3
"""
Helper to create timestamped output directories with git commit hash.
Usage: python create_timestamped_output_dir.py <base_name>
Returns: <base_name>_YYYYMMDD_HHMMSS_<git_short_hash>
"""

import subprocess
import sys
from datetime import datetime


def get_git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except:
        return "nogit"


def create_output_dir(base_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_hash = get_git_hash()
    return f"{base_name}_{timestamp}_{git_hash}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_timestamped_output_dir.py <base_name>")
        sys.exit(1)

    import os

    output_dir = create_output_dir(sys.argv[1])
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
