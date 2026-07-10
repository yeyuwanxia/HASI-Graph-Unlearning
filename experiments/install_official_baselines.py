from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines import official_specs_as_dict
from baselines.official_sources import OFFICIAL_BASELINE_SPECS


def parse_args():
    parser = argparse.ArgumentParser(description="Clone official baseline repositories into external/official_baselines.")
    parser.add_argument(
        "--baseline",
        default="all",
        choices=["all", *sorted(OFFICIAL_BASELINE_SPECS)],
        help="Official source to install.",
    )
    parser.add_argument("--force", action="store_true", help="Re-clone an existing directory.")
    parser.add_argument("--list", action="store_true", help="Only list official source metadata.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list:
        print(json.dumps(official_specs_as_dict(ROOT), indent=2))
        return

    install_root = ROOT / "external" / "official_baselines"
    install_root.mkdir(parents=True, exist_ok=True)
    keys = sorted(OFFICIAL_BASELINE_SPECS) if args.baseline == "all" else [args.baseline]
    results = []
    for key in keys:
        spec = OFFICIAL_BASELINE_SPECS[key]
        target = install_root / spec.local_dir
        if target.exists():
            if not args.force:
                results.append({"baseline": key, "status": "exists", "path": str(target)})
                continue
            _remove_directory(target)
        command = ["git", "clone", "--depth", "1", spec.repo_url, str(target)]
        try:
            subprocess.run(command, check=True)
            results.append({"baseline": key, "status": "cloned", "path": str(target), "repo_url": spec.repo_url})
        except subprocess.CalledProcessError as exc:
            results.append(
                {
                    "baseline": key,
                    "status": "failed",
                    "path": str(target),
                    "repo_url": spec.repo_url,
                    "returncode": exc.returncode,
                    "command": command,
                }
            )
    print(json.dumps(results, indent=2))


def _remove_directory(path: Path) -> None:
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            child.rmdir()
    path.rmdir()


if __name__ == "__main__":
    main()
