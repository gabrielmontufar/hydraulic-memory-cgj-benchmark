from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"manifest_sha256.csv", "run_all_log.csv"}:
            continue
        if "__pycache__" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
    DATA.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA / "manifest_sha256.csv", index=False)
    print(f"refresh_manifest_sha256=ok files={len(rows)}")


if __name__ == "__main__":
    main()
