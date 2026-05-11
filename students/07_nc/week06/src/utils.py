from __future__ import annotations

import shutil
from pathlib import Path


def repo_root() -> Path:
    """Return repository root from students/07_nc/week06/src/utils.py."""
    return Path(__file__).resolve().parents[4]


def week06_root() -> Path:
    return Path(__file__).resolve().parents[1]


def setup_results_dir() -> Path:
    """Create a clean results directory under students/07_nc/week06."""
    results_dir = week06_root() / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def require_data_file() -> Path:
    """Find the course CSV without putting CSV files in this submission folder."""
    candidates = [
        repo_root() / "homework" / "week06" / "data" / "q3_marketing.csv",
        week06_root() / "data" / "q3_marketing.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "找不到 q3_marketing.csv。请确认数据文件位于以下任一路径：\n" + checked
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
