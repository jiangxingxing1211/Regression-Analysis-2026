import shutil
from pathlib import Path

def setup_results_dir() -> Path:
    """清空或创建 results/ 文件夹，返回其路径。"""
    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir