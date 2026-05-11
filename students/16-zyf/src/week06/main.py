import shutil
from pathlib import Path

from simulation import (
    generate_summary_report,
    scenario_A_synthetic,
    scenario_B_real_world,
)


def setup_results_dir() -> Path:
    """
    自动管理 results 文件夹
    位置：src/week06/results
    """
    results_dir = Path(__file__).parent / "results"

    if results_dir.exists():
        shutil.rmtree(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


if __name__ == "__main__":
    results_dir = setup_results_dir()

    scenario_A_synthetic(results_dir)
    scenario_B_real_world(results_dir)
    generate_summary_report(results_dir)

    print("✅ Homework finished. All outputs are saved in src/week06/results/")