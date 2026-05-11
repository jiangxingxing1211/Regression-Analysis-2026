from __future__ import annotations

from reporting import write_week06_report
from scenarios import scenario_A_synthetic, scenario_B_real_world
from utils import setup_results_dir


def main() -> None:
    results_dir = setup_results_dir()
    synthetic_results, synthetic_section = scenario_A_synthetic(results_dir)
    real_rows, real_world_section = scenario_B_real_world(results_dir)
    write_week06_report(results_dir, synthetic_results, real_rows, synthetic_section, real_world_section)

    print("第 6 周作业运行完成。")
    print(f"中文 Markdown 总报告已生成：{results_dir / 'week06_report.md'}")
    print(f"图片已生成到：{results_dir}")


if __name__ == "__main__":
    main()
