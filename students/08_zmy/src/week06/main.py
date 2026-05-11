from utils import setup_results_dir
from scenarios import scenario_A_synthetic, scenario_B_real_world

def main():
    # 1. 准备 results 目录（位于 week06/results/）
    results_dir = setup_results_dir()
    print(f"Results directory: {results_dir}")

    # 2. 运行场景 A（合成数据）
    scenario_A_synthetic(results_dir)
    print("Scenario A finished. Check synthetic_report.md")

    # 3. 运行场景 B（真实数据）
    scenario_B_real_world(results_dir)
    print("Scenario B finished. Check real_world_report.md and market_comparison.png")

if __name__ == "__main__":
    main()