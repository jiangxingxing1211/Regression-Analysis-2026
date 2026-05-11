from pathlib import Path
from utils import setup_results_dir
from scenarios import scenario_A_synthetic, scenario_B_real_world

def main():
    # 准备输出目录
    results_dir = setup_results_dir()

    print("=== 运行场景 A：合成数据基准测试 ===")
    scenario_A_synthetic(results_dir)

    print("\n=== 运行场景 B：真实市场营销数据 ===")
    scenario_B_real_world(results_dir)

    print(f"\n所有分析报告已保存至：{results_dir.resolve()}")

if __name__ == "__main__":
    main()