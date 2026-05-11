from pathlib import Path
from utils import setup_results_dir
from scenarios import scenario_A_synthetic, scenario_B_real_world


def main():
    results_dir = setup_results_dir()
    
    print("=" * 60)
    print("Milestone Project 1: The Inference Engine")
    print("Implementation: Class-Based OOP Approach")
    print("=" * 60)
    print()
    
    print("Running Scenario A: Synthetic Data Baseline Test...")
    scenario_A_synthetic(results_dir)
    print()
    
    print("Running Scenario B: Real-World Marketing Data Analysis...")
    scenario_B_real_world(results_dir)
    print()
    
    print("=" * 60)
    print("All tasks completed successfully!")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()