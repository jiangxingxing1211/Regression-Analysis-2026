import numpy as np
import matplotlib.pyplot as plt
from simulation import run_monte_carlo

if __name__ == "__main__":
    beta_true = np.array([0.0, 5.0, 3.0])
    beta_hat_A, _ = run_monte_carlo(rho=0.0)
    beta_hat_B, _ = run_monte_carlo(rho=0.99)

    plt.figure(figsize=(8, 8))

    plt.scatter(beta_hat_A[:, 1], beta_hat_A[:, 2],
                alpha=0.5, s=8, color="blue", label="A: ρ=0.0 (Independent)")

    plt.scatter(beta_hat_B[:, 1], beta_hat_B[:, 2],
                alpha=0.5, s=8, color="red", label="B: ρ=0.99 (Collinear)")

    plt.scatter(beta_true[1], beta_true[2],
                color="black", marker="*", s=300, label="True β = (5, 3)")

    plt.xlabel(r"$\hat{\beta}_1$")
    plt.ylabel(r"$\hat{\beta}_2$")
    plt.title("Monte Carlo Estimates of β1 vs β2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis("equal")
    
    # 保存图片
    plt.savefig("scatter.png", dpi=300, bbox_inches="tight")
    plt.show()