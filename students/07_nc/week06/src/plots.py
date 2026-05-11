from __future__ import annotations

from pathlib import Path
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_chinese_font():
    """设置 Matplotlib 中文字体，兼容 Windows 和 WSL/Linux。"""
    font_files = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]

    for font_file in font_files:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
            font_name = font_manager.FontProperties(fname=str(font_file)).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name

    plt.rcParams["axes.unicode_minus"] = False
    return None


setup_chinese_font()


def save_predicted_vs_actual(y_true, y_pred, path: Path, title: str) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_residual_plot(y_true, y_pred, path: Path, title: str) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("预测销售额")
    plt.ylabel("残差：真实值 - 预测值")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_market_comparison(df: pd.DataFrame, path: Path) -> None:
    summary = df.groupby("Region")[["Sales", "TV_Budget", "Radio_Budget", "SocialMedia_Budget"]].mean()
    ax = summary.plot(kind="bar", figsize=(9, 5))
    ax.set_title("北美与欧洲市场均值对比")
    ax.set_xlabel("市场")
    ax.set_ylabel("平均值")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
