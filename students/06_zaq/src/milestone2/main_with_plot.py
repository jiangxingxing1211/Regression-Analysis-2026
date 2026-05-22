    # 可选加分：绘制柱状图
    try:
        import matplotlib.pyplot as plt
        
        metrics = ['RMSE', 'MAE', 'MAPE']
        bad_values = [bad_results['rmse_mean'], bad_results['mae_mean'], bad_results['mape_mean']]
        good_values = [good_results['rmse_mean'], good_results['mae_mean'], good_results['mape_mean']]
        
        x = range(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar([i - width/2 for i in x], bad_values, width, label='有泄露 (Bad CV)', color='red', alpha=0.7)
        bars2 = ax.bar([i + width/2 for i in x], good_values, width, label='无泄露 (Good CV)', color='green', alpha=0.7)
        
        ax.set_xlabel('指标')
        ax.set_ylabel('误差值')
        ax.set_title('数据泄露对模型评估的影响对比')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / "leakage_analysis.png", dpi=150)
        plt.close()
        print(f"✅ 柱状图已保存: {results_dir / 'leakage_analysis.png'}")
    except ImportError:
        print("⚠️ matplotlib 未安装，跳过绘图")
