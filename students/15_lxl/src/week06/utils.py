import time
import shutil
from pathlib import Path
from models import CustomOLS
from sklearn.linear_model import LinearRegression


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
    start_time = time.perf_counter()
    
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time
    
    r2_score = model.score(X_test, y_test)
    
    result_str = f"| {model_name} | {fit_time:.5f} sec | {r2_score:.4f} |\n"
    return result_str


def setup_results_dir() -> Path:
    results_dir = Path(__file__).parent / "results"
    
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir