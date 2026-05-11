import time
import shutil
from pathlib import Path

def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    r2 = model.score(X_test, y_test)
    return f"| {name:20} | {fit_time:.5f} | {r2:.6f} |\n", r2

def setup_results_dir():
    d = Path(__file__).parent.parent.parent.parent / "results"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d

def add_intercept(X):
    return [[1.0] + row for row in X]