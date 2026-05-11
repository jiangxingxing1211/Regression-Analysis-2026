from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationResult:
    model_name: str
    fit_time: float
    r2_score: float
    mse: float

    def to_markdown_row(self) -> str:
        return f"| {self.model_name} | {self.fit_time:.6f} 秒 | {self.r2_score:.4f} | {self.mse:.4f} |"


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> EvaluationResult:
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time

    y_pred = model.predict(X_test)
    r2_score = float(model.score(X_test, y_test))
    mse = float(np.mean((np.asarray(y_test) - y_pred) ** 2))

    return EvaluationResult(model_name=model_name, fit_time=fit_time, r2_score=r2_score, mse=mse)
