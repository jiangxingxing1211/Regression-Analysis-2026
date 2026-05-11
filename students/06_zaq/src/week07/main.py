"""Week 7: Pure Python - No numpy, no sklearn, no matplotlib needed"""
import math
import random
import csv
from pathlib import Path


def mean(values):
    return sum(values) / len(values)


class AnalyticalOLS:
    def __init__(self):
        self.coef_ = None
    
    def fit(self, X, y):
        n, p = len(X), len(X[0])
        XTX = [[0.0]*p for _ in range(p)]
        for i in range(p):
            for j in range(p):
                s = 0.0
                for k in range(n):
                    s += X[k][i] * X[k][j]
                XTX[i][j] = s
        
        XTy = [0.0]*p
        for i in range(p):
            s = 0.0
            for k in range(n):
                s += X[k][i] * y[k]
            XTy[i] = s
        
        # 高斯消元求逆
        aug = [[0.0]*(2*p) for _ in range(p)]
        for i in range(p):
            for j in range(p):
                aug[i][j] = XTX[i][j]
            aug[i][p+i] = 1.0
        
        for i in range(p):
            max_row = i
            for r in range(i+1, p):
                if abs(aug[r][i]) > abs(aug[max_row][i]):
                    max_row = r
            aug[i], aug[max_row] = aug[max_row], aug[i]
            
            pivot = aug[i][i]
            for j in range(2*p):
                aug[i][j] /= pivot
            
            for r in range(p):
                if r != i:
                    factor = aug[r][i]
                    for j in range(2*p):
                        aug[r][j] -= factor * aug[i][j]
        
        inv = [[aug[i][p+j] for j in range(p)] for i in range(p)]
        
        self.coef_ = [0.0]*p
        for i in range(p):
            s = 0.0
            for j in range(p):
                s += inv[i][j] * XTy[j]
            self.coef_[i] = s
        return self
    
    def predict(self, X):
        return [sum(self.coef_[j] * row[j] for j in range(len(self.coef_))) for row in X]
    
    def score(self, X, y):
        pred = self.predict(X)
        y_mean = mean(y)
        ss_res = sum((y[i] - pred[i])**2 for i in range(len(y)))
        ss_tot = sum((y[i] - y_mean)**2 for i in range(len(y)))
        return 1 - ss_res/ss_tot if ss_tot > 0 else 0


class GradientDescentOLS:
    def __init__(self, lr=0.01, tol=1e-5, max_iter=1000, gd_type="full_batch", batch_frac=0.2):
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_frac = batch_frac
        self.coef_ = None
        self.loss_history_ = []
    
    def fit(self, X, y, seed=42):
        random.seed(seed)
        n, p = len(X), len(X[0])
        self.coef_ = [0.0]*p
        self.loss_history_ = []
        
        for epoch in range(self.max_iter):
            if self.gd_type == "mini_batch":
                batch_size = max(1, int(n * self.batch_frac))
                indices = random.sample(range(n), batch_size)
                X_batch = [X[i] for i in indices]
                y_batch = [y[i] for i in indices]
            else:
                X_batch, y_batch = X, y
            
            pred_batch = [sum(self.coef_[j] * row[j] for j in range(p)) for row in X_batch]
            grad = [0.0]*p
            for j in range(p):
                s = 0.0
                for i in range(len(X_batch)):
                    s += (pred_batch[i] - y_batch[i]) * X_batch[i][j]
                grad[j] = 2 * s / len(X_batch)
            
            for j in range(p):
                self.coef_[j] -= self.lr * grad[j]
            
            pred_all = [sum(self.coef_[j] * row[j] for j in range(p)) for row in X]
            mse = sum((pred_all[i] - y[i])**2 for i in range(n)) / n
            self.loss_history_.append(mse)
            
            if epoch > 0 and abs(self.loss_history_[-1] - self.loss_history_[-2]) < self.tol:
                break
        return self
    
    def predict(self, X):
        return [sum(self.coef_[j] * row[j] for j in range(len(self.coef_))) for row in X]
    
    def score(self, X, y):
        pred = self.predict(X)
        y_mean = sum(y)/len(y)
        ss_res = sum((y[i] - pred[i])**2 for i in range(len(y)))
        ss_tot = sum((y[i] - y_mean)**2 for i in range(len(y)))
        return 1 - ss_res/ss_tot if ss_tot > 0 else 0


def add_intercept(X):
    return [[1.0] + row for row in X]


def standardize(X_train, X_val, X_test):
    n_features = len(X_train[0])
    means = [0.0]*n_features
    stds = [0.0]*n_features
    
    for j in range(n_features):
        col = [row[j] for row in X_train]
        means[j] = sum(col)/len(col)
        var = sum((x - means[j])**2 for x in col)/len(col)
        stds[j] = math.sqrt(var) if var > 0 else 1
    
    def transform(X):
        return [[(row[j] - means[j])/stds[j] for j in range(n_features)] for row in X]
    
    return transform(X_train), transform(X_val), transform(X_test)


def rmse(y_true, y_pred):
    return math.sqrt(sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true))) / len(y_true))


def kfold_cv(X, y, k=5):
    print("\n" + "="*60)
    print("Task 2: 5-Fold Cross-Validation")
    print("="*60)
    
    n = len(X)
    fold_size = n // k
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    
    r2_scores = []
    rmse_scores = []
    
    for fold in range(k):
        val_idx = indices[fold*fold_size:(fold+1)*fold_size] if fold < k-1 else indices[fold*fold_size:]
        train_idx = [i for i in indices if i not in val_idx]
        
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]
        
        model = AnalyticalOLS().fit(X_train, y_train)
        pred = model.predict(X_val)
        
        r2 = model.score(X_val, y_val)
        r2_scores.append(r2)
        rmse_scores.append(rmse(y_val, pred))
        
        print(f"Fold {fold+1}: R²={r2:.4f}, RMSE={rmse_scores[-1]:.4f}")
    
    print(f"\nAverage R²: {sum(r2_scores)/k:.4f}")
    print(f"Average RMSE: {sum(rmse_scores)/k:.4f}")


def main():
    print("="*60)
    print("Week 7: Optimization Engine (Pure Python)")
    print("="*60)
    
    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    data_path = Path("/mnt/c/Users/Del/OneDrive/Desktop/q3_marketing.csv")
    
    if not data_path.exists():
        print("No data file, using synthetic data...")
        random.seed(42)
        n = 500
        data = []
        for i in range(n):
            tv = random.gauss(50, 10)
            radio = random.gauss(30, 5)
            social = random.gauss(40, 8)
            sales = 80 + tv*0.5 + radio*0.3 + social*0.1 + random.gauss(0, 5)
            data.append([tv, radio, social, sales])
    else:
        with open(data_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = []
            for row in reader:
                data.append([float(x) for x in row[1:]])
    
    X = [row[:-1] for row in data]
    y = [row[-1] for row in data]
    
    print(f"Data: {len(X)} samples, {len(X[0])} features")
    
    X_cv = add_intercept(X)
    kfold_cv(X_cv, y)
    
    random.seed(42)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    
    train_idx = indices[:int(n*0.6)]
    val_idx = indices[int(n*0.6):int(n*0.8)]
    test_idx = indices[int(n*0.8):]
    
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    
    print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    X_train_s, X_val_s, X_test_s = standardize(X_train, X_val, X_test)
    
    X_train_s = add_intercept(X_train_s)
    X_val_s = add_intercept(X_val_s)
    X_test_s = add_intercept(X_test_s)
    
    print("\nTask 3: Tuning Learning Rate")
    print("-" * 40)
    
    lrs = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    best_lr = None
    best_r2 = -float('inf')
    
    for lr in lrs:
        model = GradientDescentOLS(lr=lr, gd_type="mini_batch", batch_frac=0.2)
        model.fit(X_train_s, y_train)
        val_r2 = model.score(X_val_s, y_val)
        print(f"LR={lr:<10} Val R²={val_r2:.4f}")
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_lr = lr
    
    print(f"\nBest LR: {best_lr}")
    
    print("\n" + "="*60)
    print("Final Test Results")
    print("="*60)
    
    gd_best = GradientDescentOLS(lr=best_lr, gd_type="mini_batch", batch_frac=0.2)
    gd_best.fit(X_train_s, y_train)
    ols = AnalyticalOLS().fit(X_train_s, y_train)
    
    gd_pred = gd_best.predict(X_test_s)
    ols_pred = ols.predict(X_test_s)
    
    print(f"\n{'Model':<20} | {'R²':<8} | {'RMSE':<8}")
    print("-" * 40)
    print(f"{'GradientDescentOLS':<20} | {gd_best.score(X_test_s, y_test):<8.4f} | {rmse(y_test, gd_pred):<8.4f}")
    print(f"{'AnalyticalOLS':<20} | {ols.score(X_test_s, y_test):<8.4f} | {rmse(y_test, ols_pred):<8.4f}")
    
    report_path = results_dir / "summary_report.md"
    with open(report_path, "w") as f:
        f.write("# Week 7 Report\n\n")
        f.write(f"Best Learning Rate: {best_lr}\n\n")
        f.write("## Test Results\n\n")
        f.write(f"- GD OLS R²: {gd_best.score(X_test_s, y_test):.4f}\n")
        f.write(f"- Analytical OLS R²: {ols.score(X_test_s, y_test):.4f}\n")
    
    print(f"\nReport saved: {report_path}")
    print("\n" + "="*60)
    print("All tasks completed!")
    print("="*60)


if __name__ == "__main__":
    main()
