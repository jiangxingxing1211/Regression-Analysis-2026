import numpy as np
from utils.models import CustomOLS

def calculate_vif(X):
    n, p = X.shape
    vif = np.zeros(p)
    for i in range(p):
        y = X[:,i]
        x = X[:,[j for j in range(p) if j!=i]]
        m = CustomOLS(alpha=0.01)
        m.fit(x,y)
        yp = m.predict(x)
        ssr = np.sum((y-yp)**2)
        sst = np.sum((y-y.mean())**2)
        r2 = 1 - ssr/sst if sst>1e-9 else 0.999
        v = 1/(1-r2) if r2 < 0.999 else 100.0
        vif[i] = v
    return vif

def detect_multicollinearity(X, names, threshold=10):
    v = calculate_vif(X)
    warn = []
    for n, vi in zip(names, v):
        if vi > threshold:
            warn.append(f"{n}: {vi:.1f}")
    return v, warn

def print_vif_report(v, names, threshold=10):
    print("\nVIF 报告:")
    for n, vi in zip(names, v):
        print(f"{n:15} {vi:.1f}")
