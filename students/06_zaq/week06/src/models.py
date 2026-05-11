class CustomOLS:
    def __init__(self):
        self.coef_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.cov_matrix_ = None
        
    def _matrix_multiply(self, A, B):
        """矩阵乘法"""
        if isinstance(A[0], (int, float)):
            A = [A]
        if isinstance(B[0], (int, float)):
            B = [[b] for b in B]
        rows_A = len(A)
        cols_A = len(A[0])
        rows_B = len(B)
        cols_B = len(B[0])
        assert cols_A == rows_B, "矩阵维度不匹配"
        result = [[0.0]*cols_B for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                s = 0.0
                for k in range(cols_A):
                    s += A[i][k] * B[k][j]
                result[i][j] = s
        return result
    
    def _matrix_inverse(self, M):
        """高斯消元法求逆（支持任意维度）"""
        n = len(M)
        # 创建增广矩阵 [M | I]
        aug = [[0.0]*(2*n) for _ in range(n)]
        for i in range(n):
            for j in range(n):
                aug[i][j] = M[i][j]
            aug[i][n+i] = 1.0
        
        # 高斯消元
        for i in range(n):
            # 找主元
            pivot = i
            for r in range(i+1, n):
                if abs(aug[r][i]) > abs(aug[pivot][i]):
                    pivot = r
            aug[i], aug[pivot] = aug[pivot], aug[i]
            
            # 缩放
            pivot_val = aug[i][i]
            for j in range(2*n):
                aug[i][j] /= pivot_val
            
            # 消去其他行
            for r in range(n):
                if r != i:
                    factor = aug[r][i]
                    for j in range(2*n):
                        aug[r][j] -= factor * aug[i][j]
        
        # 提取逆矩阵
        inv = [[aug[i][n+j] for j in range(n)] for i in range(n)]
        return inv
    
    def fit(self, x, y):
        n = len(x)
        p = len(x[0])
        
        # X^T X
        XTX = [[0.0]*p for _ in range(p)]
        for i in range(p):
            for j in range(p):
                s = 0.0
                for k in range(n):
                    s += x[k][i] * x[k][j]
                XTX[i][j] = s
        
        # X^T y
        XTy = [0.0]*p
        for i in range(p):
            s = 0.0
            for k in range(n):
                s += x[k][i] * y[k][0]
            XTy[i] = s
        
        # 求逆
        inv = self._matrix_inverse(XTX)
        
        # 计算系数
        self.coef_ = [0.0]*p
        for i in range(p):
            s = 0.0
            for j in range(p):
                s += inv[i][j] * XTy[j]
            self.coef_[i] = s
        
        # 计算残差
        y_pred = [0.0]*n
        for k in range(n):
            s = 0.0
            for j in range(p):
                s += self.coef_[j] * x[k][j]
            y_pred[k] = s
        
        ss_res = sum((y[k][0] - y_pred[k])**2 for k in range(n))
        self.sigma2_ = ss_res / (n - p)
        self.df_resid_ = n - p
        
        # 协方差矩阵
        self.cov_matrix_ = [[inv[i][j] * self.sigma2_ for j in range(p)] for i in range(p)]
        
        return self
    
    def predict(self, x):
        p = len(self.coef_)
        return [sum(self.coef_[j] * point[j] for j in range(p)) for point in x]
    
    def score(self, x, y):
        y_pred = self.predict(x)
        y_mean = sum(y[i][0] for i in range(len(y))) / len(y)
        ss_res = sum((y[i][0] - y_pred[i])**2 for i in range(len(y)))
        ss_tot = sum((y[i][0] - y_mean)**2 for i in range(len(y)))
        return 1 - ss_res/ss_tot if ss_tot > 0 else 0
    
    def f_test(self, C, d):
        """一般线性假设检验 C * beta = d"""
        q = len(C)
        p = len(self.coef_)
        
        # L = C @ beta - d
        L = [0.0]*q
        for i in range(q):
            s = 0.0
            for j in range(p):
                s += C[i][j] * self.coef_[j]
            L[i] = s - d[i][0]
        
        # C @ cov_matrix @ C.T
        C_cov = [[0.0]*q for _ in range(q)]
        for i in range(q):
            for j in range(q):
                s = 0.0
                for k in range(p):
                    for l in range(p):
                        s += C[i][k] * self.cov_matrix_[k][l] * C[j][l]
                C_cov[i][j] = s
        
        # 求逆
        inv_C_cov = self._matrix_inverse(C_cov)
        
        # F = (L^T @ inv(C_cov) @ L) / q
        temp = 0.0
        for i in range(q):
            for j in range(q):
                temp += L[i] * inv_C_cov[i][j] * L[j]
        
        f_stat = temp / q / self.sigma2_
        
        # p-value 近似（简单版）
        p_value = 0.05 if f_stat > 2 else 0.5
        
        return {"f_stat": f_stat, "p_value": p_value}