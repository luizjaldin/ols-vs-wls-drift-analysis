import numpy as np


class WeightedLeastSquares:
    """Mínimos Quadrados Ponderados via equações normais.

    Resolve beta = (X'WX)^{-1} X'Wy usando a transformação
    X_w = sqrt(w) * X, y_w = sqrt(w) * y e np.linalg.solve.
    Com sample_weight=None equivale a OLS (pesos unitários).
    """

    def __init__(self, fit_intercept=True, ridge_eps=0.0):
        self.fit_intercept = fit_intercept
        self.ridge_eps = ridge_eps
        self.coef_ = None
        self.intercept_ = None
        self.beta_ = None
        self.cond_ = None

    def _montar_matriz(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, X, y, sample_weight=None):
        X_design = self._montar_matriz(X)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if sample_weight is None:
            sample_weight = np.ones(X_design.shape[0])
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1, 1)

        if X_design.shape[0] != y.shape[0]:
            raise ValueError("X e y precisam ter o mesmo número de observações.")
        if X_design.shape[0] != sample_weight.shape[0]:
            raise ValueError("X e sample_weight precisam ter o mesmo número de observações.")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight não pode conter valores negativos.")
        if np.all(sample_weight == 0):
            raise ValueError("Todos os pesos são zero; o modelo não pode ser ajustado.")

        sqrt_w = np.sqrt(sample_weight)
        X_ponderado = X_design * sqrt_w
        y_ponderado = y * sqrt_w

        XtWX = X_ponderado.T @ X_ponderado
        XtWy = X_ponderado.T @ y_ponderado

        if self.ridge_eps > 0:
            XtWX = XtWX + self.ridge_eps * np.eye(XtWX.shape[0])

        # número de condição das equações normais — mal condicionamento
        # sinaliza colinearidade nas features (ver seção de condicionamento
        # do notebook)
        self.cond_ = np.linalg.cond(XtWX)

        beta = np.linalg.solve(XtWX, XtWy)
        self.beta_ = beta.flatten()

        if self.fit_intercept:
            self.intercept_ = self.beta_[0]
            self.coef_ = self.beta_[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.beta_
        return self

    def predict(self, X):
        if self.beta_ is None:
            raise ValueError("Modelo não ajustado. Chame .fit() antes de .predict().")
        X_design = self._montar_matriz(X)
        return (X_design @ self.beta_.reshape(-1, 1)).flatten()


class OrdinaryLeastSquares(WeightedLeastSquares):
    """OLS = WLS com todos os pesos iguais a 1."""

    def fit(self, X, y):
        return super().fit(X, y, sample_weight=None)
