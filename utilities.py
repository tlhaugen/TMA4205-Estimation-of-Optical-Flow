import numpy as np
from typing import Tuple, Optional, Callable


def forward_diff_x_boundary(I: np.ndarray) -> np.ndarray:
    n, m = I.shape
    dx = np.zeros_like(I)
    dx[:, : m - 1] = I[:, 1:] - I[:, :-1]
    dx[:, m - 1] = I[:, m - 1] - I[:, m - 2]
    return dx


def forward_diff_y_boundary(I: np.ndarray) -> np.ndarray:
    n, m = I.shape
    dy = np.zeros_like(I)
    dy[: n - 1, :] = I[1:, :] - I[:-1, :]
    dy[n - 1, :] = I[n - 1, :] - I[n - 2, :]
    return dy


def compute_derivatives(
    I0: np.ndarray, I1: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ix = 0.5 * (forward_diff_x_boundary(I0) + forward_diff_x_boundary(I1))
    Iy = 0.5 * (forward_diff_y_boundary(I0) + forward_diff_y_boundary(I1))
    It = I1 - I0
    return Ix, Iy, It


def build_rhs(
    Ix: np.ndarray, Iy: np.ndarray, It: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return -It * Ix, -It * Iy


def laplacian5_dirichlet(u: np.ndarray) -> np.ndarray:
    n, m = u.shape
    L = np.zeros_like(u)
    L[1 : n - 1, 1 : m - 1] = (
        u[0 : n - 2, 1 : m - 1]
        + u[2:n, 1 : m - 1]
        + u[1 : n - 1, 0 : m - 2]
        + u[1 : n - 1, 2:m]
        - 4.0 * u[1 : n - 1, 1 : m - 1]
    )
    L[0, :] = 0.0
    L[-1, :] = 0.0
    L[:, 0] = 0.0
    L[:, -1] = 0.0
    return L


def apply_A(
    u: np.ndarray, v: np.ndarray, Ix: np.ndarray, Iy: np.ndarray, reg: float
) -> Tuple[np.ndarray, np.ndarray]:
    Lu = laplacian5_dirichlet(u)
    Lv = laplacian5_dirichlet(v)
    Au = (Ix * Ix) * u + (Ix * Iy) * v - reg * Lu
    Av = (Ix * Iy) * u + (Iy * Iy) * v - reg * Lv
    Au[0, :] = 0.0
    Au[-1, :] = 0.0
    Au[:, 0] = 0.0
    Au[:, -1] = 0.0
    Av[0, :] = 0.0
    Av[-1, :] = 0.0
    Av[:, 0] = 0.0
    Av[:, -1] = 0.0
    return Au, Av


def of_cg(
    u0: np.ndarray,
    v0: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    reg: float,
    rhsu: np.ndarray,
    rhsv: np.ndarray,
    tol: float = 1e-8,
    maxit: int = 2000,
    callback: Optional[Callable[[int, float], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    u = u0.copy()
    v = v0.copy()
    Au, Av = apply_A(u, v, Ix, Iy, reg)
    ru = rhsu - Au
    rv = rhsv - Av
    r2_0 = float(np.sum(ru * ru + rv * rv))
    if r2_0 == 0.0:
        return u, v, 0, 0.0
    pu = ru.copy()
    pv = rv.copy()
    r2_old = r2_0
    for it in range(1, maxit + 1):
        Apu, Apv = apply_A(pu, pv, Ix, Iy, reg)
        denom = float(np.sum(pu * Apu + pv * Apv))
        if denom <= 0.0:
            break
        alpha = r2_old / denom
        u += alpha * pu
        v += alpha * pv
        ru -= alpha * Apu
        rv -= alpha * Apv
        r2_new = float(np.sum(ru * ru + rv * rv))
        rel = r2_new / r2_0
        if callback is not None:
            callback(it, rel)
        if rel < tol:
            return u, v, it, rel
        beta = r2_new / r2_old
        pu = ru + beta * pu
        pv = rv + beta * pv
        r2_old = r2_new
    return u, v, it, r2_old / r2_0
