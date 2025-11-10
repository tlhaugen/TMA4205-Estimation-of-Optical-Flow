import numpy as np
from scipy.ndimage import gaussian_filter

def interior_image(I):
    return I[1:-1, 1:-1]

def smooth(I, sigma):    
    return gaussian_filter(I, sigma)

def forward_diff_boundary(I):
    n, m = I.shape
    dx = np.zeros_like(I)
    dy = np.zeros_like(I)

    dx[:, :m-1] = I[:, 1:] - I[:, :-1]
    dx[:, m-1] = I[:, m-1] - I[:, m-2]

    dy[:n-1, :] = I[1:, :] - I[:-1, :]
    dy[n-1, :] = I[n-1, :] - I[n-2, :]
    return dx, dy

def compute_derivatives(I0, I1):
    Ix0, Iy0 = forward_diff_boundary(I0)
    Ix1, Iy1 = forward_diff_boundary(I1)

    Ix = 0.5 * (Ix0 + Ix1)
    Iy = 0.5 * (Iy0 + Iy1)

    It = I1 - I0
    return Ix, Iy, It


def build_rhs(Ix, Iy, It):
    return -It * Ix, -It * Iy


def laplacian5(u):
    n, m = u.shape
    L = np.zeros_like(u)
    L[1:n-1, 1:m-1] = (
        u[0:n-2, 1:m-1] + u[2:n, 1:m-1] +
        u[1:n-1, 0:m-2] + u[1:n-1, 2:m] - 4.0 * u[1:n-1, 1:m-1])
    return L

def apply_A(u, v, Ix, Iy, reg):
    Lu = laplacian5(u)
    Lv = laplacian5(v)
    Au = (Ix * Ix) * u + (Ix * Iy) * v - reg * Lu
    Av = (Ix * Iy) * u + (Iy * Iy) * v - reg * Lv
    return Au, Av


def of_cg(u0, v0, Ix, Iy, reg, rhsu, rhsv, tol=1e-8, maxit=2000):
    u = u0.copy()
    v = v0.copy()
    Au, Av = apply_A(u, v, Ix, Iy, reg)

    ru = rhsu - Au
    rv = rhsv - Av
    r2_0 = np.vdot(ru, ru) + np.vdot(rv, rv)

    pu = ru.copy()
    pv = rv.copy()
    r2_old = r2_0.copy()

    for it in range(maxit):
        Ap_u, Ap_v = apply_A(pu, pv, Ix, Iy, reg)

        alpha = r2_old / (np.vdot(pu, Ap_u) + np.vdot(pv, Ap_v))       # (r_k^T r_K) / (p_k^T A p_k)

        u += alpha * pu # update solution
        v += alpha * pv

        ru -= alpha * Ap_u  # residual
        rv -= alpha * Ap_v

        r2_new = np.vdot(ru, ru) + np.vdot(rv, rv)
        rel = r2_new / r2_0

        if rel < tol:
            return u, v, it, rel

        beta = r2_new / r2_old
        pu = ru + beta * pu
        pv = rv + beta * pv

        r2_old = r2_new

    return u, v, it, r2_old / r2_0
