import numpy as np
from scipy.ndimage import gaussian_filter
#from V_cycles import V_cycle


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


def image_preprocess(I0, I1, sigma=0.0):
    if sigma > 0.0:
        I0 = smooth(I0, sigma)
        I1 = smooth(I1, sigma)

    Ix, Iy, It = compute_derivatives(I0, I1)
    rhsu, rhsv = build_rhs(Ix, Iy, It)

    u0 = np.zeros_like(I0)
    v0 = np.zeros_like(I0)

    return u0, v0, Ix, Iy, rhsu, rhsv, I0, I1


def laplacian5(u):
    n, m = u.shape
    L = np.zeros_like(u)
    L[1:n-1, 1:m-1] = (
        u[0:n-2, 1:m-1] + u[2:n, 1:m-1] +
        u[1:n-1, 0:m-2] + u[1:n-1, 2:m] - 4.0 * u[1:n-1, 1:m-1])
    return L

def zero_boundary(a):
    a[0, :] = a[-1, :] = 0.0
    a[:, 0] = a[:, -1] = 0.0
    return a

def apply_A(u, v, Ix, Iy, reg):
    Lu = laplacian5(u)
    Lv = laplacian5(v)
    Au = (Ix * Ix) * u + (Ix * Iy) * v - reg * Lu
    Av = (Ix * Iy) * u + (Iy * Iy) * v - reg * Lv
    return zero_boundary(Au), zero_boundary(Av) #Enforce zero BCs


def of_cg(u0, v0, Ix, Iy, reg, rhsu, rhsv, tol=1e-8, maxit=2000):
    u = zero_boundary(u0.copy())
    v = zero_boundary(v0.copy())
    Au, Av = apply_A(u, v, Ix, Iy, reg) # Initial residual

    ru = zero_boundary(rhsu.copy()) - Au
    rv = zero_boundary(rhsv.copy()) - Av
    r2_0 = np.vdot(ru, ru) + np.vdot(rv, rv)

    pu = ru.copy()
    pv = rv.copy()
    r2_old = r2_0.copy()
    res_hist = np.zeros(maxit)

    for it in range(maxit):
        Ap_u, Ap_v = apply_A(pu, pv, Ix, Iy, reg)

        alpha = r2_old / (np.vdot(pu, Ap_u) + np.vdot(pv, Ap_v)) # (r_k^T r_K) / (p_k^T A p_k)

        u += alpha * pu # update solution
        v += alpha * pv

        ru -= alpha * Ap_u  # residual
        rv -= alpha * Ap_v

        r2_new = np.vdot(ru, ru) + np.vdot(rv, rv)
        rel = np.sqrt(r2_new) / np.sqrt(r2_0)

        if rel < tol:
            return u, v, it, rel, res_hist

        beta = r2_new / r2_old
        pu = ru + beta * pu
        pv = rv + beta * pv

        r2_old = r2_new
        res_hist[it] = rel

    return u, v, it, rel, res_hist


# def run_pcg(u0, v0, Ix, Iy, reg, rhsu, rhsv, tol=1e-8, maxit=2000):
#     '''
#     Solve the optical flow problem using Preconditioned Conjugate Gradient (PCG).
#     '''
#     pu,pv = V_cycle(u0, v0, Ix, Iy, reg, rhsu, rhsv, s1=2, s2=2, level=0, max_level=3)
#     u, v, it, relres, _ = of_cg(pu, pv, Ix, Iy, reg, rhsu, rhsv, tol, maxit)
#     return u, v, it, relres

