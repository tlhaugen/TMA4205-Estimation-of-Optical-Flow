import numpy as np


def forward_diff_x_boundary(I):
    n, m = I.shape
    dx = np.zeros_like(I)
    dx[:, :m-1] = I[:, 1:] - I[:, :-1]
    dx[:, m-1] = I[:, m-1] - I[:, m-2]
    return dx

def forward_diff_y_boundary(I):
    n, m = I.shape
    dy = np.zeros_like(I)
    dy[:n-1, :] = I[1:, :] - I[:-1, :]
    dy[n-1, :] = I[n-1, :] - I[n-2, :]
    return dy

def compute_derivatives(I0, I1):
    Ix = 0.5 * (forward_diff_x_boundary(I0) + forward_diff_x_boundary(I1))
    Iy = 0.5 * (forward_diff_y_boundary(I0) + forward_diff_y_boundary(I1))
    It = I1 - I0
    return Ix, Iy, It

def maybe_smooth(I, sigma):
    if sigma and sigma > 0.0:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(I, sigma=sigma)
    return I

def compute_derivatives_blur(I0, I1, sigma):
    I0b = maybe_smooth(I0, sigma)
    I1b = maybe_smooth(I1, sigma)
    return compute_derivatives(I0b, I1b)

def build_rhs(Ix, Iy, It):
    return -It * Ix, -It * Iy


def laplacian5_dirichlet(u):
    n, m = u.shape
    L = np.zeros_like(u)
    L[1:n-1, 1:m-1] = (
        u[0:n-2, 1:m-1] + u[2:n, 1:m-1] +
        u[1:n-1, 0:m-2] + u[1:n-1, 2:m] -
        4.0 * u[1:n-1, 1:m-1]
    )
    L[0, :] = 0.0; L[-1, :] = 0.0; L[:, 0] = 0.0; L[:, -1] = 0.0
    return L

def apply_A(u, v, Ix, Iy, reg):
    Lu = laplacian5_dirichlet(u)
    Lv = laplacian5_dirichlet(v)
    Au = (Ix * Ix) * u + (Ix * Iy) * v - reg * Lu
    Av = (Ix * Iy) * u + (Iy * Iy) * v - reg * Lv
 
    Au[0, :] = 0.0; Au[-1, :] = 0.0; Au[:, 0] = 0.0; Au[:, -1] = 0.0
    Av[0, :] = 0.0; Av[-1, :] = 0.0; Av[:, 0] = 0.0; Av[:, -1] = 0.0
    return Au, Av

def enforce_bc(u, v):
    u[0, :] = 0.0; u[-1, :] = 0.0; u[:, 0] = 0.0; u[:, -1] = 0.0
    v[0, :] = 0.0; v[-1, :] = 0.0; v[:, 0] = 0.0; v[:, -1] = 0.0


def of_cg(u0, v0, Ix, Iy, reg, rhsu, rhsv, tol=1e-8, maxit=2000, callback=None):
    u = u0.copy(); v = v0.copy()

    # initial residual
    Au, Av = apply_A(u, v, Ix, Iy, reg)
    ru = rhsu - Au
    rv = rhsv - Av
    r2_0 = float(np.sum(ru * ru + rv * rv))
    if r2_0 == 0.0:
        enforce_bc(u, v)
        return u, v, 0, 0.0

    pu = ru.copy(); pv = rv.copy()
    enforce_bc(pu, pv)

    r2_old = r2_0
    for it in range(1, maxit + 1):
        Apu, Apv = apply_A(pu, pv, Ix, Iy, reg)
        denom = float(np.sum(pu * Apu + pv * Apv))
        if denom <= 0.0:
            break

        alpha = r2_old / denom

        # update and enforce Dirichlet on the solution
        u += alpha * pu
        v += alpha * pv
        enforce_bc(u, v)

        # residual
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
        enforce_bc(pu, pv)  # keep the border zero

        r2_old = r2_new

    return u, v, it, r2_old / r2_0
