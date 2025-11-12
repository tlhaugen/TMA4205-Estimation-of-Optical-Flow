import numpy as np
from utilities import zero_boundary, apply_A
from scipy.sparse import diags

def smoothing(u0, v0, Ix, Iy, reg, rhsu, rhsv, s1, level, parity=0):
    """
    Simple red-black Gauss-Seidel smoother (pointwise 2x2 solves).
    - u0, v0: full-grid arrays including Dirichlet border
    - Ix, Iy: coefficient fields (full-grid)
    - reg: regularisation parameter
    - rhsu, rhsv: right-hand sides (full-grid)
    - level: unused (kept for API compatibility)
    - s1: number of RB-GS sweeps
    - parity: choose red-black ordering starting with red(0) or black(1)
    Returns the smoothed u, v fields.
    """
    u = u0.copy(); v = v0.copy()
    n, m = u.shape
    eps = 1e-12

    # level scaling
    h2inv = 4.0**(-level)      # 1/h^2 with h = 2^level
    gamma = reg * h2inv        # this multiplies Laplacian terms

    ir = range(1, n-1); jr = range(1, m-1)
    sweeps = max(1, s1 - int(level))
    a = Ix * Ix + 4.0 * gamma
    b = Ix * Iy
    c = Iy * Iy + 4.0 * gamma
    det = a * c - b * b
    det[np.abs(det) < eps] = eps

    for _ in range(sweeps):
        for p in (parity, 1 - parity):
            mask = np.zeros_like(u, dtype=bool)
            mask[1:-1, 1:-1] = ((np.add.outer(np.arange(1, n-1), np.arange(1, m-1))) & 1) == p

            # neighbor sums (Dirichlet border assumed fixed; border values are present in u/v)
            Su = (u[:-2, 1:-1] + u[2:, 1:-1] +
                  u[1:-1, :-2] + u[1:-1, 2:])
            Sv = (v[:-2, 1:-1] + v[2:, 1:-1] +
                  v[1:-1, :-2] + v[1:-1, 2:])

            # local RHS (move laplacian neighbour terms to RHS)
            fu = rhsu[1:-1, 1:-1] + gamma * Su
            fv = rhsv[1:-1, 1:-1] + gamma * Sv

            # local 2x2 block coefficients
            a_loc = a[1:-1, 1:-1]
            b_loc = b[1:-1, 1:-1]
            c_loc = c[1:-1, 1:-1]
            det_loc = det[1:-1, 1:-1]

            u_new = (c_loc * fu - b_loc * fv) / det_loc
            v_new = (-b_loc * fu + a_loc * fv) / det_loc

            u[1:-1, 1:-1][mask[1:-1, 1:-1]] = u_new[mask[1:-1, 1:-1]]
            v[1:-1, 1:-1][mask[1:-1, 1:-1]] = v_new[mask[1:-1, 1:-1]]

    u = zero_boundary(u)
    v = zero_boundary(v)
    return u, v


def  residual(u, v, Ix, Iy, reg, rhsu, rhsv, level=0):
    '''
    Compute the residual r = b - A*x for the optical flow problem.  
    '''
    Au, Av = apply_A(u, v, Ix, Iy, reg, level=level)
    rhu = zero_boundary(rhsu.copy()) - Au    
    rhv = zero_boundary(rhsv.copy()) - Av
    return rhu, rhv


def restriction(rhu, rhv, Ix, Iy):
    '''Create 1D restriction operators with stencil [1/4, 1/2, 1/4] 
    and apply them to residuals and image derivatives for coarsening.'''
    n, m = rhu.shape
    print(f"{rhu.shape}")  # print dimensions even if we abort
    # Compute coarse grid dimensions (include boundary points)
    interior_n = n - 2  # fine interior points (exclude boundaries)
    interior_m = m - 2
    interior_coarse_n = (interior_n + 1) // 2  # half of interior (rounding up)
    interior_coarse_m = (interior_m + 1) // 2
    n_coarse = interior_coarse_n + 2  # add boundary points
    m_coarse = interior_coarse_m + 2

    # Create 1D restriction operators in y and x directions
    R1dy = diags(
        [0.25 * np.ones(n_coarse - 1), 0.5 * np.ones(n_coarse), 0.25 * np.ones(n_coarse)],
        offsets=[-1, 0, 1], shape=(n_coarse, n)
    ).tocsr()
    R1dx = diags(
        [0.25 * np.ones(m_coarse - 1), 0.5 * np.ones(m_coarse), 0.25 * np.ones(m_coarse)],
        offsets=[-1, 0, 1], shape=(m_coarse, m)
    ).tocsr()

    # Apply restriction (first in y, then in x)
    r2hu = R1dy @ rhu @ R1dx.T
    r2hv = R1dy @ rhv @ R1dx.T
    Ix2h = R1dy @ Ix  @ R1dx.T
    Iy2h = R1dy @ Iy  @ R1dx.T

    return r2hu, r2hv, Ix2h, Iy2h


def prolongation(e2hu, e2hv):
    '''Implement prolongation by linear interpolation (bilinear for 2D).'''
    n_coarse, m_coarse = e2hu.shape  # coarse grid shape (including boundaries)
    # Compute fine grid dimensions from coarse (include boundary points)
    n_fine = 2 * (n_coarse - 2) +1
    if n_fine // 2 != 0:
        n_fine += 1      # because fine interior ≈ 2 * coarse interior
    m_fine = 2 * (m_coarse - 2) +1
    if m_fine // 2 != 0:
        m_fine += 1 
    print(f"{e2hu.shape} -> ({n_fine},{m_fine})")  # print dimensions even if we abort
    # Create 1D prolongation operators in y and x directions
    P1dy = 2 * diags(
        [0.5 * np.ones(n_coarse), 1.0 * np.ones(n_coarse), 0.5 * np.ones(n_coarse - 1)],
        offsets=[-1, 0, 1], shape=(n_fine, n_coarse)
    ).tocsr()
    P1dx = 2 * diags(
        [0.5 * np.ones(m_coarse), 1.0 * np.ones(m_coarse), 0.5 * np.ones(m_coarse - 1)],
        offsets=[-1, 0, 1], shape=(m_fine, m_coarse)
    ).tocsr()

    # Apply prolongation (first in y, then in x)
    ehu = P1dy @ e2hu @ P1dx.T
    ehv = P1dy @ e2hv @ P1dx.T
    return ehu, ehv


# def restriction(rhu, rhv, Ix, Iy):
#     ''' cread 1d restriction matrix with stencil [1/4,1/2,1/4] 
#     and apply it to the residuals and the image derivatives
#     '''
#     n,m = rhu.shape
#     n_coarse = (n + 1) // 2  # coarse rows (y-direction)
#     m_coarse = (m + 1) // 2  # coarse columns (x-direction)
#     #create 1D restriction operators in x and y direction
#     main_diag = np.ones(n_coarse) * 0.5
#     off_diag = np.ones(n_coarse - 1) * 0.25
#     R1dy = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n_coarse, n))
#     R1dy = R1dy.tocsr()
#     main_diag = np.ones(m_coarse) * 0.5
#     off_diag = np.ones(m_coarse - 1) * 0.25
#     R1dx = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(m_coarse, m))
#     R1dx = R1dx.tocsr()
    
#     #apply restriction
#     r2hu = R1dy @ rhu @ R1dx.T
#     r2hv = R1dy @ rhv @ R1dx.T
#     Ix2h = R1dy @ Ix @ R1dx.T
#     Iy2h = R1dy @ Iy @ R1dx.T

#     return r2hu, r2hv, Ix2h, Iy2h


# def prolongation(rhu, rhv, Ix, Iy):
#     ''' implement prolongation by linear interpolation'''
#     n,m = rhu.shape
#     n_fine = 2 * (n - 1) + 1  # fine rows (y-direction)
#     m_fine = 2 * (m - 1) + 1  # fine columns (x-direction)
#     #create 1D prolongation operators in x and y direction
#     P1dy = diags([0.5 * np.ones(n - 1), np.ones(n), 0.5 * np.ones(n - 1)], [-1, 0, 1], shape=(n_fine, n))
#     P1dy = P1dy.tocsr()
#     P1dx = diags([0.5 * np.ones(m - 1), np.ones(m), 0.5 * np.ones(m - 1)], [-1, 0, 1], shape=(m_fine, m))
#     P1dx = P1dx.tocsr()
#     #apply prolongation
#     ehu = P1dy @ rhu @ P1dx.T
#     ehv = P1dy @ rhv @ P1dx.T

#     return ehu, ehv