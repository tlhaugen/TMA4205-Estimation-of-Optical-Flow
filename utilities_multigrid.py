import numpy as np
from utilities import zero_boundary, apply_A


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
    eps = 1e-10

    # level scaling
    h2inv = 4.0**(-level)      # 1/h^2 with h = 2^level
    gamma = reg * h2inv        #multiplier for laplacian term

    ir = range(1, n-1); jr = range(1, m-1)
    sweeps = max(1, s1 - int(level))
    a = Ix * Ix + 4.0 * gamma
    b = Ix * Iy
    c = Iy * Iy + 4.0 * gamma
    det = a * c - b * b
    det[np.abs(det) < eps] = eps #clamp to avoid division by zero

    for _ in range(sweeps):
        for p in (parity, 1 - parity):
            mask = np.zeros_like(u, dtype=bool)
            mask[1:-1, 1:-1] = ((np.add.outer(np.arange(1, n-1), np.arange(1, m-1))) & 1) == p #mask to select red or black points

            # neighbor sums 
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

    n_f, m_f = rhu.shape
    n_c = n_f // 2 + 1
    m_c = m_f // 2 + 1


    #slices for the 3×3 stencil around coarse centers (2i, 2j)
    i_center = slice(2, n_f-1, 2)   # 2,4,...,n_f-3
    j_center = slice(2, m_f-1, 2)

    i_up     = slice(1, n_f-2, 2)   # 1,3,...,n_f-4
    i_down   = slice(3, n_f,   2)   # 3,5,...,n_f-1

    j_left   = slice(1, m_f-2, 2)
    j_right  = slice(3, m_f,   2)

    def restrict_one(rf):
        r2 = np.zeros((n_c, m_c))
        c = r2[1:-1, 1:-1]  # coarse interior

        c[:] = (
            4 * rf[i_center, j_center] +
            2 * (
                rf[i_up,   j_center] + rf[i_down, j_center] +
                rf[i_center, j_left] + rf[i_center, j_right]
            ) +
            (
                rf[i_up,   j_left] + rf[i_up,   j_right] +
                rf[i_down, j_left] + rf[i_down, j_right]
            )
        ) / 16.0
        return r2

    r2hu = restrict_one(rhu)
    r2hv = restrict_one(rhv)
    Ix2h = restrict_one(Ix)
    Iy2h = restrict_one(Iy)

    return r2hu, r2hv, Ix2h, Iy2h



def prolongation(e2hu, e2hv):
    n_c, m_c = e2hu.shape
    n_f = 2*(n_c - 1) + 1
    m_f = 2*(m_c - 1) + 1


    ehu = np.zeros((n_f, m_f))
    ehv = np.zeros((n_f,m_f))
    # inject coarse nodes
    ehu[0::2, 0::2] = e2hu
    ehv[0::2, 0::2] = e2hv
    # interpolate edges
    ehu[1::2, 0::2] = 0.5*(ehu[:-2:2, 0::2] + ehu[2::2, 0::2])
    ehu[0::2, 1::2] = 0.5*(ehu[0::2, :-2:2] + ehu[0::2, 2::2])

    ehv[1::2, 0::2] = 0.5*(ehv[:-2:2, 0::2] + ehv[2::2, 0::2])
    ehv[0::2, 1::2] = 0.5*(ehv[0::2, :-2:2] + ehv[0::2, 2::2])
    # interpolate centers
    ehu[1::2, 1::2] = 0.25*(
        ehu[:-2:2, :-2:2] + ehu[:-2:2, 2::2] +
        ehu[2::2,  :-2:2] + ehu[2::2,  2::2]
    )
    ehv[1::2, 1::2] = 0.25*(
        ehv[:-2:2, :-2:2] + ehv[:-2:2, 2::2] +
        ehv[2::2,  :-2:2] + ehv[2::2,  2::2]
    )    

    return ehu,ehv
