import numpy as np
from utilities import zero_boundary, apply_A
from scipy.sparse import diags

def smoothing(u0, v0, Ix, Iy, reg, rhsu, rhsv, level, s1, parity= 0):
    """
    Simple red-black Gauss-Seidel smoother (pointwise 2x2 solves).
    - u0, v0: full-grid arrays including Dirichlet border
    - Ix, Iy: coefficient fields (full-grid)
    - reg: regularisation parameter
    - rhsu, rhsv: right-hand sides (full-grid)
    - level: unused (kept for API compatibility)
    - s1: number of RB-GS sweeps
    -parity: choose red-black ordering starting with red(0) or black(1)
    Returns the smoothed u, v fields.

    """
    u = u0.copy()
    v = v0.copy()
    n, m = u.shape
  
    eps = 1e-10  # to avoid division by zero
    ir = range(1, n - 1)
    jr = range(1, m - 1)
    sweeps = max(1, s1 - int(level)) #calculate how many sweeps based on level
    for _ in range(sweeps):
        for p in (parity, 1-parity): #update order
            for i in ir:
                for j in jr:
                    if ((i + j) & 1) != p:
                        continue
                    Ixij = float(Ix[i, j])
                    Iyij = float(Iy[i, j])

                    # local 2x2 block coefficients
                    a = Ixij * Ixij + 4.0 * reg
                    b = Ixij * Iyij
                    c = Iyij * Iyij + 4.0 * reg

                    # neighbor sums (Dirichlet border assumed fixed; border values are present in u/v)
                    Su = u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1]
                    Sv = v[i-1, j] + v[i+1, j] + v[i, j-1] + v[i, j+1]

                    # local RHS (move laplacian neighbour terms to RHS)
                    fu = float(rhsu[i, j]) + reg * Su
                    fv = float(rhsv[i, j]) + reg * Sv

                    det = a * c - b * b
                    if abs(det) < eps:
                        # fallback to diagonal (decoupled) update
                        u_new = fu / (a + eps)
                        v_new = fv / (c + eps)
                    else:
                        u_new = (c * fu - b * fv) / det
                        v_new = (-b * fu + a * fv) / det

                    u[i, j] = u_new
                    v[i, j] = v_new

    # enforce Dirichlet border
    u = zero_boundary(u)
    v = zero_boundary(v)
    return u, v


def  residual(u, v, Ix, Iy, reg, rhsu, rhsv):
    '''
    Compute the residual r = b - A*x for the optical flow problem.  
    '''
    Au, Av = apply_A(u, v, Ix, Iy, reg)
    rhu = rhsu - Au    
    rhv = rhsv - Av
    return rhu, rhv

def restriction(rhu, rhv, Ix, Iy):
    ''' cread 1d restriction matrix with stencil [1/4,1/2,1/4] 
    and apply it to the residuals and the image derivatives
    '''
    n,m = rhu.shape
    n_coarse = (n + 1) // 2  # coarse rows (y-direction)
    m_coarse = (m + 1) // 2  # coarse columns (x-direction)
    #create 1D restriction operators in x and y direction
    R1dy = diags([0.25*np.ones(n_coarse-1), 0.5*np.ones(n_coarse), 0.25*np.ones(n_coarse)], offsets=[-1, 0, +1], shape=(n_coarse, n)).tocsr()

    R1dx = diags([0.25*np.ones(m_coarse-1), 0.5*np.ones(m_coarse), 0.25*np.ones(m_coarse)], offsets=[-1, 0, +1], shape=(m_coarse, m)).tocsr()
    
    #apply restriction
    r2hu = R1dy @ rhu @ R1dx.T
    r2hv = R1dy @ rhv @ R1dx.T
    Ix2h = R1dy @ Ix @ R1dx.T
    Iy2h = R1dy @ Iy @ R1dx.T

    return r2hu, r2hv, Ix2h, Iy2h


def prolongation_sparse(e2hu, e2hv):
    ''' implement prolongation by linear interpolation'''
    n_coarse, m_coarse = e2hu.shape
    n_fine = 2*(n_coarse-1) + 1 # fine rows (y-direction)
    m_fine = 2*(m_coarse-1) + 1 # fine columns (x-direction)

    #create 1D prolongation operators in x and y direction
    P1dy = diags([0.5*np.ones(n_coarse), 1.0*np.ones(n_coarse), 0.5*np.ones(n_coarse-1)], offsets=[-1, 0, +1], shape=(n_fine, n_coarse)).tocsr()
    P1dx = diags([0.5*np.ones(m_coarse), 1.0*np.ones(m_coarse), 0.5*np.ones(m_coarse-1)], offsets=[-1, 0, +1], shape=(m_fine, m_coarse)).tocsr()

    ehu = P1dy @ e2hu @ P1dx.T
    ehv = P1dy @ e2hv @ P1dx.T
    return ehu, ehv

