
from utilities import *
from scipy.sparse import diags

def smoothing(u0, v0, Ix, Iy, reg, rhsu, rhsv, level, s1):
    """
    Simple red-black Gauss-Seidel smoother (pointwise 2x2 solves).
    - u0, v0: full-grid arrays including Dirichlet border
    - Ix, Iy: coefficient fields (full-grid)
    - reg: regularisation parameter
    - rhsu, rhsv: right-hand sides (full-grid)
    - level: unused (kept for API compatibility)
    - n_iters: number of RB-GS sweeps
    """
    u = u0.copy()
    v = v0.copy()
    n, m = u.shape
  
    eps = 1e-10  # to avoid division by zero
    ir = range(1, n - 1)
    jr = range(1, m - 1)
    sweeps = max(1, s1 - int(level)) #calculate how many sweeps based on level
    for _ in range(sweeps):
        for parity in (0, 1):  # red (0) then black (1)
            for i in ir:
                for j in jr:
                    if ((i + j) & 1) != parity:
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

    # enforce Dirichlet border exactly
    u,v = enforce_bc(u, v)
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
    main_diag = np.ones(n_coarse) * 0.5
    off_diag = np.ones(n_coarse - 1) * 0.25
    R1dy = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n_coarse, n))
    R1dy = R1dy.tocsr()
    main_diag = np.ones(m_coarse) * 0.5
    off_diag = np.ones(m_coarse - 1) * 0.25
    R1dx = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(m_coarse, m))
    R1dx = R1dx.tocsr()
    
    #apply restriction
    r2hu = R1dy @ rhu @ R1dx.T
    r2hv = R1dy @ rhv @ R1dx.T
    Ix2h = R1dy @ Ix @ R1dx.T
    Iy2h = R1dy @ Iy @ R1dx.T

    return r2hu, r2hv, Ix2h, Iy2h


def prolongation(rhu, rhv, Ix, Iy):
    ''' implement prolongation by linear interpolation'''
    n,m = rhu.shape
    n_fine = 2 * (n - 1) + 1  # fine rows (y-direction)
    m_fine = 2 * (m - 1) + 1  # fine columns (x-direction)
    #create 1D prolongation operators in x and y direction
    P1dy = diags([0.5 * np.ones(n - 1), np.ones(n), 0.5 * np.ones(n - 1)], [-1, 0, 1], shape=(n_fine, n))
    P1dy = P1dy.tocsr()
    P1dx = diags([0.5 * np.ones(m - 1), np.ones(m), 0.5 * np.ones(m - 1)], [-1, 0, 1], shape=(m_fine, m))
    P1dx = P1dx.tocsr()
    #apply prolongation
    ehu = P1dy @ rhu @ P1dx.T
    ehv = P1dy @ rhv @ P1dx.T

    return ehu, ehv


def V_cycle(u0, v0, Ix, Iy, reg, rhsu, rhsv, s1, s2, level, max_level):
    '''
    V-cycle for the optical flow problem.
    input:
    u0 - initial guess for u
    v0 - initial guess for v
    Ix - x-derivative of the first frame
    Iy - y-derivative of the first frame
    reg - regularisation parameter (lambda)
    rhsu - right-hand side in the equation for u
    rhsv - right-hand side in the equation for v
    s1 - number of pre-smoothings
    s2 - number of post-smoothings
    level - current level
    max_level - total number of levels
    output:
    u - numerical solution for u
    v - numerical solution for v
    '''

    u,v = smoothing(u0, v0, Ix, Iy, reg, rhsu, rhsv, level,s1)

    rhu,rhv = residual(u, v, Ix, Iy, reg, rhsu, rhsv)
    r2hu,r2hv,Ix2h,Iy2h = restriction(rhu, rhv, Ix, Iy)
    if level == max_level - 1:
        e2hu,e2hv = of_cg(np.zeros_like(r2hu), np.zeros_like(r2hv),
        Ix2h, Iy2h, reg, rhsu, rhsv, 1e-8, 1000, level+1)
    else:
        e2hu,e2hv = V_cycle(np.zeros_like(r2hu), np.zeros_like(r2hv),
        Ix2h, Iy2h, reg, r2hu, r2hv, s1, s2, level+1, max_level)
    ehu,ehv = prolongation(e2hu, e2hv)
    u = u + ehu
    v = v + ehv
    u,v = smoothing(u, v, Ix, Iy, reg, rhsu, rhsv, level, s2)
    return u, v


