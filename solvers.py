import numpy as np

from utilities import zero_boundary, apply_A
from utilities_multigrid import prolongation, restriction, residual, smoothing

def of_cg(u0, v0, Ix, Iy, reg, rhsu, rhsv, tol=1e-8, maxit=2000, level=0):
    u = zero_boundary(u0.copy())
    v = zero_boundary(v0.copy())
    Au, Av = apply_A(u, v, Ix, Iy, reg, level=level) # Initial residual
    it = 0

    ru = zero_boundary(rhsu.copy()) - Au
    rv = zero_boundary(rhsv.copy()) - Av
    r2_0 = np.vdot(ru, ru) + np.vdot(rv, rv)
    r_0 = np.sqrt(r2_0)

    pu = ru.copy()
    pv = rv.copy()
    r2_old = r2_0
    res_hist = [1.0]
    rel = 1.0
    dot = np.vdot

    while it < maxit and rel > tol:
        Ap_u, Ap_v = apply_A(pu, pv, Ix, Iy, reg, level)

        alpha = r2_old / (dot(pu, Ap_u) + dot(pv, Ap_v)) # (r_k^T r_K) / (p_k^T A p_k)

        u += alpha * pu # update solution
        v += alpha * pv

        ru -= alpha * Ap_u  # residual
        rv -= alpha * Ap_v

        r2_new = dot(ru, ru) + dot(rv, rv)
        rel = np.sqrt(r2_new) / r_0


        beta = r2_new / r2_old
        pu = ru + beta * pu
        pv = rv + beta * pv

        r2_old = r2_new
        res_hist.append(rel)
        it += 1
    return u, v, it, rel, res_hist


def V_cycle(u0, v0, Ix, Iy, reg, rhsu, rhsv, s1, s2, level, max_level):
    '''
    V-cycle for the optical flow problem. 
    keeps the SPD structure of the problem.
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

    u,v = smoothing(u0, v0, Ix, Iy, reg, rhsu, rhsv, s1, level=level, parity=0)
    rhu,rhv = residual(u, v, Ix, Iy, reg, rhsu, rhsv, level=level)

    if (level == max_level - 1): 
        ehu, ehv, *_ = of_cg(np.zeros_like(rhu), np.zeros_like(rhv), Ix, Iy, reg, rhu, rhv, level=level)

        u += ehu
        v += ehv

    else:
        r2hu,r2hv,Ix2h,Iy2h = restriction(rhu, rhv, Ix, Iy)
        e2hu,e2hv = V_cycle(np.zeros_like(r2hu), np.zeros_like(r2hv), Ix2h, Iy2h, reg, r2hu, r2hv, s1, s2, level+1, max_level)

        ehu, ehv = prolongation(e2hu, e2hv)
        u += ehu
        v += ehv

    u,v = smoothing(u, v, Ix, Iy, reg, rhsu, rhsv, s2, level=level, parity=1)
    return u, v

def of_vc(u0, v0, Ix, Iy, reg, rhsu, rhsv, s1=2, s2=2, max_level=4, tol=1e-8, maxit=2000):
    u, v = u0.copy(), v0.copy()
    rhu, rhv = residual(u, v, Ix, Iy, reg, rhsu, rhsv, level=0)
    r2_0 = np.vdot(rhu, rhu) + np.vdot(rhv, rhv)
    if r2_0 == 0.0:
        return u, v, 0, 0.0, [0.0]


    it = 0
    rel = 1
    res_hist = [1.0]

    while it < maxit and rel > tol:
        u, v = V_cycle(u, v, Ix, Iy, reg, rhsu, rhsv, s1, s2, level=0, max_level=max_level)
        rhu, rhv = residual(u, v, Ix, Iy, reg, rhsu, rhsv, level=0)
        r2_new = np.vdot(rhu, rhu) + np.vdot(rhv, rhv)
        rel = np.sqrt(r2_new) / np.sqrt(r2_0)
        res_hist.append(rel)
        it += 1

    return u, v, it, rel, res_hist



def run_pcg(u0, v0, Ix, Iy, reg, rhsu, rhsv, tol=1e-8, maxit=2000):
    '''
    Solve the optical flow problem using Preconditioned Conjugate Gradient (PCG).
    '''
    pu,pv = V_cycle(u0, v0, Ix, Iy, reg, rhsu, rhsv, s1=2, s2=2, level=0, max_level=3)
    u, v, it, relres, res_history = of_cg(pu, pv, Ix, Iy, reg, rhsu, rhsv, tol, maxit)
    return u, v, it, relres, res_history