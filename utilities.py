import numpy as np
from scipy.ndimage import gaussian_filter

def forward_diff_boundary(I):
    dx = np.empty_like(I)
    dy = np.empty_like(I)

    dx[:, :-1] = I[:, 1:] - I[:, :-1]
    dx[:, -1]  = I[:, -1] - I[:, -2]

    dy[:-1, :] = I[1:, :] - I[:-1, :]
    dy[-1, :]  = I[-1, :] - I[-2, :]
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

def pad_to_odd(im):
    n, m = im.shape
    pad_n = 1 if n % 2 == 0 else 0
    pad_m = 1 if m % 2 == 0 else 0
    return np.pad(im, ((0,pad_n),(0,pad_m)), mode='constant')

def image_preprocess(I0, I1, sigma=0.0):
    # pad first to odd dimensions
    I0 = pad_to_odd(I0)
    I1 = pad_to_odd(I1)

    if sigma > 0.0:
        I0 = gaussian_filter(I0, sigma)
        I1 = gaussian_filter(I1, sigma)

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


def apply_A(u, v, Ix, Iy, reg, level=0):
    h2inv = 4.0 ** (-level)
    Lu = laplacian5(u)
    Lv = laplacian5(v)

    Ix2  = Ix * Ix
    Iy2  = Iy * Iy
    IxIy = Ix * Iy

    Au = Ix2 * u + IxIy * v - reg * (h2inv * Lu)
    Av = IxIy * u + Iy2 * v - reg * (h2inv * Lv)

    return zero_boundary(Au), zero_boundary(Av)  #Enforce zero BCs


def compute_gradients_full(Im0, Im1):
    Im0 = Im0.astype(float)
    Im1 = Im1.astype(float)

    # Spatial derivatives (centered differences)
    Ix = np.zeros_like(Im0)
    Iy = np.zeros_like(Im0)

    Ix[:, 1:-1] = 0.5 * (Im0[:, 2:] - Im0[:, :-2])
    Iy[1:-1, :] = 0.5 * (Im0[2:, :] - Im0[:-2, :])

    # Temporal derivative
    It = Im1 - Im0

    return Ix, Iy, It
