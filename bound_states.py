"""Fig. 5: the two-glueball problem in the relative coordinate R, Eq. (48)."""

import numpy as np


C2 = 10/3  # <vac|B^2 (B†)^2|vac>, since B(B†)^2|vac> = (10/3) B†|vac>

def build_com_H(p, L):
    """Two-glueball Hamiltonian in the relative coordinate R, Eq. (48).

    Params:
        - p (float): centre-of-mass momentum K.
        - L (int): number of sites in R.
    Returns:
        - H (ndarray, shape (L, L)): real symmetric.
    """
    c = np.cos(p/2)
    t = np.ones(L-1); t[0] = np.sqrt(C2)
    H = -2*np.eye(L); H[1, 1] += -2/3
    off = -c*t
    return H + np.diag(off, 1) + np.diag(off, -1)

def bound_state_sweep(ps, ncav):
    """Spectrum of build_com_H over a set of centre-of-mass momenta, and the bound state at each.

    Params:
        - ps (array): centre-of-mass momenta K.
        - ncav (int): number of sites L in the relative coordinate R.
    Returns:
        - E (list of ndarray): the sorted spectrum at each K.
        - V_bound (list of ndarray): the lowest eigenvector, the bound state, at each K.
        - bound_state_E (ndarray): the lowest eigenvalue at each K.
    """
    E, V_bound = [], []
    for p in ps:
        H_com = build_com_H(p, ncav)
        e, V = np.linalg.eigh(H_com)
        sort = e.argsort()
        e = e[sort]
        V = V[:,sort]
        V_bound.append(V[:,0]) # pick bound state
        E.append(np.real(e))
    bound_state_E = np.transpose(E)[0]
    return E, V_bound, bound_state_E
