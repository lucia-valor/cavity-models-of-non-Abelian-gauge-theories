"""Figs. 3-4: a single driven cavity in the (N, l, m = 0) basis, and the effective glueball model."""

import numpy as np
import scipy.linalg as sl
from math import factorial


# ---------------------------------------------------------------- basis and driven Hamiltonian

def construct_basis_m0(Nmax, lmax):
    """Basis of the m = 0 subspace.

    Params:
        - Nmax, lmax (int): max. photon number N and angular momentum l retained.
    Returns:
        - basis (ndarray, shape (D, 2)): basis[q] = (N, l), with N <= Nmax and l <= lmax.
        - ibasis (dict): inverse map, ibasis[N, l] = q.
    """
    basis = []
    ibasis = {}
    q = 0
    for l in range(lmax+1):
        for N in range(l, Nmax+1, 2):
                basis.append( (N, l) )
                ibasis[N, l] = q
                q += 1
    return np.array(basis), ibasis

def vac(Nmax, lmax):
    """Vacuum state in the m = 0 basis.

    Params:
        - Nmax, lmax (int): basis truncation.
    Returns:
        - vac (ndarray, shape (D,), complex): unit amplitude on |0,0>.
    """
    basis, ibasis = construct_basis_m0(Nmax, lmax)
    D = len(basis)
    vac = np.zeros(D, dtype=complex)
    vac[ibasis[0,0]] = 1
    return vac

def betas(N,l,m):
    """Matrix elements of a3^dag out of |N,l,m>.

    Params:
        - N, l, m (int): quantum numbers of the initial state.
    Returns:
        - <N+1,l+1,m|a3^dag|N,l,m>, <N+1,l-1,m|a3^dag|N,l,m> (float).
    """
    return (-np.sqrt( (N+l+3)*(l+m+1)*(l-m+1)/((2*l+3)*(2*l+1)) ),
            -np.sqrt( (N-l+2)*(l+m)*(l-m)/((2*l+1)*(2*l-1)) ))

def H_drive(epsilon, Delta, alpha, Nmax, lmax):
    """Driven cavity Hamiltonian H = Delta N + alpha J^2 + epsilon (a3 + a3dag).

    Params:
        - epsilon, Delta, alpha (float): drive amplitude, detuning, interaction U.
        - Nmax, lmax (int): basis truncation.
    Returns:
        - H (ndarray, shape (D, D)): real symmetric, in the m = 0 basis.
    """
    basis, ibasis = construct_basis_m0(Nmax, lmax)
    D = len(basis)
    H = np.zeros((D,D))
    for N, l in basis:
        ## Diagonal terms
        H[ibasis[N,l],ibasis[N,l]] += Delta*N + alpha*l*(l+1)
        betaP, betaM = betas(N, l, 0)
        ## Drive term epsilon a3dag
        if (N+1,l+1) in ibasis:
            H[ibasis[N+1,l+1], ibasis[N,l]] += epsilon*betaP
            H[ibasis[N,l], ibasis[N+1,l+1]] += epsilon*betaP
        if (N+1,l-1) in ibasis:
            H[ibasis[N+1,l-1], ibasis[N,l]] += epsilon*betaM
            H[ibasis[N,l], ibasis[N+1,l-1]] += epsilon*betaM
    return H

def evol_list(lt, E, psi_eig, psi0):
    """Time evolution of a state, from the eigendecomposition of the Hamiltonian.

    Params:
        - lt (array): times.
        - E, psi_eig (ndarray): eigenvalues and eigenvectors (columns), from np.linalg.eigh.
        - psi0 (ndarray): initial state, e.g. vac(Nmax, lmax).
    Returns:
        - psi_t (ndarray, shape (len(lt), D)): the state at each time.
    """
    Et = np.einsum("k,t->kt", E, lt)
    return np.einsum("kt,nk,mk,m->tn", np.exp(-1j*Et), psi_eig, np.conj(psi_eig), psi0, optimize=True)


# ---------------------------------------------------------------- observables in the exact (N, l) basis

def ng_distr(state, Nmax, lmax):
    """Glueball number distribution, P(𝒩) = sum_{N,l} delta_{N,l+2𝒩} |<N,l|psi>|^2.

    Params:
        - state (ndarray): state in the m = 0 basis.
        - Nmax, lmax (int): basis truncation.
    Returns:
        - P (ndarray, shape (Nmax//2 + 1,)).
    """
    basis, ibasis = construct_basis_m0(Nmax, lmax)
    P = np.zeros(Nmax//2+1)
    for N, l in basis:
        ng = (N-l)//2
        P[ng] += np.abs(state[ibasis[N,l]])**2
    return P

def ng_avg(state, Nmax, lmax):
    """Average glueball number <𝒩> in a state.

    Params:
        - state (ndarray): state in the m = 0 basis.
        - Nmax, lmax (int): basis truncation.
    Returns:
        - <𝒩> (float).
    """
    ng_list = np.arange(Nmax//2+1)
    return np.sum( ng_list*ng_distr(state, Nmax, lmax) )

def B_avg(state, Nmax, lmax):
    """Expectation value of the glueball operator B in the full (N,l) basis.

    Params:
        - state (ndarray): state in the m = 0 basis.
        - Nmax, lmax (int): basis truncation.
    Returns:
        - <B> (complex).
    """
    basis, ibasis = construct_basis_m0(Nmax, lmax)
    res = 0.0 + 0.0j
    for N, l in basis:
        if (N-2, l) in ibasis:
            ng = (N - l)//2
            coeff = np.sqrt(2*ng*(2*ng+1)/6)
            res += coeff * np.conj(state[ibasis[N-2, l]]) * state[ibasis[N, l]]
    return res

def l_distr(state, Nmax, lmax):
    """Angular momentum distribution, P(l) = sum_N |<N,l|psi>|^2.

    Params:
        - state (ndarray): state in the m = 0 basis.
        - Nmax, lmax (int): basis truncation.
    Returns:
        - P (ndarray, shape (lmax + 1,)).
    """
    basis, ibasis = construct_basis_m0(Nmax, lmax)
    P = np.zeros(lmax+1)
    for N, l in basis:
        P[l] += np.abs(state[ibasis[N,l]])**2
    return P


# ---------------------------------------------------------------- effective model in the glueball ladder

def build_ops(ng_max):
    """Glueball number and ladder operators in the |𝒩> basis.

    Params:
        - ng_max (int): number of glueball states retained.
    Returns:
        - ng_op, B, Bd (ndarray, shape (ng_max, ng_max)): number operator, lowering and raising operators.
    """
    lK = np.arange(ng_max)
    ng_op = np.diag(lK)
    B = np.diag(np.sqrt(2*lK[1:]*(2*lK[1:]+1)), 1)/np.sqrt(6)
    Bd = B.T
    return ng_op, B, Bd

def glueball_dist_num(t, B, Bd):
    """Numerical glueball distribution, from exponentiation of Heff.

    Params:
        - t (float): evolution time.
        - B, Bd (ndarray): from build_ops.
    Returns:
        - c (ndarray, complex): amplitudes c_𝒩 of the state evolved from the vacuum.
    """
    return sl.expm((1j*t*(B + Bd)/np.sqrt(6)))[:,0]

def exp_vals(t, ng_op, B, Bd):
    """Glueball number and amplitude <B> in the effective model at time t.

    Params:
        - t (float): evolution time.
        - ng_op, B, Bd (ndarray): from build_ops.
    Returns:
        - <𝒩> (float), <B> (complex).
    """
    psi = glueball_dist_num(t, B, Bd)
    ng_val = np.real(np.vdot(psi, ng_op @ psi))
    Bval = np.vdot(psi, B @ psi)
    return ng_val, Bval

def glueball_dist_analyt(t, ng_max=1000):
    """Analytic glueball distribution, squeezing parameter r = t/3.

    Params:
        - t (float): evolution time.
        - ng_max (int): number of glueball states returned.
    Returns:
        - c (ndarray, shape (ng_max,), complex): amplitudes c_𝒩.
    """
    ## Squeezing parameter xi = r e^{i theta}
    r = t/3
    psi = np.zeros(ng_max, dtype=complex)
    for ng in range(ng_max):
        psi[ng] = float(factorial(2*ng+1))**0.5 * (1j*np.tanh(r))**ng /( 2**ng * factorial(ng) * np.cosh(r)**(3/2) )
    return psi
