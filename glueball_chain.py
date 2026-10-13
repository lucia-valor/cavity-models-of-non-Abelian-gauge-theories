"""Fig. 6: many glueballs on a ring of cavities, in the effective glueball Hamiltonian."""

import itertools
import numpy as np


# ---------------------------------------------------------------- basis and effective Hamiltonian

def compositions(n, k):
    """All ways of distributing n glueballs over k sites.

    Params:
        - n, k (int): number of glueballs and number of sites.
    Yields:
        - (tuple of int, length k): non-negative occupations summing to n.
    """
    if k == 1:
        yield (n,)
    else:
        for i in range(n + 1):
            for tail in compositions(n - i, k - 1):
                yield (i,) + tail

def construct_basis_chain_glueball(ng, Ncav):
    """Basis of glueball Fock states for ng glueballs in Ncav cavities.

    Params:
        - ng, Ncav (int): glueball number 𝒩 and number of cavities.
    Returns:
        - basis (ndarray, shape (D, Ncav)): basis[q] = (k0, k1, ..., k{Ncav-1}), the glueball number per cavity.
        - ibasis (dict): inverse map, ibasis[k0, k1, ...] = q.
    """
    basis = []
    ibasis = {}
    for k, state in enumerate(compositions(ng, Ncav)):
        basis.append(np.array(state))
        ibasis[state] = k
    return np.array(basis), ibasis

def Heff_glueball(ng, Ncav=2, alpha0=1, t=0, periodic=True, include_crossKerr=True):
    """Effective Hamiltonian for ng glueballs in Ncav cavities,
        H = -t^2/alpha0 sum_x [ K_x + 2/3 K_x K_{x+1} + 1/12(B_x^dag B_{x+1} + B_{x+1}^dag B_x) ]
    where
        K_x |k> = k_x
        B_x^dag |k> = sqrt((2k_x + 2)(2k_x + 3)) |k + e_x>
        B_x |k> = sqrt(2k_x(2k_x + 1)) |k - e_x>

    Params:
        - ng, Ncav (int): glueball number 𝒩 and number of cavities.
        - alpha0, t (float): interaction U and inter-cavity hopping.
        - periodic (bool): close the chain into a ring (only if Ncav > 2).
        - include_crossKerr (bool): keep the 2/3 K_x K_{x+1} term.
    Returns:
        - H (ndarray, shape (D, D), complex): in the basis of construct_basis_chain_glueball.
    """
    basis, ibasis = construct_basis_chain_glueball(ng, Ncav)
    D = len(basis)
    H = np.zeros((D,D), dtype=complex)
    ## Unit matrices such that e[x] = ( delta_{y,x} )_{y} to add and substract photon easily
    e = np.eye(Ncav)
    for k in basis:
        for x in range(Ncav):
            ## On-site energy K_x if Ncav>2 else K_x/2
            H[ibasis[*k], ibasis[*k]] += k[x] if Ncav>2 else k[x]/2
            ## Cross-Kerr
            if include_crossKerr:
                if (periodic and Ncav>2) or x>0:
                    H[ibasis[*k], ibasis[*k]] += 2/3*k[x]*k[x-1]
            ## Hopping 1/12 B^dag_x B_{x-1}
            if (periodic and Ncav>2) or x>0:
                if k[x-1] > 0:
                    l = k + e[x] - e[x-1]
                    H[ibasis[*l], ibasis[*k]] += 1/12*np.sqrt(2*k[x-1]*(2*k[x-1]+1)*(2*k[x]+2)*(2*k[x]+3))
            ## Hopping 1/12 B^dag_{x-1} B_{x}
            if (periodic and Ncav>2) or x>0:
                if k[x] > 0:
                    l = k - e[x] + e[x-1]
                    H[ibasis[*l], ibasis[*k]] += 1/12*np.sqrt(2*k[x]*(2*k[x]+1)*(2*k[x-1]+2)*(2*k[x-1]+3))
    ## global prefactor
    H = -t**2/alpha0*H
    return H


# ---------------------------------------------------------------- many-body wavefunction and correlator

def psi_multiglueball(ng, state_glueball, basis_glueball, ibasis_glueball):
    """ng-body wavefunction Psi(x1, ..., x_ng): the amplitude of the configuration with a glueball at each x_i.

    Params:
        - ng (int): glueball number 𝒩.
        - state_glueball (ndarray): state in the basis of construct_basis_chain_glueball.
        - basis_glueball, ibasis_glueball: that basis and its inverse.
    Returns:
        - Psi (ndarray, shape (Ncav,)*ng, complex).
    """
    Ncav = len(basis_glueball[0])
    Psi = np.zeros((Ncav,)*ng, dtype=complex)
    for xs in itertools.product(range(Ncav), repeat=ng):
        k = np.bincount(xs, minlength=Ncav)   # glueball number per cavity
        Psi[xs] = state_glueball[ibasis_glueball[tuple(k)]]
    return Psi

def G2_r(state, basis):
    """Density-density correlator against the distance r from site 0 on the ring,
        G2(0) = <K_0 (K_0 - 1)>,   G2(r) = <K_0 K_r> + <K_0 K_{-r}>  for r > 0.

    Params:
        - state (ndarray): state in the basis of construct_basis_chain_glueball.
        - basis (ndarray, shape (D, Ncav)): that basis.
    Returns:
        - G2 (ndarray, shape (Ncav,)): indexed by r = 0, ..., Ncav-1.
    """
    p = np.abs(state)**2 * basis[:, 0]   # probability of each configuration, times K_0
    G2 = p @ basis                        # <K_0 K_x>
    G2[0] -= p.sum()                      # remove the self-correlation at x = 0
    M = len(G2)
    out = [G2[0]] + [G2[r] + G2[-r] for r in range(1, (M + 1)//2)]
    if M % 2 == 0:
        out.append(G2[M//2])   # R = M/2 is its own mirror site: count it once
    return np.array(out)       # R = 0, ..., M//2
