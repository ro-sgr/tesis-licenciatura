# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from H2.f_pq import cPM
from .v_mn2 import Vmn2


def gpqrs(P: NDArray, d: NDArray, a: NDArray, RA: NDArray, RB: NDArray) -> float:
    """ Elemento de matriz g_pqrs
    
    P : vector de elementos de la base, P = (p,q,r,s)
    d : vector de coeficientes de expansión (d1, d2, ..., dk)
    a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
    (RA, RB) : coordenada del núcleo (A, B)
    """
    p,q,r,s = P
    nucleos: list[NDArray] = [RA, RB]

    Mpqrs: float = 0.0 # elemento de tensor

    # si ocurre que (p,r) o (q,s) no son simultáneamente par o impar, g_pqrs = 0 (debido al espín)
    if ((p in [1,3] and r not in [1,3]) or (p in [2,4] and r not in [2,4])) or ((q in [1,3] and s not in [1,3]) or (q in [2,4] and s not in [2,4])):
        Mpqrs: float = 0 # elemento de tensor
    else:
        sgn_p: int = 1 if p in [1,2] else -1 # signo de p
        sgn_q: int = 1 if q in [1,2] else -1 # signo de q
        sgn_r: int = 1 if r in [1,2] else -1 # signo de r
        sgn_s: int = 1 if s in [1,2] else -1 # signo de s
    
        # coeficientes de normalización (p,q,r,s)
        cp: float = cPM(d, a, RA, RB, sgn_p)
        cq: float = cPM(d, a, RA, RB, sgn_q)
        cr: float = cPM(d, a, RA, RB, sgn_r)
        cs: float = cPM(d, a, RA, RB, sgn_s)
        
        for A in nucleos:
            for B in nucleos:
                for C in nucleos:
                    for D in nucleos:
                        sign: int = 1
                        if np.array_equal(A, RB):
                            sign *= sgn_p
                        if np.array_equal(B, RB):
                            sign *= sgn_q
                        if np.array_equal(C, RB):
                            sign *= sgn_r
                        if np.array_equal(D, RB):
                            sign *= sgn_s

                        mpqrs: float = sign * Vmn2(d, a, A, B, C, D)
                        Mpqrs += mpqrs
    
        Mpqrs: float = cp*cq*cr*cs*Mpqrs # multiplicar Mpqrs por coefs. de normalización
        
    return Mpqrs