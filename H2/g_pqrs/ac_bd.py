# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from H2.f_pq import RP # coordenada de centro de carga
from H2.f_pq import F0 # función de Boys, n = 0
from .k2 import K2 # factor pre-exponencial


def ACBD(a: float, b: float, c: float, d: float, RA: NDArray, RB: NDArray, RC: NDArray, RD: NDArray) -> float:
    """ Integral de dos electrones (AC|BD)
    
    (a, b, c, d) : exponente orbital Gaussiano
    (RA, RB, RC, RD) : coordenadas del núcleo (A, B, C, D)
        A, C : asociados al electrón 1
        B, D : asociados al electrón 2
    """

    factor: float = 2*np.power(np.pi, 5/2) / ( (a+c) * (b+d) * np.sqrt(a+b+c+d) )

    if np.array_equal(RA, RB) and np.array_equal(RB, RC) and np.array_equal(RC, RD):
        return factor
    else:
        Rp: NDArray = RP(a, c, RA, RC)
        Rq: NDArray = RP(b, d, RB, RD)
        RPQ2: float = np.square(np.linalg.norm(Rp-Rq))
    
        if (np.array_equal(RA, RB) and np.array_equal(RC, RD)) and ( (a==b and c==d) or (a==c and b==d) or (a==d and b==c) ):
            term: float = factor * K2(a, b, c, d, RA, RB, RC, RD)
        elif (np.array_equal(RA, RD) and np.array_equal(RB, RC)) and ( (a==b and c==d) or (a==c and b==d) or (a==d and b==c) ):
            term: float = factor * K2(a, b, c, d, RA, RB, RC, RD)
        elif (np.array_equal(RA, RC) and np.array_equal(RB, RD)) and ( (a==b and c==d) or (a==c and b==d) or (a==d and b==c) ):
            term: float = factor * F0((a+c)*(b+d)/(a+b+c+d)*RPQ2)
        else:
            term: float = factor * K2(a, b, c, d, RA, RB, RC, RD) * F0((a+c)*(b+d)/(a+b+c+d)*RPQ2)
            
        return term