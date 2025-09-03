# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from .r_p import RP # coordenada de centro de carga
from .k1 import K1 # factor pre-exponencial
from .f0 import F0 # función de Boys, n = 0


def Vpq_AB(a: float, b: float, RA: NDArray, RB: NDArray, RC: NDArray, ZC: float) -> float:
    """ Integral cinética V_pq(C)

    Parámetros
        (a, b) : exponente orbital Gaussiano
        (RA, RB, RC) : coordenada del núcleo (A, B, C)
        ZC : carga del núcleo C
    """
    factor: float = -2*np.pi/(a+b) * ZC # factor común del término pq
    
    if np.array_equal(RA, RB) and np.array_equal(RB, RC): # todos los núcleos iguales
        Vpq: float = factor
    else: # cualquier otro caso
        RAB2: float = np.square(np.linalg.norm(RA-RB))
        Rp: NDArray = RP(a, b, RA, RB) # coordenada de centro de carga
        RPC2: float = np.square(np.linalg.norm(Rp-RC))
        Vpq: float = factor * K1(a, b, RA, RB) * F0((a+b)*RPC2)
        
    return Vpq