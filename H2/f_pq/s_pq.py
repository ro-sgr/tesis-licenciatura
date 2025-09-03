# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from .gauss_norm import GaussNorm
from .k1 import K1


def Spq(a: float, b: float, RA: NDArray, RB: NDArray) -> float:
    """ Integral de traslape S_pq (normalizada)
    
    Parámetros
        (a, b)   : exponente orbital Gaussiano
        (RA, RB) : coordenada del núcleo (A, B)
    """
    return GaussNorm(a) * GaussNorm(b) * np.power(np.pi/(a+b), 3/2) * K1(a, b, RA, RB)