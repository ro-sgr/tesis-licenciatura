# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from .t_pq import Tpq


def Tmn(d: NDArray, a: NDArray, RA: NDArray, RB: NDArray) -> float:
    """ Integral cinética total T_mn

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
    """
    Mmn: float = 0.0 # elemento de matriz
    
    for dp, ap in zip(d, a):
        for dq, aq in zip(d, a):
            Mmn += dp * dq * Tpq(ap, aq, RA, RB) # elemento de matriz
            
    return Mmn