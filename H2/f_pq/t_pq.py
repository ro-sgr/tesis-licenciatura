# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from .s_pq import Spq


def Tpq(a: NDArray, b: NDArray, RA: NDArray, RB: NDArray) -> float:
    """ Integral cinética T_pq (normalizada)

    Parámetros
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (RA, RB) : coordenada del núcleo (A, B)
    """
    RAB2: float = np.square(np.linalg.norm(RA-RB)) # cuadrado de diferencia internuclear
    
    return (a*b)/(a+b) * (3 - 2*(a*b)/(a+b)*RAB2 ) * Spq(a, b, RA, RB)