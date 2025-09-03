# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from .r_gto import R_GTO


def R_STO_kG(ζ: float, d: NDArray, a: NDArray, l: int, r: NDArray) -> NDArray:
    """ Combinación lineal de 'k' Gaussianas (parte radial)

    Parámetros
        ζ : exponente orbital de Slater
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        l : número cuántico azimutal
        r : coordenada radial
    """
    suma = 0
    # calcular cada uno de los k términos de la combinación lineal
    for di, ai in zip(d,a):
        suma += di * R_GTO(l, np.power(ζ,2)*ai, r) # k-ésimo término
    
    return suma