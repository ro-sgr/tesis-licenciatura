# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from .chi_gto import χ_GTO


def χ_STO_kG(ζ: float, d: NDArray, a: NDArray, l: int, m: int, r: NDArray, θ: float, phi: float) -> NDArray:
    """ Combinación lineal de 'k' Gaussianas

    Parámetros
        ζ : exponente de Slater
        d : vector de coeficientes de expansión (d1, d2, ..., dk)
        a : vector de exponentes orbitales Gaussianos (a1, a2, ..., ak)
        (l,m) : número cuántico (azimutal, magnético)
        (r,θ,phi) : coordenada (radial, polar, azimutal)
    """
    suma = 0
    for di, ai in zip(d,a): # calcular cada uno de los k términos de la suma
        suma += di * χ_GTO(l, m, np.power(ζ,2)*ai, r, θ, phi) # término k-ésimo
    
    return suma