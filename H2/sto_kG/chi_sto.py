# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation
from scipy.special import sph_harm_y as sph_harm # armónicos esféricos

from .r_sto import R_STO


def χ_STO(n: int, l: int, m: int, ζ: float, r: NDArray, θ: NDArray, ϕ: NDArray) -> NDArray:
    """ Función de tipo Slater (Slater Type Orbital)

    Parámetros
        (n, l, m) : número cuántico (principal, azimutal, magnético)
        ζ : exponente orbital de Slater
        (r, θ, ϕ) : coordenada (radial, polar, azimutal)
    """
    return R_STO(n, ζ, r) * sph_harm(l, m, ϕ, θ)