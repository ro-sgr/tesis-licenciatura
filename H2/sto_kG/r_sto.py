# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation
from scipy.special import factorial2 # doble factorial


def R_STO(n: int, ζ: float, r: NDArray) -> NDArray:
    """ Función radial de tipo Slater

    Parámetros
        n : número cuántico principal
        ζ : exponente oribital de Slater
        r : coordenada radial
    """
    return np.power(2*ζ, 1.5) / np.sqrt(factorial2(2*n)) * np.power(2*ζ*r, n-1) * np.exp(-ζ*r)