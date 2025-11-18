# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation
from scipy.special import factorial2 # doble factorial


def R_GTO(l: int, α: float, r: NDArray) -> NDArray:
    """ Función radial de tipo Guassiana

    Parámetros
        n : número cuántico principal
        α : exponente orbital Gaussiano
        r : coordenada radial
    """
    m1: float = 2*np.power(2*α, 0.75) / np.power(np.pi, 0.25)
    m2: float = np.sqrt(np.power(2,l) / factorial2(2*l+1))
    m3: NDArray = np.power(np.sqrt(2*α)*r, l)
    m4: NDArray = np.exp(-α*np.power(r,2))
    return m1 * m2 * m3 * m4