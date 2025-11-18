# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from .arg import arg


def K1(a: float, b: float, RA: NDArray, RB: NDArray) -> float:
    """ Factor pre-exponencial (integral 1 cuerpo)
    
    Parámetros
        (a, b)   : exponente orbital Gaussiano
        (RA, RB) : coordenada del núcleo (A, B)
    """
    return np.exp(arg(a, b, RA, RB))