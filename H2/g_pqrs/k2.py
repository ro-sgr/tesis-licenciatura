# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from H2.f_pq import arg

def K2(a: float, b: float, c: float, d: float, RA: NDArray, RB: NDArray, RC: NDArray, RD: NDArray) -> float:
    """ Factor pre-exponencial (integral 2 cuerpos)
    
    (a, b, c, d) : exponente orbital Gaussiano
    (RA, RB, RC, RD) : coordenada del núcleo (A, B, C, D)
    """
    return np.exp(arg(a, c, RA, RC) + arg(b, d, RB, RD))