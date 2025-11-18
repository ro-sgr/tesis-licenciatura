# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation


def arg(a: float, b: float, RA: NDArray, RB: NDArray) -> float:
    """ Argumento del factor pre-exponencial K
    
    Parámetros
        (a, b)   : exponente orbital Gaussiano
        (RA, RB) : coordenada del núcleo (A, B)
    """
    p: float = a+b # exponente total
    mu: float = a*b/p
    RAB2: float = np.square(np.linalg.norm(RA-RB))
    
    return -mu*RAB2