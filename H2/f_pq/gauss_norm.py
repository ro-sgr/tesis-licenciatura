# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np


def GaussNorm(a: int) -> float:
    """ Factor de normalización Gaussiano 1s

    Parámetro
        a : exponente Gaussiano
    """
    return np.power(2*a/np.pi, 3/4)