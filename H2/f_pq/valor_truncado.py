# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation


def valor_truncado(valor: float | NDArray, decimal: int) -> float:
    """ Devuelve un 'valor' truncado hasta una cierta precisión 'decimal'
    """
    ord_mag = np.power(10, decimal) # orden de magnitud
    return np.trunc(valor * ord_mag) / ord_mag