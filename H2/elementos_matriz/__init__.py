# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

"""
STO-3G (:mod:`H2.elementos_matriz`)
===========================================

.. currentmodule:: H2.elementos_matriz

Elementos de matriz de uno y dos cuerpos para H2.
Base espín orbital:
    χ1 = ψ+ α
    χ2 = ψ+ β
    χ3 = ψ- α
    χ4 = ψ- β
"""

from .un_cuerpo import un_cuerpo
from .dos_cuerpos import dos_cuerpos

from .elementos_matriz import elementos_matriz
from .elementos_matriz_rango import elementos_matriz_rango
from .elementos_H2 import elementos_H2

__all__ = [
    "un_cuerpo",
    "dos_cuerpos",
    "elementos_matriz",
    "elementos_matriz_distancias",
    "elementos_H2"
]