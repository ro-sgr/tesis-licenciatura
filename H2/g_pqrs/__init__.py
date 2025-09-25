# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

"""
STO-3G (:mod:`H2.g_pqrs`)
==========================================================================

.. currentmodule:: H2.g_pqrs

Elementos de matriz de dos cuerpos para H2.
Base espín orbital:
    χ1 = ψ+ α
    χ2 = ψ+ β
    χ3 = ψ- α
    χ4 = ψ- β

donde ψ+ y ψ- representan los orbitales moleculares ligantes y antiligantes
donde α y β son el espín arriba (+1/2) y abajo (-1/2)
"""

# Factores
from .k2 import K2

# Integral de dos electrones
from .ac_bd import ACBD
from .v_12 import V12
from .v_mn2 import Vmn2

# Elemento de matriz
from .g_pqrs import gpqrs

__all__ = [
    # Factores
    "K2",
    # Integral de dos electrones
    "ACBD",
    "V12",
    "Vmn2",
    # Elemento de matriz
    "gpqrs",
]