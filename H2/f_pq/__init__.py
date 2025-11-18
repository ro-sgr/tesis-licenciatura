# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

"""
STO-3G (:mod:`H2.f_pq`)
==========================================================================

.. currentmodule:: H2.f_pq

Elementos de matriz de un cuerpo para H2.
Base espín orbital:
    χ1 = ψ+ α
    χ2 = ψ+ β
    χ3 = ψ- α
    χ4 = ψ- β

donde ψ+ y ψ- representan los orbitales moleculares ligantes y antiligantes
donde α y β son el espín arriba (+1/2) y abajo (-1/2)
"""

# Factores
from .gauss_norm import GaussNorm
from .arg import arg
from .k1 import K1

# Integral de traslape
from .s_pq import Spq
from .s_mn import Smn

# Integral cinética
from .t_pq import Tpq
from .t_mn import Tmn

# Integral Coulombiana
from .f0 import F0
from .r_p import RP
from .v_pq_AB import Vpq_AB
from .v_mn1 import Vmn1

# Correcciones decimales
from .decimal_no_cero import decimal_no_cero
from .valor_truncado import valor_truncado

# Elementos de matriz
from .c_pm import cPM # cte. de normalización
from .f_pp import fpp


__all__ = [
    "GaussNorm",
    "arg",
    "K1",
    # Integral traslape
    "Spq",
    "Smn",
    # Integral cinética
    "Tpq",
    "Tmn",
    # Integral Coulombiana
    "F0",
    "RP",
    "Vpq_AB",
    "Vmn1",
    # Correcciones decimales
    "decimal_no_cero",
    "valor_truncado",
    # Elementos de matriz
    "cPM",
    "fpp",
]