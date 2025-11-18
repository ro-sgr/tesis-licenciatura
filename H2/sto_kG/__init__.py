# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

"""
STO-3G (:mod:`H2.sto_kG`)
==================================================

.. currentmodule:: H2.sto_kG

Funciones de tipo Slater y tipo Gaussianas para H2
"""

# Funciones de tipo Slater
from .r_sto import R_STO
from .chi_sto import χ_STO

# Funciones de tipo Gaussianas
from .r_gto import R_GTO
from .chi_gto import χ_GTO

# Contracciones de 'k' Gaussianas
from .r_sto_kG import R_STO_kG
from .chi_sto_kG import χ_STO_kG

# STO-3G, paŕametros
from .sto_3g import dSte, aSte # Stewart (ζ = 1)
from .sto_3g import dSO, aSO # Szabo & Ostlund (ζ = 1.24)
from .sto_3g import d, a1, a2 # Este trabajo (a1: ζ = 1, a2: ζ = 1.24)

__all__ = [
    # Funciones de tipo Slater
    "R_STO",
    "χ_STO",
    # Funciones de tipo Gaussianas
    "R_GTO",
    "χ_GTO",
    # Contracciones de 'k' Gaussianas
    "R_STO_kG",
    "χ_STO_kG",
    # Parámetros de Stewart (ζ = 1)
    "dSte",
    "aSte",
    # Parámetros de Szabo & Ostlund (ζ = 1.24)
    "dSO",
    "aSO",
    # Parámetros de este trabajo
    "d",
    "a1", # ζ = 1
    "a2", # ζ = 1.24
]