# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

import numpy as np
from numpy.typing import NDArray # type annotation

from H2.utils import cargar


# Coeficientes de Stewart [Ref1]
dSte: list[float] = [0.154329, 0.535328, 0.444635]
aSte: list[float] = [2.22766, 0.405771, 0.109818]

# Coeficientes de Szabo & Ostlund [Ref2]
dSO: NDArray = np.array([0.444635, 0.535328, 0.154329])
aSO: NDArray = np.array([0.168856, 0.623913, 3.42525])

# Este trabajo
data: NDArray = cargar('H2_STO3G.csv', usecols=(0,1), dtype=('str'), unpack=False) # cargar valores calculados
valores = dict()
for valor in data:
    valores[str(valor[0])] = float(valor[1])
    
d: NDArray = np.array([valores['d1'], valores['d2'], valores['d3']]) # coeficientes de contracción
a1: NDArray = np.array([valores['a1'], valores['a2'], valores['a3']]) # exponentes orbitales Gaussianos (ζ = 1)
a2: NDArray = np.array([valores['a1_2'], valores['a2_2'], valores['a3_2']]) # exponentes orbitales Gaussianos (ζ = 1.24)


""" Referencias

[Ref1] Robert F. Stewart. Small Gaussian Expansions of Atomic Orbitals. The Journal of Chemical Physics 50, 2485 (1969); doi: 10.1063/1.1671406

[Ref2] A. Szabo and N. Ostlund, Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory (Dover Publications, Mineola, NY, 1996).
"""