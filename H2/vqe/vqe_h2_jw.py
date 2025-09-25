# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

import numpy as np
from numpy.typing import NDArray # type annotation

# Coeficientes STO3G
from H2.sto_kG import d # coeficientes de contracción
from H2.sto_kG import a2 as a # exponentes orbitales Gaussianos

from .ham_h2_jw import H2_Ham_JW
from .vqe import VQE


RA: NDArray = np.array([0, 0, 0])
RB: NDArray = np.array([1.4, 0, 0])

# carga nuclear
ZA: float = 1.0
ZB: float = 1.0


def VQE_H2_JW(
    # Parámetros Gaussianos
    d: NDArray = d, a: NDArray = a,
    # Coord. nucleares
    RA: NDArray = RA, RB: NDArray = RB,
    # Cargas nucleares
    ZA: float = ZA, ZB: float = ZB,
    # Ansatz
    params_inicial: NDArray = None,
    # Simulación
    backend = None, noise_model = None, nivel_optimizacion: int = 3, batch = None,
    # Config. simulación
    maxiter: int = 10, tol: float = 0.01
) -> tuple[list[float]]:
    """ Variational Quantum Eigensolver para H2 en la base STO-3G
        y bajo la transformación de Jordan-Wigner

    Parámetros
        backend: 'backend' ideal o 'fake' para realizar la simulación
        noise_model : NoiseModel()
        nivel_optimizacion: 
        params_inicial : parámetros iniciales

    Devuelve:
        energia: valor de la energía calculado
        params_optimizados: parámetros optimizados
    """
    ### (1) Hamiltoniano
    hamiltoniano = H2_Ham_JW(d, a, RA, RB, ZA, ZB)

    ### (2) Creación del ansatz
    # (2.1) Estado inicial
    num_orbitales_espaciales  = 2 # dos orbitales espaciales (ligante y antiligante)
    num_particulas  = (1,1) # una partícula con espín arriba y otra con espín abajo
    mapeo_qubit  = JordanWignerMapper() # mapeo de qubits
    edo_inicial = HartreeFock(num_orbitales_espaciales, num_particulas, mapeo_qubit)
    # (2.2) Ansatz
    ansatz = UCCSD(num_orbitales_espaciales, num_particulas, mapeo_qubit, initial_state=edo_inicial)
    # (2.3) Parámetros iniciales del ansatz
    if params_inicial is None:
        num_params = ansatz.num_parameters # número de parámetros T_i en el ansatz
        params_inicial = [0] * num_params # parámetros iniciales (todos cero)

    energia, params_opt = VQE(hamiltoniano, ansatz, params_inicial, backend, noise_model, nivel_optimizacion, batch, maxiter, tol)

    return energia, params_opt