# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.

from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import Batch
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService

from numpy.typing import NDArray # type annotation
from scipy.optimize import minimize # minimizador

from .funcion_costo import funcion_costo


def VQE(
    hamiltoniano, # Hamiltoniano
    ansatz, # circuito del ansatz
    params_inicial: NDArray = None,
    # Simulación
    backend = None, noise_model = None, nivel_optimizacion: int = 3, batch = None,
    # Config. simulación
    maxiter: int = 10, tol: float = 0.01
) -> tuple[list[float]]:
    """ Variational Quantum Eigensolver

    Parámetros
        backend: 'backend' ideal o 'fake' para realizar la simulación
        noise_model : NoiseModel()
        nivel_optimizacion: nivel de optimización
        params_inicial : parámetros iniciales

    Devuelve:
        energia: valor de la energía calculado
        params_optimizados: parámetros optimizados
    """

    # Simulador ideal
    if backend is None:
        # Simulación sin ruido
        if noise_model is None:
            sim = AerSimulator()
        # Simulación con ruido personalizado
        else:
            sim = AerSimulator(noise_model=noise_model)
    # Simulación con perfil de ruido real
    else:
        sim = backend
    
    # Parámetros iniciales del ansatz
    if params_inicial is None:
        num_params = ansatz.num_parameters # número de parámetros T_i en el ansatz
        params_inicial = [0] * num_params # parámetros iniciales (todos cero)
    else:
        params_inicial = params_inicial # parámetros iniciales

    # Pass Manager
    pass_manager = generate_preset_pass_manager(backend=sim, optimization_level=nivel_optimizacion)
    ansatz_isa = pass_manager.run(ansatz)
    hamiltoniano_isa = hamiltoniano.apply_layout(layout=ansatz_isa.layout)

    # Estimador
    batch = Batch(backend=sim)
    estimador = Estimator(mode=batch)
    
    # Proceso de minimización de la energía
    resultado = minimize(
        funcion_costo,
        params_inicial,
        args = (ansatz_isa, hamiltoniano_isa, estimador),
        method = "COBYLA",
        options = {
            "maxiter": maxiter, # máximo número de iteraciones
            "tol": tol # toleracia de convergencia
        }
    )

    batch.close()
    params_opt = resultado.x # parámetros optimizados

    # Energía total
    energia = resultado.fun

    return energia, params_opt