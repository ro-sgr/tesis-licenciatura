# Este código es parte del proyecto de tesis de licenciatura de Rodrigo Segura Moreno.
#
# (C) Copyright Rodrigo Segura Moreno, 2025.
#
# Este código está licenciado bajo la Licencia MIT.


cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}

def funcion_costo(params, ansatz, hamiltoniano, estimator):
    """ Regresa el estimado de energía a partir del estimador

    Parámetros
        params (NDArray): Arreglo de parámetros del ansatz
        ansatz (QuantumCircuit): Circuito ansatz parametrizado
        hamiltoniano (SparsePauliOp): Representación operacional del Hamiltoniano
        estimator (EstimatorV2): Instancia de la primitiva estimador

    Devuelve
        float: Estimación de energía
    """        

    pub = (ansatz, [hamiltoniano], [params])
    job = estimator.run(pubs = [pub])
    result = job.result()
    pub_result = result[0]
    cost = pub_result.data.evs[0] # energía
    
    return cost