import numpy as np

def guardar(nombre: str, data: list) -> None:
    """ Guardar lista de datos

    Parámetros
        nombre : nombre del archivo
        data   : lista de datos por ser guardada
    """
    with open(f"data/{nombre}.csv", "w") as file:
        file.write(f"{nombre}\n")
        for item in data:
            file.write(f"{item}\n")


def cargar(nombre: str) -> np.ndarray[np.float64]:
    """ Cargar conjunto de datos

    Parámetro
        nombre : nombre del archivo
    """
    return np.loadtxt(f"data/{nombre}.csv", delimiter=",", dtype=('float'), skiprows=1)
	