import numpy as np
from numpy.typing import NDArray # type annotation

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


def cargar(nombre: str,
           usecols: int|tuple[int] = None,
           unpack: bool = True) -> NDArray[float]:
    """ Cargar conjunto de datos

    Parámetro
        nombre : nombre del archivo
    """
    return np.loadtxt(f"data/{nombre}.csv", dtype=('float'), delimiter=",", skiprows=1, usecols=usecols, unpack=unpack)
	