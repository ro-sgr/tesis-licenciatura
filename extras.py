def guardar(nombre:str, data:list) -> None:
    """ Guardar lista de datos

    nombre: nombre del archivo
    data: lista de datos por ser guardada
    """
    with open(f"data/{nombre}.csv", "w") as file:
        file.write(f"{nombre}\n")
        for item in data:
            file.write(f"{item}\n")