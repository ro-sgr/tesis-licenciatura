# tesis-licenciatura-codigo

[![python - 3.13.5](https://img.shields.io/badge/python-3.13.5-3170A1?logo=python&logoColor=ffffff)](https://www.python.org/downloads/release/python-3135/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<table>
    <tr>
        <td><img src="imgs/Logo_FC.jpg" alt="Left" style="float: center; width: 70%"/></td>
        <td><img src="imgs/Logo_IF.jpg" alt="Left" style="float: right; width: 30%"/></td>
    </tr>
</table>

<h2 style="text-align: center;"><b>Teoría y Computación Cuántica para Sistemas Moleculares: Un Estudio desde la Molécula de Hidrógeno</b></h2>

Código de Tesis

Que para obtener el título de: Físico

Presenta: Rodrigo Segura Moreno

Tutor: Dr. Ricardo Gutiérrez Jáuregui

Facultad de Ciencias, UNAM.

Ciudad Universitaria, CDMX, 2025

---

## Estructura del código

### Cuadernos de Jupyter

Los cuadernos de Jupyter (`.ipynb`) explican a detalle el cómo calcular los elementos para la base STO-3G, así como los elementos de matriz $f_{pq}$ y $g_{pqrs}$. Todo esto para la molécula de hidrógeno.

`(1) STO-3G.ipynb` Cálculo de los parámetros óptimos para STO-3G.

`(2) f_pq.ipynb` Cálculo de los elementos de matriz de un cuerpo.

`(3) g_pqrs.ipynb` Cálculo de los elementos de matriz de dos cuerpos.

`(4) H2.ipynb` Cálculo de la energía de la molécula de hidrógeno a partir de la aproximación de Hartree-Fock (RHF & UHF) y el método de interacción de configuraciones (CI).

`(5) VQE.ipynb` Variational Quantum Eigensolver.

`(plt) XYZ.ipynb` Cuaderno con código de las imágenes en la tesis.

### Scripts

El directorio `H2` se trata de una paquetería la cual contiene todos los scripts (archivos `.py`) con el código de los cuadernos de Jupyter. Cada archivo `lipsum.py` en el directorio `H2` corresponde al cuaderno de Jupyter `(#) lipsum.ipynb`. Ningún script fue escrito pensado para correrse de manera independiente, sino para ser importados desde los cuadernos de Jupyter.

### Directorios

`imgs/` contiene las gráficas obtenidas para los elementos de matriz así como de las energías de la molécula de hidrógeno para cada uno de los métodos abordados en el presente trabajo y los logos institucionales.

`data/` contiene archivos `.csv` con diversos conjuntos de datos calculados y empleados a lo largo del código.

## Configuración

El código empleado usa el lenguaje de programación Python v3.13.5

1. Instalar **uv** (`pip install uv==0.8.3`)¹

2. Correr laboratorio de Jupyter (`uv run --with jupyter jupyter lab`).

¹ [uv](https://docs.astral.sh/uv/) es un gerente de paquetes y proyectos escrito en Rust, significativamente más rápido que `pip`.