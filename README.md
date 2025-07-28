# tesis-licenciatura

Código realizado para mi tesis de Licenciatura en la carrera de Física (plan 2002) de la Facultad de Ciencias, UNAM.

## Estructura

### Cuadernos de Jupyter

Los cuadernos de Jupyter (`.ipynb`) explican a detalle el cómo calcular los elementos para la base STO-3G, así como los elementos de matriz $f_{pq}\,$ y $\,g_{pqrs}$. Todo esto para la molécula de hidrógeno.

`(1) STO-3G.ipynb` Cálculo de los parámetros óptimos para STO-3G.

`(2) f_pq.ipynb` Cálculo de los elementos de matriz de un cuerpo.

`(3) g_pqrs.ipynb` Cálculo de los elementos de matriz de dos cuerpos.

`(4) H2.ipynb` Cálculo de la energía de la molécula de hidrógeno a partir de la aproximación de Hartree-Fock (RHF & UHF) y el método de interacción de configuraciones (CI).

`(5) VQE.ipynb` Variational Quantum Eigensolver.

### Scripts

El directorio `H2` se trata de una paquetería la cual contiene todos los scripts (archivos `.py`) con el código de los cuadernos de Jupyter. Cada archivo `lipsum.py` en el directorio `H2` corresponde al cuaderno de Jupyter `(#) lipsum.ipynb`. Ningún script fue escrito pensado para correrse de manera independiente, sino para ser importados a los cuadernos de Jupyter.

### Otros

Directorio `imgs` contiene los gráficas obtenidas para los elementos de matriz así como de las energías de la molécula de hidrógeno para cada uno de los métodos abordados en el presente trabajo.

Directorio `data` contiene archivos `.csv` con diversos conjuntos de datos calculados y empleados a lo largo del código.

## Configuración

1. Instalar **uv** (`pip install uv==0.8.3`)¹

2. Correr laboratorio de Jupyter (`uv run --with jupyter jupyter lab`).

¹ [uv](https://docs.astral.sh/uv/) es un gerente de paquetes y proyectos escrito en Rust, significativamente más rápido que `pip`.