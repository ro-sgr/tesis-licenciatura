import numpy as np
import matplotlib.pyplot as plt

def compararSTO(nombre:str, formato:str, ref:dict, datasets:list, xticks:list, yticks:list):
    """ Gráfica de una función principal f(x) y múltiples funciones gi(x)

    nombre : nombre del archivo .formato (si 'None' no se guarda la gráfica)
    ref : conjunto de referencia de la forma {'x':[...], 'y':[...]}
    datasets : lista de conjuntos por comparar [data1, data2, ...]
        dataN.keys() = 'y', 'labels', 'linestyle', 'linewidth', 'marker', 'markersize', 'color', 'N'
        N : [a,b,...]
            1er conjunto -> se consideran únicamente cada 'a' número de elementos
            2do conjunto -> se consideran únicamente cada 'b' número de elementos
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # 1 fila, 2 columnas, tamaño de figura (15, 5)

    # Comparación función de Slater con Gaussianas
    axs[0].plot(ref['x'], ref['y'], label=ref['label'], color=ref['color']) # función principal
    for data in datasets: # funciones aproximadas
        x = ref['x'][::data['N']]
        y = data['y'][::data['N']]
        axs[0].plot(x, y, label=data['label'], linestyle=data['linestyle'], linewidth=data['linewidth'], marker=data['marker'], markersize=data['markersize'], color=data['color'])
    axs[0].set_ylabel(r'$R_{1s}$', fontsize='xx-large', labelpad=10)
    axs[0].set_xticks(xticks[0])
    axs[0].set_yticks(yticks[0])

    # Comparación de las funciones de distribución radiales de Slater con Gaussianas
    k = 4*np.pi*np.power(ref['x'],2)
    axs[1].plot(ref['x'], k*np.power(ref['y'],2), label='Slater', color='k') # función principal
    for data in datasets: # funciones aproximadas
        x = ref['x'][::data['N']]
        k = 4*np.pi*np.power(x,2)
        y = data['y'][::data['N']]
        axs[1].plot(x, k*np.power(y,2), label=data['label'], linestyle=data['linestyle'], linewidth=data['linewidth'], marker=data['marker'], markersize=data['markersize'], color=data['color'])
    axs[1].set_ylabel(r'$4\pi r^2 |R_{1s}|^2$', fontsize='xx-large', labelpad=10)
    axs[1].set_xticks(xticks[1])
    axs[1].set_yticks(yticks[1])

    for ax in axs:
        ax.set_xlabel('Radio ($a_0$)', fontsize='xx-large', labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize='xx-large')
        ax.legend(fontsize='xx-large', loc='upper right')
        ax.grid()

    fig.subplots_adjust(wspace=0.25)
    if nombre is not None: # guardar
        plt.savefig(f"imgs/{nombre}.{formato}", format=f"{formato}")
    plt.show()


def graficarVQE(nombre:str, ref:dict, datasets:list):
    """ Gráfica de datos obtenidos para el VQE (unidades atómicas y Angstroms)

    nombre : nombre del archivo .svg (si 'None' no se guarda la gráfica)
    ref : diccionario con datos de referencia (e.g. Kandala)
        x : distancias en Angstroms
        y : energía en Hartrees
    datasets -> [data1, data2, ... ] : lista de diccionarios con cada conjunto de datos por graficar
        x : distancias en unidades atómicas de distancia
        y : energía en Hartrees
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    factor = 0.529177249 # unidades atómicas de distancia a Angstroms
    
    # Unidades atómicas
    axs[0].plot(ref['x']/factor, ref['y'], label=ref['label'], linewidth=ref['linewidth'], linestyle=ref['linestyle'], marker=ref['marker'], color=ref['color'])
    for data in datasets:
        axs[0].plot(data['x'], data['y'], label=data['label'], linewidth=data['linewidth'], linestyle=data['linestyle'], marker=data['marker'], color=data['color'])
    axs[0].set_xlabel("Distancia interatómica (Unidades Atómicas)", fontsize='xx-large', labelpad=10)
    axs[0].set_ylabel("Energía (Hartrees)", fontsize='xx-large', labelpad=10)

    # Angstroms
    axs[1].plot(ref['x'], ref['y'], label=ref['label'], linewidth=ref['linewidth'], linestyle=ref['linestyle'], marker=ref['marker'], color=ref['color'])
    for data in datasets:
        axs[1].plot(data['x']*factor, data['y'], label=data['label'], linewidth=data['linewidth'], linestyle=data['linestyle'], marker=data['marker'], color=data['color'])
    axs[1].set_xlabel("Distancia interatómica (Angstrom)", fontsize='xx-large', labelpad=10)
    
    order = np.arange(0,len(datasets)+1) # añade leyendas en el orden que se insertaron los conjuntos de datos
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize='xx-large', loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize='xx-large')
        ax.grid()
    
    fig.subplots_adjust(wspace=0.2) # espacio entre figuras
    if nombre is not None: # guardar
        plt.savefig(f"imgs/{nombre}.svg", format="svg")
    plt.show()