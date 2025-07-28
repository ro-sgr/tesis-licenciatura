# Script para realizar diversas gráficas comparativas e ilustrativas a lo largo del trabajo presente

import numpy as np
import matplotlib.pyplot as plt

def compararSTO(nombre: str, formato: str, ref: dict, datasets: list, xticks: list, yticks: list) -> None:
    """ Gráfica de una función principal f(x) y múltiples funciones gi(x)

    nombre : nombre del archivo .formato (si 'None' no se guarda la gráfica)
    ref : conjunto de datos de referencia
        ref.keys() = ['x', 'y', 'label', 'color']
    datasets : lista de conjuntos por comparar [data1, data2, ...]
        dataN.keys() = ['y', 'label', 'linestyle', 'linewidth', 'marker', 'markersize', 'color', 'N']
            N : [a,b,...]
                1er conjunto -> se consideran únicamente cada 'a' número de elementos
                2do conjunto -> se consideran únicamente cada 'b' número de elementos
    """
    fig, axs = plt.subplots(1, 2, figsize = (15, 5))  # 1 fila, 2 columnas, tamaño de figura (15, 5)

    # Comparación función de Slater con Gaussianas
    axs[0].plot(ref['x'], ref['y'], label = ref['label'], color = ref['color']) # función principal
    for data in datasets: # funciones aproximadas
        x = ref['x'][::data['N']]
        y = data['y'][::data['N']]
        axs[0].plot(x, y, label = data['label'], linestyle = data['linestyle'], linewidth = data['linewidth'], marker = data['marker'], markersize = data['markersize'], color = data['color'])
    axs[0].set_ylabel(r'$R_{1s}$', fontsize = 'xx-large', labelpad = 10)
    axs[0].set_xticks(xticks[0])
    axs[0].set_yticks(yticks[0])

    # Comparación de las funciones de distribución radiales de Slater con Gaussianas
    k = 4 * np.pi * np.power(ref['x'], 2)
    axs[1].plot(ref['x'], k * np.power(ref['y'], 2), label = 'Slater', color = 'k') # función principal
    for data in datasets: # funciones aproximadas
        x = ref['x'][::data['N']]
        k = 4 * np.pi * np.power(x, 2)
        y = data['y'][::data['N']]
        axs[1].plot(x, k * np.power(y, 2), label = data['label'], linestyle = data['linestyle'], linewidth = data['linewidth'], marker = data['marker'], markersize=data['markersize'], color = data['color'])
    axs[1].set_ylabel(r'$4\pi r^2 |R_{1s}|^2$', fontsize = 'xx-large', labelpad = 10)
    axs[1].set_xticks(xticks[1])
    axs[1].set_yticks(yticks[1])

    for ax in axs:
        ax.set_xlabel('Radio ($a_0$)', fontsize = 'xx-large', labelpad = 10)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 'xx-large')
        ax.legend(fontsize = 15, loc = 'upper right')
        ax.grid(alpha = alpha, which = 'both')

    fig.subplots_adjust(wspace=0.25)
    if nombre is not None: # guardar
        fig.savefig(f"imgs/{nombre}.{formato}", format = f"{formato}", bbox_inches = 'tight')
    plt.show()


def graficar_elementos_matriz(R, elementos, opciones, guardar: bool = False) -> None:
    """ Graficar elementos de matriz para H2

    Parámetros
        R : distancia interatómica
        elementos : h11, h22, J11, J22, J12, K12
    """

    h11, h22, J11, J22, J12, K12 = elementos
    linewidth, fontsize, labelsize, alpha = opciones
    
    #####################
    ###### Gráfica ######
    #####################
    fig, axs = plt.subplots(1, 2, figsize = (15, 6))  # 1 fila, 1 columnas, tamaño de figura (15, 5)
    
    ### Gráfica derecha ###
    #######################
    axs[0].set_rasterization_zorder(10)
    axs[0].plot(R, h11, linewidth = linewidth, linestyle = 'solid', label = 'h11', color = '#ff595e')
    axs[0].plot(R, h22, linewidth = linewidth, linestyle = 'dashed', label = 'h22', color = '#ff924c')
    # x config
    axs[0].set_xlim(-0.05, 7)
    axs[0].set_xlabel(r'Distancia interatómica ($a_0$)', fontsize = fontsize, labelpad = 5)
    # y config
    axs[0].set_ylim(-1.8, 0.4)
    axs[0].set_ylabel(r'Energía (E$_\mathrm{h}$)', fontsize = fontsize, labelpad = 10)
    axs[0].set_yticks(np.arange(-1.75, 0.3, 0.50))
    axs[0].set_yticks(np.arange(-1.75, 0.3, 0.25), minor = True)
    # plot config
    axs[0].legend(fontsize = fontsize)
    axs[0].tick_params(axis = 'both', which = 'major', labelsize = labelsize)
    axs[0].grid(alpha = alpha)
    
    ### Gráfica izquierda ###
    #########################
    axs[1].plot(R, J11, linewidth = linewidth, linestyle = 'solid', label = 'J11', color = '#ffca3a')
    axs[1].plot(R, J22, linewidth = linewidth, linestyle = 'dashed', label = 'J22', color = '#8ac926')
    axs[1].plot(R, J12, linewidth = linewidth, linestyle = 'dotted', label = 'J12', color = '#1982c4')
    axs[1].plot(R, K12, linewidth = linewidth, linestyle = 'dashdot', label = 'K12', color = '#6a4c93')
    # x config
    axs[1].set_xlim(-0.05, 14)
    axs[1].set_xlabel(r'Distancia interatómica ($a_0$)', fontsize = fontsize)
    # y config
    axs[1].set_ylim(0.1, 1.1)
    # plot config
    axs[1].legend(fontsize = fontsize)
    axs[1].tick_params(axis = 'both', which = 'major', labelsize = labelsize)
    axs[1].grid(alpha = alpha)

    ### Configuración general ###
    #############################
    fig.tight_layout(pad=2)
    plt.show()

    if guardar is True:
        fig.savefig('imgs/H2_elementos_matriz.svg', format='svg', bbox_inches='tight')


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
    axs[0].plot(ref['x']/factor, ref['y'], label = ref['label'], linewidth = ref['linewidth'], linestyle = ref['linestyle'], marker = ref['marker'], color = ref['color'])
    for data in datasets:
        axs[0].plot(data['x'], data['y'], label = data['label'], linewidth = data['linewidth'], linestyle = data['linestyle'], marker = data['marker'], color = data['color'])
    axs[0].set_xlabel("Distancia interatómica (Unidades Atómicas)", fontsize = 'xx-large', labelpad = 10)
    axs[0].set_ylabel("Energía (Hartrees)", fontsize = 'xx-large', labelpad = 10)

    # Angstroms
    axs[1].plot(ref['x'], ref['y'], label = ref['label'], linewidth = ref['linewidth'], linestyle = ref['linestyle'], marker = ref['marker'], color = ref['color'])
    for data in datasets:
        axs[1].plot(data['x']*factor, data['y'], label = data['label'], linewidth = data['linewidth'], linestyle = data['linestyle'], marker = data['marker'], color = data['color'])
    axs[1].set_xlabel("Distancia interatómica (Angstrom)", fontsize = 'xx-large', labelpad = 10)
    
    order = np.arange(0,len(datasets)+1) # añade leyendas en el orden que se insertaron los conjuntos de datos
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize = 'xx-large', loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize='xx-large')
        ax.grid()
    
    fig.subplots_adjust(wspace=0.2) # espacio entre figuras
    if nombre is not None: # guardar
        plt.savefig(f"imgs/{nombre}.svg", format="svg")
    plt.show()
	