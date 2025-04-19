import numpy as np
import matplotlib.pyplot as plt

def comparar_funcs(nombre:str, x:np.ndarray, f, g:list, N:list, labels:list, linestyle:list, linewidth:list, marker:list, markersize:list, color:list, xticks:list, yticks:list):
    """ Gráfica de una función principal f(x) y múltiples funciones gi(x)

    nombre : nombre del archivo .svg
    x : dominio
    f : función principal
    g : lista de funciones a comparar
    N : [a,b]
        1er conjunto -> se consideran únicamente cada 'a' número de elementos
        2do conjunto -> se consideran únicamente cada 'b' número de elementos
    (labels, linestyle, linewidth) : lista de (etiquetas, estilos de línea, grosor de línea) para las funciones gi(x)
    (marker, markersize, color) : lista de (marcador, tamaño de marcador, color) para las funciones gi(x)
    (xticks, yticks) : 
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # 1 fila, 2 columnas, tamaño de figura (10, 5)

    # Comparación función de Slater con Gaussianas
    axs[0].plot(x, f, label='Slater', color='k') # función principal
    for i in range(len(g)): # funciones aproximadas
        xi = x[::N[i]]
        gi = g[i][::N[i]]
        axs[0].plot(xi, gi, label=labels[i], linestyle=linestyle[i], linewidth=linewidth[i], marker=marker[i], markersize=markersize[i], color=color[i])
    axs[0].set_xlabel('Radio (u.a.)', fontsize='xx-large', labelpad=10)
    axs[0].set_ylabel(r'$R_{1s}$', fontsize='xx-large', labelpad=10)
    axs[0].set_xticks(xticks[0])
    axs[0].set_yticks(yticks[0])

    # Comparación de las funciones de distribución radiales de Slater con Gaussianas
    k = 4*np.pi*np.power(x,2)
    axs[1].plot(x, k*np.power(f,2), label='Slater', color='k') # función principal
    for i in range(len(g)): # funciones aproximadas
        xi = x[::N[i]]
        ki = 4*np.pi*np.power(xi,2)
        gi = g[i][::N[i]]
        axs[1].plot(xi, ki*np.power(gi,2), label=labels[i], linestyle=linestyle[i], linewidth=linewidth[i], marker=marker[i], markersize=markersize[i], color=color[i])
    axs[1].set_xlabel('Radio (u.a.)', fontsize='xx-large', labelpad=10)
    axs[1].set_ylabel(r'$4\pi r^2 |R_{1s}|^2$', fontsize='xx-large', labelpad=10)
    axs[1].set_xticks(xticks[1])
    axs[1].set_yticks(yticks[1])

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize='xx-large')
        ax.legend(fontsize='xx-large', loc='upper right')
        ax.grid()

    fig.subplots_adjust(wspace=0.25) # espacio entre figuras
    plt.savefig(f"imgs/{nombre}.svg", format="svg")
    plt.show()