import numpy as np
import matplotlib.pyplot as plt

def comparar_funcs(x, f, g, labels, linestyle):
    """ Gráfica de una función principal f(x) y múltiples funciones gi(x)
    
    x : dominio
    f : función principal
    g : lista de funciones a comparar
    labels : lista de etiquetas para las funciones gi(x)
    linestyle : lista de estilos de línea para las funciones gi(x)
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # 1 fila, 2 columnas, tamaño de figura (10, 5)

    # Comparación función de Slater con Gaussianas
    axs[0].plot(x, f, label='Slater', color='k') # función principal
    for i in range(len(g)): # funciones aproximadas
        axs[0].plot(x, g[i], label=labels[i], linestyle=linestyle[i])
    axs[0].set_xlabel('Radio (u.a.)')
    axs[0].set_ylabel(r'$R_{1s}$')
    axs[0].legend()
    axs[0].grid()

    # Comparación de las funciones de distribución radiales de Slater con Gaussianas
    k = 4*np.pi*np.power(x,2)
    axs[1].plot(x, k*np.power(f,2), label='Slater', color='k') # función principal
    for i in range(len(g)): # funciones aproximadas
        axs[1].plot(x, k*np.power(g[i],2), label=labels[i], linestyle=linestyle[i])
    axs[1].set_xlabel('Radio (u.a.)')
    axs[1].set_ylabel(r'$4\pi r^2 |R_{1s}|^2$')
    axs[1].legend()
    axs[1].grid()

    plt.show()