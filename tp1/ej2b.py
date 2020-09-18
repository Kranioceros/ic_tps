import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # Para graficos 3D
from neurona import Neurona, validacion_cruzada
from utils import extender, particionar, part_apply, configurar_grilla


def main():
    for nro in ["10", "50", "70"]:
        # Leemos archivo de entrenamiento
        data = np.genfromtxt(f"tp1/icgtp1datos/spheres2d{nro}.csv",
                             dtype=float, delimiter=',')
        # Agregamos una columna constante -1 para el sesgo
        data_ext = extender(data)

        # Creamos diccionario con las opciones para inicalizar la neurona
        init_opts = {"dim": 4, "aleatorizar": (-0.5, 0.5)}

        # Creamos diccionario con las opciones para entrenar la neurona
        trn_opts = {"umbral_err": 0.05, "coef_apren": 0.1, "max_epocas": 40}

        # Realizamos 5 particiones de entrenamiento con relacion 80/20 de entrenamiento/pruebas
        parts = particionar(data_ext, 5, 0.8, random=True)

        # Realizamos validacion cruzada
        (neuronas, errores) = validacion_cruzada(
            data_ext, parts, init_opts, trn_opts)

        # Creamos 8 subplots con proyeccion 3D
        fig, axs = plt.subplots(4, 2, subplot_kw={'projection': '3d'})

        # Sustituimos los ultimos 2 por una proyeccion 2D...
        axs[3, 0].remove()
        axs[3, 1].remove()
        fig.add_subplot(4, 2, (7, 8))
        axs = fig.get_axes()
        part_n = 0

        # Graficamos los patrones de cada particion
        for an, ax, (_part_trn, part_tst) in zip(neuronas, axs[:5], parts):
            # Configuramos axis
            configurar_grilla(ax)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('x3')
            ax.set_title(f'Prueba de particion n. {part_n}')
            part_n += 1
            # Evaluamos neurona para datos de prueba
            x_ext = data_ext[part_tst, :-1]
            y = np.apply_along_axis(an.evaluar, 1, x_ext)
            # Graficamos patrones de prueba
            idx_t = (y > 0)
            idx_f = (y < 0)
            x = data[part_tst, :-1]
            ax.scatter(x[idx_t, 0], x[idx_t, 1], zs=x[idx_t, 2], c='b')
            ax.scatter(x[idx_f, 0], x[idx_f, 1], zs=x[idx_f, 2], c='r')

        # Graficamos la salida deseada para cada patron del dataset
        ax = axs[5]
        configurar_grilla(ax)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        ax.set_title('Salida deseada para todo el dataset')
        x = data
        yd = x[:, -1]
        idx_t = (yd > 0)
        idx_f = (yd < 0)
        ax.scatter(x[idx_t, 0], x[idx_t, 1], zs=x[idx_t, 2], c='b')
        ax.scatter(x[idx_f, 0], x[idx_f, 1], zs=x[idx_f, 2], c='r')

        # Graficamos los errores como un stem
        ax = axs[6]
        ax.stem(errores)
        ax.set_xlim(-0.3, 4.3)
        ax.set_title("Error por particion")

        # Error promedio
        ax.axhline(y=np.average(errores), c='g', label='Promedio')

    plt.show()

    # Ejecutar la funcion main solo si el script es ejecutado directamente
if __name__ == "__main__":
    main()
