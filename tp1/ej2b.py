import numpy as np
import matplotlib.pyplot as plt
from neurona import Neurona, validacion_cruzada
from utils import extender, particionar, part_apply


def main():
    for nro in ["10", "50", "70"]:
        # Leemos archivo de entrenamiento
        datos = np.genfromtxt(f"tp1/icgtp1datos/spheres2d{nro}.csv",
                              dtype=float, delimiter=',')
        # Agregamos una columna constante -1 para el sesgo
        datos_ext = extender(datos)

        # Creamos diccionario con las opciones para inicalizar la neurona
        init_opts = {"dim": 4, "aleatorizar": (-0.5, 0.5)}

        # Creamos diccionario con las opciones para entrenar la neurona
        entr_opts = {"umbral_err": 0.05, "coef_apren": 0.1, "max_epocas": 40}

        # Realizamos 10 particiones con relacion 80/20 de entrenamiento/pruebas
        parts = particionar(datos_ext, 10, 0.8, random=True)

        # Realizamos validacion cruzada
        (_neuronas, errores) = validacion_cruzada(
            datos_ext, parts, init_opts, entr_opts)

        # Graficamos los errores de cada particion
        print(errores)
        plt.stem(errores)
        plt.show()


    # Ejecutar la funcion main solo si el script es ejecutado directamente
if __name__ == "__main__":
    main()
