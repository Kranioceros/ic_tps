from NN import NN
import numpy as np
import matplotlib.pyplot as plt
from utils import particionar_k_out, convert_to_one_dimension, WinnerTakesAll, particionar, Calcular_Area

def main():
    nn = NN([2,5,4,3], learning_rate=.2)#nn = NN([4,4,3], learning_rate=.1)
    datos = np.genfromtxt("icgtp1datos/irisbin.csv", dtype=float, delimiter=',')

    datos_area = Calcular_Area(datos)

    patrones_test = 25
    particiones = particionar_k_out(datos, patrones_test)

    datos_x = datos_area[:,:-3]
    datos_labels = datos[:,-3:]

    v_errores_particiones = []

    min_error = -1
    v_c1_best = []
    v_c2_best = []
    v_c3_best = []

    for _i in range(len(particiones)):
        epocas_convergencia_iteracion = nn.Train(datos_area[particiones[_i][0]], max_epochs=100, tol_error=.1, alfa=0.5, tam_output=3)

        outputs_particiones = []
        
        cant_error = 0
        v_clase1 = []
        v_clase2 = []
        v_clase3 = []

        for _p in particiones[_i][1]:
            #Salida de la red (devuelve una matriz de una fila con un vector con las salidas)
            output = nn.Test(np.array(datos_x[_p]))

            #Salida con 1 y -1s como una lista 
            output_wta = WinnerTakesAll(output[0][:])
            #print(f"output_wta: {output_wta} || {datos_x[_p,0]} , {datos_x[_p,1]}, {datos_x[_p,2]}, {datos_x[_p,3]}")
            #Agrego la salida al vector de salidas
            outputs_particiones.append(output_wta)

            clase_output = np.argmax(output_wta)
            #print(f"clase: {clase_output}")
            if(clase_output==0):
                v_clase1.append(_p)
            elif(clase_output == 1):
                v_clase2.append(_p)
            else:
                v_clase3.append(_p)

            #Si cualquiera es verdadero (distinto) entonces hay un error
            if((output_wta!=datos_labels[_p]).any()):
                cant_error += 1
        
        error = cant_error / patrones_test
        v_errores_particiones.append(error)

        if(min_error == -1 or error < min_error):
            min_error = error
            v_c1_best = v_clase1
            v_c2_best = v_clase2
            v_c3_best = v_clase3
        print(f"Error en particion {_i} {error}")

    
    print(f"Media {np.mean(v_errores_particiones)}")
    print(f"STD {np.std(v_errores_particiones)}")

    plt.scatter(datos_x[v_c1_best,0], datos_x[v_c1_best,1], color=(1,0,0))
    plt.scatter(datos_x[v_c2_best,0], datos_x[v_c2_best,1], color=(0,0,1))
    plt.scatter(datos_x[v_c3_best,0], datos_x[v_c3_best,1], color=(0,1,0))
    plt.show()


if __name__ == "__main__":
    main()

