from NN import NN
import numpy as np
import matplotlib.pyplot as plt
from utils import particionar_k_out, convert_to_one_dimension, WinnerTakesAll, particionar

def main():
    nn = NN([4,5,4,3], learning_rate=.2)#nn = NN([4,4,3], learning_rate=.1)
    datos = np.genfromtxt("icgtp1datos/irisbin.csv", dtype=float, delimiter=',')

    patrones_test = 25
    particiones = particionar_k_out(datos, patrones_test)

    datos_x = datos[:,:-3]
    datos_labels = datos[:,-3:]

    v_errores_particiones = []
    for _i in range(len(particiones)):
        epocas_convergencia_iteracion = nn.Train(datos[particiones[_i][0]], max_epochs=100, tol_error=.2, alfa=0.5, tam_output=3)

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
        
        if(_i%10==0):
            plt.scatter(datos_x[v_clase1,2], datos_x[v_clase1,3], color=(1,0,0))
            plt.scatter(datos_x[v_clase2,2], datos_x[v_clase2,3], color=(0,0,1))
            plt.scatter(datos_x[v_clase3,2], datos_x[v_clase3,3], color=(0,1,0))
            plt.show()
        
        error = cant_error / patrones_test
        v_errores_particiones.append(error)
        print(f"Error en particion {_i} {error}")
    
    print(f"Media {np.mean(v_errores_particiones)}")
    print(f"STD {np.std(v_errores_particiones)}")







if __name__ == "__main__":
    main()

