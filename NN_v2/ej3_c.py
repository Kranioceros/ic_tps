from NN import NN
import numpy as np
import matplotlib.pyplot as plt
from utils import particionar, convert_to_one_dimension

def main():
    #nn = NN([1,1], learning_rate=.1)
    datos = np.genfromtxt("icgtp1datos/concentlite.csv", dtype=float, delimiter=',')
    v_datos = convert_to_one_dimension(datos) #datos en una dimensión
    particiones = particionar(v_datos, 5, .8, True)
    
    v_datos_x = v_datos[:,0]
    v_datos_y = v_datos[:,-1]
    
    for _i in range(len(particiones)):
        nn = NN([1,1], learning_rate=.1)
        v_true = []
        v_false = []
        v_false_positive = []
        v_false_negative = []
        epocas_convergencia_iteracion = nn.Train(v_datos[particiones[_i][0]], max_epochs=150, tol_error=.1, alfa=0)
        print(f"Epocas para convergencia en particion {_i+1}: {epocas_convergencia_iteracion+1}")  #max_epochs = 150    
        outputs_particiones = []
        
        for _p in particiones[_i][1]:
            outputs_particiones.append(nn.Test(np.array([v_datos_x[_p]]))) 
        
        cont_error = 0
        cont = 0
        for _k in particiones[_i][1]:
            auxLabel = -1
            if(outputs_particiones[cont]) > 0:
                auxLabel = 1
                v_true.append(_k)
            else:
                auxLabel = -1
                v_false.append(_k)

            if(auxLabel != v_datos_y[_k]):
                if(v_datos_y[_k] == -1):
                    v_false_positive.append(_k)
                else:
                    v_false_negative.append(_k)
                    
                cont_error += 1
            cont += 1
        errorPart = cont_error/len(outputs_particiones)
        print("Error en partición "+repr(_i+1)+" ->  "+repr(errorPart))
        print(f"Cantidad de errores en particion: {cont_error} de {len(outputs_particiones)} patrones.")
        print(f"Falsos positivos: {len(v_false_positive)} | Falsos negativos: {len(v_false_negative)}.")
        print("\n")


    plt.scatter(datos[v_true,0], datos[v_true,1], color=(1,0,0),label="Verdadero")
    plt.scatter(datos[v_false,0], datos[v_false,1], color=(0,0,1),label="Falso")
    plt.scatter(datos[v_false_positive,0], datos[v_false_positive,1], color=(0,1,0),label="Falso Positivo")
    plt.scatter(datos[v_false_negative,0], datos[v_false_negative,1], color=(1,1,0),label="Falso Negativo")
    plt.legend(loc="lower right", title="", frameon=False)
    plt.title("Concentlite con una dimensión")
    plt.show()


if __name__ == "__main__":
    main()

