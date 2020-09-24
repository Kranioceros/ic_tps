from NN import NN
import numpy as np
import matplotlib.pyplot as plt
from utils import particionar

def main():
    nn = NN([2,3,2,1], learning_rate=.1) # nn = NN([2,3,2,1], learning_rate=.2) 
    datos = np.genfromtxt("icgtp1datos/concentlite.csv", dtype=float, delimiter=',')
    #Matriz de patrones sin etiquetas
    m_inputs = datos[:,:-1]
    #Vector de etiquetas de los patrones
    v_labels = datos[:, -1]
    particiones = particionar(datos, 5, .8, True)

    
    for _i in range(len(particiones)):
        v_true = []
        v_false = []
        v_false_positive = []
        v_false_negative = []
        epocas_convergencia_iteracion = nn.Train(datos[particiones[_i][0]], max_epochs=150, tol_error=.25, alfa=0.5)
        print(f"Epocas para convergencia en particion {_i+1}: {epocas_convergencia_iteracion+1}")  #max_epochs = 150    
        outputs_particiones = []
        for _p in particiones[_i][1]:
            outputs_particiones.append(nn.Test(m_inputs[_p])) 
        
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

            if(auxLabel != v_labels[_k]):
                if(v_labels[_k] == -1):
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


    plt.scatter(m_inputs[v_true,0], m_inputs[v_true,1], color=(1,0,0))
    plt.scatter(m_inputs[v_false,0], m_inputs[v_false,1], color=(0,0,1))
    plt.scatter(m_inputs[v_false_positive,0], m_inputs[v_false_positive,1], color=(0,1,0))
    plt.scatter(m_inputs[v_false_negative,0], m_inputs[v_false_negative,1], color=(1,1,0))
    plt.show()

if __name__ == "__main__":
    main()
