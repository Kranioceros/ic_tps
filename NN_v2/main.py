from NN import NN
import numpy as np
import matplotlib.pyplot as plt
from utils import particionar

def main():
    #nn = NN([2,3,2,1], learning_rate=.1) # nn = NN([2,3,2,1], learning_rate=.2) 
    datos = np.genfromtxt("icgtp1datos/concentlite.csv", dtype=float, delimiter=',')
    #Matriz de patrones sin etiquetas
    m_inputs = datos[:,:-1]
    #Vector de etiquetas de los patrones
    v_labels = datos[:, -1]
    particiones = particionar(datos, 5, .8, True)

    min_error = -1
    v_true_best = []
    v_false_best = []
    v_fn_best = []
    v_fp_best = []

    for _i in range(len(particiones)):
        nn = NN([2,5,4,1], learning_rate=.2) # nn = NN([2,3,2,1], learning_rate=.2) 
        v_true = []
        v_false = []
        v_false_positive = []
        v_false_negative = []

        epocas_convergencia_iteracion = nn.Train(datos[particiones[_i][0]], max_epochs=300, tol_error=.25, alfa=0)
        print(f"Epocas para convergencia en particion {_i+1}: {epocas_convergencia_iteracion+1}") 

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

        if(min_error == -1 or errorPart < min_error):
            min_error = errorPart
            v_true_best = v_true
            v_false_best = v_false
            v_fn_best = v_false_negative
            v_fp_best = v_false_positive


    plt.scatter(m_inputs[v_true_best,0], m_inputs[v_true_best,1], color=(1,0,0), label="Verdadero")
    plt.scatter(m_inputs[v_false_best,0], m_inputs[v_false_best,1], color=(0,0,1), label="Falso")
    plt.scatter(m_inputs[v_fp_best,0], m_inputs[v_fp_best,1], color=(0,1,0), label="Falso Positivo")
    plt.scatter(m_inputs[v_fn_best,0], m_inputs[v_fn_best,1], color=(1,1,0), label="Falso Negativo")
    plt.legend(loc="lower right", title="", frameon=False)
    plt.title("Concentlite")
    plt.show()

if __name__ == "__main__":
    main()

