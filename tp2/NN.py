import numpy as np
import matplotlib.pyplot as plt
from utils import sigmoid, dsigmoid, signo, dsigno, WinnerTakesAll, identidad

#Clase que modela una Red Neuronal (Neural Network -> NN)
#Lo único necesario es determinar la arquitectura (arq), donde cada elemento de la lista representa la cantidad de neuronas en cada capa
    #Deben existir al menos dos capas (entrada y salida)
        #TODO: CONTROLAR QUE LA DIMENSION DE 'arq' SEA AL MENOS DE 2
    
class NN:
    def __init__(self, arq=list, learning_rate=.3, activation=signo, dactivation=dsigno):
        self.learning_rate = learning_rate

        #Cantidad de neuronas de entrada
        self.input_nodes = arq[0]
        #Cantidad de neuronas de salida
        self.output_nodes = arq[-1]
        
        #Función de activación
        self.f_activation = activation
        #Derivada de funcion de activación
        self.f_dactivation = dactivation

        #Vector con los bias de cada capa
        self.v_bias = []
        #Vector con las matrices de pesos de cada capa
        self.v_weights = []

        #La cantidad de matrices de pesos va a ser siempre una menos que la cantidad de capas
            #eg: si no hay capas ocultas => hay una capa de entrada y una de salida, y una sola matriz de pesos que las une
            #eg: si ahora tenemos una capa ocualta => hay una matriz de pesos que une la capa de entrada con la oculta, y otra que une la oculta con la de salida
        for _i in range(len(arq)-1):
            self.v_weights.append(np.random.rand(arq[_i], arq[_i+1])-0.5) #Matriz de pesos con pesos aleatoriamente entre -0.5 y 0.5 (si unimos capa X con la capa X+1 => X->filas y X+1->columnas)
            self.v_bias.append(np.ones((1,arq[_i+1]))) #Matriz de biases de 1 fila y tantos elementos como neuronas en la capa siguiente


    #Propagación hacia adelante
    #Devuelve un vector con las salidas producidas en cada capa
    #outputs[-1] contendrá la salida final de la red
    def FeedForward(self, inputs):
        outputs = []
        #Obligar a que sea entendido como una matriz de una fila
        
        inputs = inputs.reshape(1,len(inputs))
        #Salida de la primera capa
        outputs.append(self.f_activation(inputs@self.v_weights[0] + self.v_bias[0]))

        #Salidas de las capas hidden y salida final
        for _i in range(1,len(self.v_weights)):
            outputs.append(self.f_activation(outputs[_i-1]@self.v_weights[_i] + self.v_bias[_i]))

        return outputs
    

    #Entrenamiento de la red
    # x -> los datos completos leídos del archivo (supone que trae cada patron en una fila, y que el último elemento de la misma es la etiqueta del patrón)
    # max_epochs -> cantidad máxima de épocas para entrenar la red
    # tol_error -> tolerancia de error medio entre cada época.
    def Train(self, m_inputs,v_labels, max_epochs=5, tol_error=.1):

        #Por cada época
        epocas_para_convergencia = 0
        for _k in range(max_epochs):
            #Vector de errores cometidos en cada patrón
            v_error = []
            #Para cada patrón
            for _i in range(m_inputs.shape[0]):
                #Patrón actual
                inputs = m_inputs[_i, :]
                
                #Etiquetas del patrón actual
                targets = v_labels[_i,:]
                
                #Propagación hacia adelante obteniendo el error cometido en cada capa
                outputs = self.FeedForward(inputs)
                
                #Error en la salida final de la red
                error_output = targets - outputs[-1]
                #print(f"target: {targets} | outputs: {outputs[-1]} | error: {error_output}")
                inputs = inputs.reshape(len(inputs),1)

                for w in self.v_weights:
                    w += self.learning_rate * (inputs@error_output)
                
                for b in self.v_bias:
                    #print(f"error: {error_output} | bias: {b}")
                    b += self.learning_rate * error_output

                #Error del patrón actual
                if(self.output_nodes==1):
                    if(self.f_activation==identidad):
                        error = np.abs(self.Test(inputs)-targets)
                        #error = ((self.Test(inputs)-targets)**2)/2
                    else:
                        error = np.abs(signo(self.Test(inputs))-targets)/2
                    v_error.append(error)
                else:
                    error = np.abs(WinnerTakesAll(self.Test(inputs)[0][:]) - targets)
                    if(sum(error)!=0):
                        v_error.append(1)
                    else:
                        v_error.append(0)

            #Error medio de la época actual
            maximo = np.abs(v_error[np.argmax(v_error)])
            if(maximo == 0): maximo = 1

            mean_error = np.mean(v_error)/maximo
            #mean_error = np.mean(v_error)
            #print("ERROR TRN: ", mean_error)
            
            #print("MEAN ERROR: ", mean_error)
            #print("Error medio epoch ", _k, ": ", mean_error)

            epocas_para_convergencia = _k

            #Si el error medio es menor que la tolerancia fijada termina el entrenamiento
            if(mean_error<tol_error):
                return epocas_para_convergencia
            
        return epocas_para_convergencia


    #Prueba de la red
    #Recibe un patrón de entrada y devuelve la salida final de la red
    def Test(self, inputs):
        #Propgación hacia adelante
        outputs = self.FeedForward(inputs)

        return outputs[-1]