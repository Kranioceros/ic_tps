import numpy as np
import matplotlib.pyplot as plt
from utils import sigmoid, dsigmoid, signo, dsigno

#Clase que modela una Red Neuronal (Neural Network -> NN)
#Lo único necesario es determinar la arquitectura (arq), donde cada elemento de la lista representa la cantidad de neuronas en cada capa
    #Deben existir al menos dos capas (entrada y salida)
        #TODO: CONTROLAR QUE LA DIMENSION DE 'arq' SEA AL MENOS DE 2
    
class NN:
    def __init__(self, arq=list, learning_rate=.3):
        self.learning_rate = learning_rate

        #Cantidad de neuronas de entrada
        self.input_nodes = arq[0]
        #Cantidad de neuronas de salida
        self.output_nodes = arq[-1]
        
        #Función de activación
        self.f_activation = sigmoid
        #Derivada de funcion de activación
        self.f_dactivation = dsigmoid

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
        inputs.reshape(1,len(inputs))
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
    def Train(self, x, max_epochs=5, tol_error=.1, alfa = 0):

        #Matriz de patrones sin etiquetas
        m_inputs = x[:,:-3]
        #Vector de etiquetas de los patrones
        v_labels = x[:, -3]
        #Delta W de la iteracion anterior para utilizar el termino de momento
        DWAnt = []
        for w in self.v_weights:
            DWAnt.append(np.zeros(w.shape))

        #DWAnt = np.zeros(self.v_weights.shape)

        #Por cada época
        epocas_para_convergencia = 0
        for _k in range(max_epochs):
            #Vector de errores cometidos en cada patrón
            v_error = []
            #Para cada patrón
            for _i in range(x.shape[0]):
                #Patrón actual
                inputs = m_inputs[_i, :]
                #Etiquetas del patrón actual
                targets = v_labels[_i]

                #Propagación hacia adelante obteniendo el error cometido en cada capa
                outputs = self.FeedForward(inputs)
                
                #Error en la salida final de la red
                error_output = targets - outputs[-1]

                #Error instantaneo en la última capa
                ei_output = error_output*self.f_dactivation(outputs[-1])

                #Vector de errores instantaneos
                v_ei = []
                v_ei.append(ei_output)

                #Errores instantaneos en las capas ocultas
                # Recorro el vector de matrices de de atrás hacia adelante
                for _j in reversed(range(0,len(self.v_weights)-1)): #el -1 es por ser base 0
                    #Error instantáneo en capa j
                    ei_j = (v_ei[0]@(self.v_weights[_j+1]).T)*self.f_dactivation(outputs[_j])
                    #Como recorro al reves las capas, voy insertando el error siempre al comienzo
                        # De esta manera cuando termine, en v_ei[0] voy a tener el error de la capa 0. De manera general, tengo en v_ei[X] el error de la capa x
                    v_ei.insert(0,ei_j)


                #Ajusto pesos y biases (desde la primera oculta)
                for _j in range(1,len(self.v_bias)):

                    DWAux = self.learning_rate*((outputs[_j-1]).T@v_ei[_j]) + alfa*DWAnt[_j]
                    self.v_weights[_j] += DWAux
                    DWAnt[_j] = DWAux
                    self.v_bias[_j] += self.learning_rate*v_ei[_j]

                #Obligar a que sea entendido como una matriz de una fila
                inputs_array = inputs.reshape(1,len(inputs))

                #Actualizo pesos y biases de la primera capa (se hace aparte porque usa los inputs)
                    #TODO: se podrían poner los inputs en outputs[0] para no tener que hacerlo aparte

                DWAux = self.learning_rate*(inputs_array.T@v_ei[0]) + alfa*DWAnt[0]
                self.v_weights[0] += DWAux
                DWAnt[0] = DWAux
                self.v_bias[0] += self.learning_rate*v_ei[0]

                #Error del patrón actual
                v_error.append(np.abs(signo(self.Test(inputs))-targets))
            
            #Error medio de la época actual
            mean_error = np.mean(v_error)
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
        #Devuelvo el error de la última capa
        wta_output = []#self.WinnerTakesAll(outputs[-1])
        return outputs[-1]
        

    def WinnerTakesAll(self, x):
        max_idx = np.argmax(x)
        v_x = np.ones((1,len(x)))*-1
        v_x[max_idx] = 1
        return v_x


