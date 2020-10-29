import numpy as np
from GA import GA

def main():

    N = 40 #Cantidad de agentes en una poblacion
    n = 10 #Cantidad de alelos en cada agente
    probMutation = .8 #Probabilidad de que un alelo se mute
    maxGen = 300 #Cantidad maxima de generacion a iterar

    #Creo el controlador de la poblacion
    ga = GA(N, n, NumberOfOnes)

    print("Primera poblacion")
    ga.DebugPopulation()

    #Itero tantas veces como generaciones maximas
    for _i in range(maxGen):
        #Poblacion nueva, me voy guardando los nuevos agentes
        newPopulation = []
        print(f"Mean Fitness generacion {_i}: {np.mean(ga.EvaluatePopulation())}")

        #Itero tantas veces como agentes en una poblacion
        for _j in range(N):
            #Eligo dos agentes al "azar"
            a1 = ga.Picker()
            a2 = ga.Picker()

            #Combino los dos agentes y obtengo uno nuevo
            newAgent = a1.CrossOver(a2,.6)
            #Muto a este agente nuevo
            newAgent.Mutate(probMutation)
            #Lo agrego a la nueva poblacion
            newPopulation.append(newAgent)

        #Una vez generados N agentes nuevos, reemplazo la poblacion actual
        ga.NewGeneration(newPopulation)

    print("Ultima poblacion")
    ga.DebugPopulation()

#Funcion de fitness de prueba, crece segun la cantidad de 1s
def NumberOfOnes(v):
    fitness = 0
    for i in v:
        if i==1:
            fitness += 10
    return fitness

if __name__ == "__main__":
    main()