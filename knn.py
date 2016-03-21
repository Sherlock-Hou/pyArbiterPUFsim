# -*- coding: utf-8 -*-

import math
import pufsim
import numpy as np

def tangential(x):
    """Anti-symmetrische sigmoide Aktivierungsfunktion."""
    return math.tanh(x)

# sigmoid function
def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

class KNN(object):
    """Ein Künstliches Neuronales Netz."""
    def __init__(self):
        self.eingabeNeuronen = []
        self.zwischenNeuronen = [[]] # zweidimensional
        self.ausgabeNeuronen = []

    def berechne(self, eingabewerte):
        for i in range(len(eingabewerte)):
            self.eingabeNeuronen[i].wert = eingabewerte[i]

        for zwischenNeuron in self.zwischenNeuronen:
            for neuron in zwischenNeuron:
                neuron.berechne()

        for ausgabeNeuron in self.ausgabeNeuronen:
            ausgabeNeuron.berechne()

        # Alle Werte der Ausgabeneuronen als Liste zurückgeben
        return [neuron.wert for neuron in self.ausgabeNeuronen]

    def toString(self):
        string = ""
        #string = "EingabeWerte: "
        #string += str([str(eingabeNeuron.wert) for eingabeNeuron in knn.eingabeNeuronen]) + "\n"
        for layer, zwischenNeuronen in enumerate(self.zwischenNeuronen, start=0):
            string += str(layer) + ". Axone "
            for neuron in zwischenNeuronen:
                string += "| "
                for axon in neuron.axone:
                    string += str(axon.gewicht) + " "
            #string += "\n" + str(layer) + ". Neuronen[" + str(len(self.zwischenNeuronen[layer])) + "]: "
            #for neuron in zwischenNeuronen:
            #    string += str(neuron.wert) + " "
        string += "\n"
        string += str(len(self.zwischenNeuronen)) + ". Axone "
        for axon in self.ausgabeNeuronen[0].axone:
            string += str(axon.gewicht) + " "
        return string + "\n"

class Axon(object):
    """ein künstliches Axon, das eine Kante in einem KNN darstellt."""
    def __init__(self, neuron=None):
        self.gewicht = 1.0
        self.neuron = neuron

class Neuron(object):
    """Ein künstliches Neuron, das einen Knoten in einem KNN darstellt."""
    def __init__(self, wert=0.0):
        self.wert = wert
        self.axone = []  # Kanten zu anderen (Eingabe-)Neuronen
        #self.aktivierung = tangential #SLP
        self.aktivierung = sigmoid # Multilayer

    def berechne(self):
        # Aufsummieren
        summe = 0.0
        for axon in self.axone:
            summe += axon.gewicht*axon.neuron.wert

        # Aktivierungsfunktion anwenden
        self.wert = self.aktivierung(summe)

def trainiereSLP(knn, trainingsdatensatz, lernrate=0.1):
    # Alle Trainingsdaten durchspielen
    for eingabedaten, ausgabeErwartet in trainingsdatensatz:
        # Ergebnis des KNN berechnen
        ausgabeBerechnet = knn.berechne(eingabedaten)
        # Netz verändern
        for i, neuron in enumerate(knn.ausgabeNeuronen):
            differenz = ausgabeErwartet[i] - ausgabeBerechnet[i]
            # Gewichte der anliegenden Axone anpassen;
            # die Differenz bestimmt, in welche Richtung
            for axon in neuron.axone:
                axon.gewicht += lernrate*differenz*axon.neuron.wert

def trainiere2(knn, trainingsdatensatz, lernrate=0.1):
    # Alle Trainingsdaten durchspielen
    for eingabedaten, ausgabeErwartet in trainingsdatensatz:
        # Ergebnis des KNN berechnen
        ausgabeBerechnet = knn.berechne(eingabedaten)
        # Netz verändern
        for i, neuron in enumerate(knn.ausgabeNeuronen):
            differenz = ausgabeErwartet[i] - ausgabeBerechnet[i]
            # Gewichte der anliegenden Axone anpassen;
            # die Differenz bestimmt, in welche Richtung
            for axon in neuron.axone:
                axon.gewicht += lernrate*differenz*axon.neuron.wert

def trainiereBackprop(knn, trainingsdatensatz):
    train = np.array(trainingsdatensatz)

    knn.berechne(trainingsdatensatz)

    for challenge, response in train:

        ausgabeBerechnet = knn.berechne(challenge)

        ausgabeError = response - ausgabeBerechnet
        ausgabeDelta = ausgabeError * sigmoid()

        # # forward propagation
        # l0 = X
        # l1 = nonlin(np.dot(l0, syn0))
        #
        # # how much did we miss?
        # l1_error = y - l1
        #
        # # multiply how much we missed by the
        # # slope of the sigmoid at the values in l1
        # l1_delta = l1_error * nonlin(l1, True)
        #
        # # update weights
        # syn0 += np.dot(l0.T, l1_delta)

class KNNBuilder(object):

    def __init__(self, numberOfInputs, innerDimentions):
        self.numberOfInputs = numberOfInputs
        self.innerDimentions = innerDimentions

    def build(self):
         # Neuronales Netz konstruieren
        knn = KNN()
        # Ausgabeneuron
        knn.ausgabeNeuronen.append(Neuron())
        # Eingabeneuronen
        for i in range(self.numberOfInputs):
            knn.eingabeNeuronen.append(Neuron())

        #todo REFACTOREN!!
        #Verbinde Neuronen bei SLP
        if knnLayers == 0:
            for eingabeNeuron in knn.eingabeNeuronen:
                knn.ausgabeNeuronen[0].axone.append(Axon(eingabeNeuron))
        else:
            for layer, numberNeurons in enumerate(knnLayers, start=0):
                #Add Neuronen für die Ebenen
                for i in range(numberNeurons):
                    knn.zwischenNeuronen[layer].append(Neuron())
                #Verbinde Neuronen der ersten, inneren Ebene zu Eingangsneuronen
                if layer == 0:
                    for firstLayerNeuronen in knn.zwischenNeuronen[0]:
                        for eingabeNeuron in knn.eingabeNeuronen:
                            firstLayerNeuronen.axone.append(Axon(eingabeNeuron))
                #Verbinde Neuronen auf Zwischenebenen
                else:
                    for layerNeuronen in knn.zwischenNeuronen[layer]:
                        for previousLayerNeuronen in knn.zwischenNeuronen[layer-1]:
                            layerNeuronen.axone.append(Axon(previousLayerNeuronen))
                #Verbinde Neuronen der letzten, inneren Ebene zu einem Ausgabeneuron
                if layer == len(knnLayers) - 1:
                    for lastLayerNeuronen in knn.zwischenNeuronen[layer]:
                        knn.ausgabeNeuronen[0].axone.append(Axon(lastLayerNeuronen))
        return knn

class DatasetBuilder(object):

    def __init__(self, dataSize, trainingEvalRatio=0.7):
        self.dataSet = []
        self._maxChallengeSize = dataSize ** 2
        self._trainingEvalRatio = trainingEvalRatio

        challenges = pufsim.genChallengeList(dataSize, self._maxChallengeSize)
        for challenge in challenges:
            self.dataSet.append((challenge, [puf.challengeBit(challenge)]))

    @property
    def trainingsSet(self):
        return self.dataSet[:int(round(self._trainingEvalRatio * self._maxChallengeSize))]

    @property
    def evaluationSet(self):
        return self.dataSet[int(round(self._trainingEvalRatio * self._maxChallengeSize)):]

def kNNRatio(knn, dataSet):
    ratio = 0
    for eingabewerte, ausgabewerte in dataSet:
        knn.berechne(eingabewerte)
        ratio += 1 if round(knn.ausgabeNeuronen[0].wert) == ausgabewerte[0] else 0
    return float(ratio) / len(dataSet)

class SameMultiplexerTimes():
    def generateTimes(self):
        return (0.966967509537182, 0.1257048514440371, 0.2706525095800949, 0.3575951436163608)

if __name__ == '__main__':
    #todo BackpropagationAlg, Refactorn, Docu
    # Puf Länge festlegen
    pufLength = 4
    knnLayers = [3]

    # Zufällige PUF erzeugen
    puf = pufsim.puf(pufsim.RNDUniform(), pufLength)
    # Gleiche PUF erzeugen
    #puf = pufsim.puf(SameMultiplexerTimes(), pufLength)

    #KNN erstellen
    knnBuilder = KNNBuilder(pufLength, knnLayers)
    knn = knnBuilder.build()

    #Datensets erstellen
    dataSetBulider = DatasetBuilder(pufLength)
    trainingsdatensatz = dataSetBulider.trainingsSet
    evaluationdatensatz = dataSetBulider.evaluationSet
    print "Trainingsset: " + str(trainingsdatensatz)
    print "Evaluationsset: " + str(evaluationdatensatz)

    print "Vor dem Training: knn vs. response"
    print "% ", kNNRatio(knn, evaluationdatensatz)
    print knn.toString()

    # Training SLP
    for i in range(10000):
        trainiereSLP(knn, trainingsdatensatz)

    # Training Multilayer
    #for i in range(10000):
    #    trainiereBackprop(knn, trainingsdatensatz)

    print "Nach dem Training: knn vs. response"
    print "% ", kNNRatio(knn, evaluationdatensatz)
    print knn.toString()
