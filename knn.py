# -*- coding: utf-8 -*-

import math
import pufsim


def tangential(x):
    """Anti-symmetrische sigmoide Aktivierungsfunktion."""
    return math.tanh(x)

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
        self.aktivierung = tangential

    def berechne(self):
        # Aufsummieren
        summe = 0.0
        for axon in self.axone:
            summe += axon.gewicht*axon.neuron.wert

        # Aktivierungsfunktion anwenden
        self.wert = self.aktivierung(summe)

def trainiere(knn, trainingsdatensatz, lernrate=0.1):
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

class SameMultiplexerTimes():
    def generateTimes(self):
        return (0.966967509537182, 0.1257048514440371, 0.2706525095800949, 0.3575951436163608)

if __name__ == '__main__':

    # Puf Länge festlegen
    pufLength = 4
    knnLayers = [3]

    # Zufällige PUF erzeugen
    #puf = pufsim.puf(pufsim.RNDUniform(), pufLength)
    # Gleiche PUF erzeugen
    puf = pufsim.puf(SameMultiplexerTimes(), pufLength)

    # Neuronales Netz konstruieren
    knn = KNN()
    # Ausgabeneuron
    knn.ausgabeNeuronen.append(Neuron())
    # Eingabeneuronen
    for i in range(pufLength):
        tmp = Neuron()
        knn.eingabeNeuronen.append(tmp)

    if knnLayers == 0:
        for eingabeNeuron in knn.eingabeNeuronen:
            knn.ausgabeNeuronen[0].axone.append(Axon(eingabeNeuron))
    else:
        for layer, numberNeurons in enumerate(knnLayers, start=0):
            for i in range(numberNeurons):
                knn.zwischenNeuronen[layer].append(Neuron())
                if layer == 0:
                    for firstLayerNeuronen in knn.zwischenNeuronen[0]:
                        for eingabeNeuron in knn.eingabeNeuronen:
                            firstLayerNeuronen.axone.append(Axon(eingabeNeuron))
                else:
                    for layerNeuronen in knn.zwischenNeuronen[layer]:
                        for previousLayerNeuronen in knn.zwischenNeuronen[layer-1]:
                            layerNeuronen.axone.append(Axon(previousLayerNeuronen))

                if layer == len(knnLayers) - 1:
                    for lastLayerNeuronen in knn.zwischenNeuronen[layer]:
                        knn.ausgabeNeuronen[0].axone.append(Axon(lastLayerNeuronen))

    challenges = pufsim.genChallengeList(pufLength, 2 ** pufLength)
    trainingsdatensatz = []
    for challenge in challenges:
        trainingsdatensatz.append((challenge, [puf.challengeBit(challenge)]))
    print(trainingsdatensatz)

    print("Vor dem Training: knn vs. response")
    ratio = 0
    for eingabewerte, ausgabewerte in trainingsdatensatz:
        knn.berechne(eingabewerte)
        ratio += 1 if round(knn.ausgabeNeuronen[0].wert) == ausgabewerte[0] else 0
        #print round(knn.ausgabeNeuronen[0].wert), ausgabewerte[0]
    print "% ", float(ratio) / len(trainingsdatensatz)

    # Knn Übersichtsausgabe
    for eingabeNeuron in knn.eingabeNeuronen:
        print str(eingabeNeuron.wert) + " -",
    for layer, zwischenNeuronen in enumerate(knn.zwischenNeuronen, start=0):
        print "\n" + str(layer) + ". Layer ",
        for neuron in zwischenNeuronen:
            print str(neuron.wert) + " ",

    # Training
    for i in range(10000):
        trainiere(knn, trainingsdatensatz)

    print "\nNach dem Training: knn vs. response"
    ratio = 0
    for eingabewerte, ausgabewerte in trainingsdatensatz:
        knn.berechne(eingabewerte)
        ratio += 1 if round(knn.ausgabeNeuronen[0].wert) == ausgabewerte[0] else 0
        #print round(knn.ausgabeNeuronen[0].wert), ausgabewerte[0]
    print "% ", float(ratio) / len(trainingsdatensatz)


    # Knn Übersichtsausgabe
    for eingabeNeuron in knn.eingabeNeuronen:
        print str(eingabeNeuron.wert) + " -",
    for layer, zwischenNeuronen in enumerate(knn.zwischenNeuronen, start=0):
        print "\n" + str(layer) + ". Layer ",
        for neuron in zwischenNeuronen:
            print str(neuron.wert) + " ",

    #create pufsim with 2 Multiplexer instances

    #do a single challenge to the pufsim with a challenge as list
    # pufsimu.challengeSingle([1,1,1,1,1,1,1])
    # pufsimu.challengeSingle([1,1,1,1,1,1,0])
    #
    # pufsimu.challengeSingle([0,1,1,1,1,1,1])
    #
    # #worst case
    # #print len(pufsim.genChallengeList(10,((2**10)-1)))
    #
    #
    #
    # mutatio = pufsim.MutatorLastBitSwitch()
    #
    # startTime = time.time()
    #
    # tryEval = pufsim.pufEval(16, rndgen, 2**8, mutatio, 1000, 4)
    # tryEval.run()
    #
    # endTime = time.time()
    # print  endTime - startTime
