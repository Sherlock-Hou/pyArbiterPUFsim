# -*- coding: utf-8 -*-

import math
import pufsim
import time

def tangential(x):
    """Anti-symmetrische sigmoide Aktivierungsfunktion."""
    return math.tanh(x)

class KNN(object):
    """Ein Künstliches Neuronales Netz."""
    def __init__(self):
        self.eingabeNeuronen = []
        self.ausgabeNeuronen = []

    def berechne(self, eingabewerte):
        for i in range(len(eingabewerte)):
            self.eingabeNeuronen[i].wert = eingabewerte[i]

        for ausgabeNeuron in self.ausgabeNeuronen:
            ausgabeNeuron.berechne()

        # Alle Werte der Ausgabeneuronen als Liste zurückgeben
        return [neuron.wert for neuron in self.ausgabeNeuronen]

class Axon(object):
    """ein kpnstliches Axon, das eine Kante in einem KNN darstellt."""
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

        # Aktivierungsfunkrion anwenden
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

    puflength = 6
    # Neuronales Netz konstruieren
    knn = KNN()

    #puf = pufsim.puf(pufsim.RNDUniform(), puflength)
    puf = pufsim.puf(SameMultiplexerTimes(), puflength)
    # Ausgabeneuron
    knn.ausgabeNeuronen.append(Neuron())
    # Zwei Sensoren
    for i in range(puflength):
        tmp = Neuron()
        knn.eingabeNeuronen.append(tmp)
        knn.ausgabeNeuronen[0].axone.append(Axon(tmp))

    challenges = pufsim.genChallengeList(puflength, 2 ** puflength)
    trainingsdatensatz = []
    for challenge in challenges:
        trainingsdatensatz.append((challenge, [puf.challengeBit(challenge)]))
    print trainingsdatensatz

    print "Vor dem Training: knn vs. response"
    ratio = 0
    for eingabewerte, ausgabewerte in trainingsdatensatz:
        knn.berechne(eingabewerte)
        ratio += 1 if round(knn.ausgabeNeuronen[0].wert) == ausgabewerte[0] else 0
        #print round(knn.ausgabeNeuronen[0].wert), ausgabewerte[0]
    print "% ", float(ratio) / len(trainingsdatensatz)

    # Training
    for i in range(10):
        trainiere(knn, trainingsdatensatz)

    print "Nach dem Training: knn vs. response"
    ratio = 0
    for eingabewerte, ausgabewerte in trainingsdatensatz:
        knn.berechne(eingabewerte)
        ratio += 1 if round(knn.ausgabeNeuronen[0].wert) == ausgabewerte[0] else 0
        #print round(knn.ausgabeNeuronen[0].wert), ausgabewerte[0]
    print "% ", float(ratio) / len(trainingsdatensatz)

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
