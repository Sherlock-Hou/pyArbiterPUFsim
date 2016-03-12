# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import itertools
import random
from multiprocessing import Process, Queue
import numpy as np
import matplotlib.pyplot as plt
import time

#single multiplexer
class multiplexer(object):
    
    def __init__(self, timeTuple):
        self.up_up, self.down_down, self.up_down, self.down_up = timeTuple
    
    #methode recives already accumulated time and adds and switches the values
    #according to the respectiv challenge bit 
    #return: both times as tuple (up, down)
    def challenge(self, bit, time_up, time_down):
        if bit == 0:
            return (time_up + self.up_up), (time_down + self.down_down)
        elif bit == 1:
            return (time_down + self.down_up), (time_up + self.up_down)
        else:
            raise RuntimeError('Bit is not 0 or 1')
#single puf
class puf(object):
    
    #needs an instance of RNDBase to create single Multiplexer and overall
    #size (numer of Multiplexer) of the puf
    def __init__(self, gen, numOfMultip):
        self.gen = gen
        self.numOfMultip = numOfMultip
        self.multiplexerList = []
        
        for i in range(0, self.numOfMultip):
            tmp = multiplexer(gen.generateTimes())
            self.multiplexerList.append(tmp)

    #returns puf size (number of multiplerxers)
    def getSize(self):
        return self.numOfMultip
    
    #single challenge to the puf
    #return: both times as tuple (up, down)
    def challenge(self, bitList):
        time_up = 0.0
        time_down = 0.0
        runner = 0
        
        if len(bitList) != self.numOfMultip:
            raise RuntimeError('Challenge has the wrong length')
        
        for plex in self.multiplexerList:
            time_up, time_down = plex.challenge(bitList[runner], time_up, time_down)
            runner += 1
        return (time_up, time_down)
        
    #single challenge to puf
    #prints nice string ...
    def challengeSingle(self, bitList):
        time_up, time_down = self.challenge(bitList)
        print('{' + ', '.join(map(str, bitList)) + '}\t' + str(time_up - time_down))
        
    #single challenge to the puf
    #return: one bit result
    def challengeBit(self, bitList):
        time_up, time_down = self.challenge(bitList)
        return 1 if ((time_up - time_down) > 0) else 0

#base class for random time generation of a single multiplexer
class RNDBase(object):
    
    __metaclass__ = ABCMeta

    #generateTimes(self) returns 4 floats as tuple (Quadrupel?) 
    #Order of Values: up_up, down_down, up_down, down_up
    @abstractmethod
    def generateTimes(self):
        pass
        
#Uniform distribution
class RNDUniform(RNDBase):
    
    def generateTimes(self):
        return (random.random(),random.random(),random.random(),random.random())

#standard normal distribution
class RNDNormal(RNDBase):
    
    def generateTimes(self):
        return (random.normalvariate(0,1.0), random.normalvariate(0,1.0), random.normalvariate(0,1.0), random.normalvariate(0,1.0))

def genChallengeList(challengeSize, numOfChallenges):
    if ((2 ** challengeSize) == numOfChallenges) :
        return map(list, itertools.product([0, 1], repeat=challengeSize))
    else :
        challengeList = []
        tmp = []
        i = 0
        while (i < numOfChallenges):
            tmp = [random.randint(0,1) for b in range(0,challengeSize)]
            if tmp not in challengeList:
                challengeList.append(tmp)
                i += 1
        return challengeList
    #asked for more Challenges then combinations possible
    raise RuntimeError('numOfChallenges > 2 ^ challengeSize')

class pufEval(object):
    
    def __init__(self, numOfMultiplexer, RNDBaseInstance, numOfChallenges, MutatorBaseInstance, numOfPufs, numOfThreads):
        
        #not nice, I know (refactor it ...)
        if isinstance(numOfMultiplexer, ( int, long )):
            self.numOfMultiplexer = numOfMultiplexer
        else:
            raise RuntimeError('1 Argument (numOfMultiplexer) is not an Int/Long')
        
        if isinstance(RNDBaseInstance, RNDBase):
            self.RNDBaseInstance = RNDBaseInstance
        else:
            raise RuntimeError('2 Argument (RNDBaseInstance) is not of the Typ RNDBase')
        
        if isinstance(numOfChallenges, ( int, long )):
            self.numOfChallenges = numOfChallenges
        else:
            raise RuntimeError('3 Argument (numOfChallenges) is not an Int/Long')

        if isinstance(MutatorBaseInstance, MutatorBase):
            self.MutatorBaseInstance = MutatorBaseInstance
        else:
            raise RuntimeError('4 Argument (MutatorBaseInstance) is not of the Typ MutatorBase')
        
        if isinstance( numOfPufs, ( int, long )):
            self.numOfPufs = numOfPufs
        else:
            raise RuntimeError('5 Argument (numOfPufs) is not an Int/Long')
        
        if isinstance(numOfThreads, ( int, long )):
            self.numOfThreads = numOfThreads
        else:
            raise RuntimeError('6 Argument (numOfThreads) is not an Int/Long')
        
        #creating puf instances
        self.pufList = []
        for i in range(0, self.numOfPufs):
            self.pufList.append(puf(self.RNDBaseInstance, self.numOfMultiplexer))

    def run(self):
        #calculating list ranges for multi core processing
        rest = self.numOfPufs % self.numOfThreads
        pufListRanges = []
        for j in range(0, self.numOfThreads):
            pufListRanges.append([(j * ((self.numOfPufs - rest) / self.numOfThreads)), ((j + 1) * ((self.numOfPufs - rest) / self.numOfThreads))])
        pufListRanges[self.numOfThreads - 1][1] = pufListRanges[self.numOfThreads - 1][1] + rest
        
        
        qList = Queue()
        pList = []
        
        startTime = time.time()
        
        for i in range(0, self.numOfThreads):
            #self.run(self.pufList[pufListRanges[i][0] : (pufListRanges[i][1] -1)], (pufListRanges[i][1] - pufListRanges[i][0]),self.numOfChallenges, self.numOfMultiplexer, self.MutatorBaseInstance, qList)
            pList.append( Process(target=runThread, args=(self.pufList[pufListRanges[i][0] : (pufListRanges[i][1])], (pufListRanges[i][1] - pufListRanges[i][0]),self.numOfChallenges, self.numOfMultiplexer, self.MutatorBaseInstance, qList)))
            pList[i].start()

        
        result = []
        for j in range(0, self.numOfThreads):
            #set block=True to block until we get a result
            result.extend(qList.get(True))
        
        endTime = time.time()
        print("Total calculation time: " + str(endTime - startTime))
        
        return result
        
    def runPrint(self):
        result = self.run()
        for k in range(0, self.numOfPufs):
            for l in range(0, self.numOfChallenges):
                print(result[k][l],)
            print()
    
    def runStats(self):
        result = self.run()
        stats = []
        tmp = 0
        for k in range(0, self.numOfPufs):
            for l in range(0, self.numOfChallenges):
                if (result[k][l][0] != result[k][l][1]):
                    tmp += 1
            stats.append(tmp/float(self.numOfChallenges))
            tmp = 0
        return stats

    def runPlot(self, save):
        stats = self.runStats()
        n, bins, patches = plt.hist(stats, 20, normed=1, facecolor='y')
        if save:
            name = 'Hist-k' + str(self.numOfMultiplexer) + 'NrOfPuf' + str(self.numOfPufs) + 'NrOfChal' + str(self.numOfChallenges)
            plt.savefig(name, bbox_inches='tight')
        else:
            plt.show()
        
        
#function for multiprocessing, not part of class because this solution seems to be faster
def runThread(pufList, pufListLen, numOfChallenges, numOfMultiplexer, MutatorBaseInstance, qList):
    result = [[0 for x in range(numOfChallenges)] for x in range(pufListLen)]
    challengeList = []
    for i in range(0, pufListLen):
        challengeList = genChallengeList(numOfMultiplexer, numOfChallenges)
        for j in range(0, numOfChallenges):
            result[i][j] = (pufList[i].challengeBit(challengeList[j]), pufList[i].challengeBit(MutatorBaseInstance.mutateChallenge(challengeList[j], numOfMultiplexer)))
    qList.put(result)

#base class for challenge mutation
class MutatorBase(object):
    
    __metaclass__ = ABCMeta

    @abstractmethod
    def mutateChallenge(self, challenge, length):
        pass
        
#last bit switch mutator
class MutatorLastBitSwitch(MutatorBase):
    
    def mutateChallenge(self, challenge, length):
        challenge[length-1] = challenge[length-1] ^ 1
        return challenge

#last bit switch mutator
class MutatorMiddleBitSwitch(MutatorBase):
    
    def mutateChallenge(self, challenge, length):
        challenge[length/2] = challenge[length/2] ^ 1
        return challenge

#switch a (single) bit chosen when initialized
class MutatorBitSwitch(MutatorBase):

    def __init__(self, i):
        self.idx = i

    def mutateChallenge(self, challenge, length):
        challenge[self.idx] = challenge[self.idx] ^ 1
        return challenge
