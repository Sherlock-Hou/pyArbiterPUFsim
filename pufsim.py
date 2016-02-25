# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import itertools
import random

class RNDJesus(object):
    
    __metaclass__ = ABCMeta

    @abstractmethod
    def generateTimes(self):
        pass
        
#erbt von RNDJesus und implementiert eine Methode
#generateTimes(self) gibt 4 float Werte als Tupel (Quadrupel?) zurück 
#Reihenfolge der Werte: up_up, down_down, up_down, down_up)
class RNDStuipdGeneator(RNDJesus):
    
    def generateTimes(self):
        return (random.random(),random.random(),random.random(),random.random())


class multiplexer(object):
        
    def __init__(self, timeTuple):
        self.up_up, self.down_down, self.up_down, self.down_up = timeTuple
    
    def challenge(self, bit, time_up, time_down):
        if bit == 0:
            return (time_up + self.up_up), (time_down + self.down_down)
        elif bit == 1:
            return (time_down + self.down_up), (time_up + self.up_down)
        else:
            raise RuntimeError('Bit is not 0 or 1')

class puf(object):
    
    def __init__(self, gen, numOfMultip):
        self.gen = gen
        self.numOfMultip = numOfMultip
        self.multiplexerList = []
        
        for i in range(0, self.numOfMultip):
            tmp = multiplexer(gen.generateTimes())
            self.multiplexerList.append(tmp)
    
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
        
    def challengeSingle(self, bitList):
        time_up, time_down = self.challenge(bitList)
        print '{' + ', '.join(map(str, bitList)) + '}\t' + str(time_up - time_down)
        
    def challengeList(self, showDif):
        if showDif > 1:
            raise RuntimeError('Param has to be 0 or 1 (0 -> Bit view 1 -> TimeDif view')
        
        challengeList = map(list, itertools.product([0, 1], repeat=self.numOfMultip))
        
        for chal in challengeList:
            time_up, time_down = self.challenge(chal)
            time_dif = time_up - time_down
            if showDif == 0:
                if time_dif <= 0:
                    print '{' + ', '.join(map(str, chal)) + '}\t' + '0'
                else:
                    print '{' + ', '.join(map(str, chal)) + '}\t' + '1'
            else:
                print '{' + ', '.join(map(str, chal)) + '}\t' + str(time_dif)
            
        

