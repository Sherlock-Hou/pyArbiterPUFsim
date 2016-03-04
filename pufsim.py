# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import itertools
import random

#base class for random time generation of a single multiplexer
class RNDBase(object):
    
    __metaclass__ = ABCMeta

    @abstractmethod
    def generateTimes(self):
        pass
        
#Uniform distribution
#generateTimes(self) returns 4 floats as tuple (Quadrupel?) 
#Order of Values: up_up, down_down, up_down, down_up
class RNDStuipdGeneator(RNDBase):
    
    def generateTimes(self):
        return (random.random(),random.random(),random.random(),random.random())

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
        
    def challengeSingle(self, bitList):
        time_up, time_down = self.challenge(bitList)
        print '{' + ', '.join(map(str, bitList)) + '}\t' + str(time_up - time_down)
