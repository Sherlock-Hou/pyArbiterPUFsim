# -*- coding: utf-8 -*-

import pufsim
import time

#create random generator instance
rndgen = pufsim.RNDUniform()
rndgen2 = pufsim.RNDNormal()
#create pufsim with 2 Multiplexer instances
pufsimu = pufsim.puf(rndgen, 7)
#do a single challenge to the pufsim with a challenge as list
pufsimu.challengePrint([1,1,1,1,1,1,1])
pufsimu.challengePrint([1,1,1,1,1,1,0])

pufsimu.challengePrint([0,1,1,1,1,1,1])

#worst case
#print len(pufsim.genChallengeList(10,((2**10)-1)))

combinerSimu = pufsim.simpleCombiner(rndgen2, 8)
combinerSimu.challengePrint([0,1,1,1,1,1,1,1])


mutatio = pufsim.MutatorLastBitSwitch()
mutatio2 = pufsim.MutatorMiddleBitSwitch()


#Aufruf hat sich verändert, zu testende Klasse muss übergeben werden.
tryEval = pufsim.pufEval(pufsim.simpleCombiner, 16, rndgen2, 2**8, mutatio, 1000, 4)
#tryEval.runStats()
#tryEval.run()
tryEval.runPlot(0)



