import pufsim
import time

#create random generator instance
rndgen = pufsim.RNDUniform()
rndgen2 = pufsim.RNDNormal()
#create pufsim with 2 Multiplexer instances
pufsimu = pufsim.puf(rndgen, 7)
#do a single challenge to the pufsim with a challenge as list
pufsimu.challengeSingle([1,1,1,1,1,1,1])
pufsimu.challengeSingle([1,1,1,1,1,1,0])

pufsimu.challengeSingle([0,1,1,1,1,1,1])

#worst case
#print len(pufsim.genChallengeList(10,((2**10)-1)))



mutatio = pufsim.MutatorLastBitSwitch()
mutatio2 = pufsim.MutatorMiddleBitSwitch()



tryEval = pufsim.pufEval(16, rndgen2, 2**8, mutatio, 1000, 4)
#tryEval.runStats()
#tryEval.run()
tryEval.runPlot(1)



