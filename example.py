import pufsim
import time

#create random generator instance
rndgen = pufsim.RNDUniform()
#create pufsim with 2 Multiplexer instances
pufsimu = pufsim.puf(rndgen, 7)
#do a single challenge to the pufsim with a challenge as list
pufsimu.challengeSingle([1,1,1,1,1,1,1])
pufsimu.challengeSingle([1,1,1,1,1,1,0])

pufsimu.challengeSingle([0,1,1,1,1,1,1])

#worst case
#print len(pufsim.genChallengeList(10,((2**10)-1)))



mutatio = pufsim.MutatorLastBitSwitch()

startTime = time.time()

tryEval = pufsim.pufEval(16, rndgen, 2**8, mutatio, 1000, 4)
tryEval.run()

endTime = time.time()
print  endTime - startTime


