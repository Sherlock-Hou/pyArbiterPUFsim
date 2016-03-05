import pufsim

#create random generator instance
rndgen = pufsim.RNDUniform()
#create pufsim with 2 Multiplexer instances
pufsimu = pufsim.puf(rndgen, 7)
#do a single challenge to the pufsim with a challenge as list
pufsimu.challengeSingle([1,1,1,1,1,1,1])
pufsimu.challengeSingle([1,1,1,1,1,1,0])

pufsimu.challengeSingle([0,1,1,1,1,1,1])

#worst case
print len(pufsim.genChallengeList(10,((2**10)-1)))

mutatio = pufsim.MutatorLastBitSwitch()

tryEval = pufsim.pufEval(8, rndgen, 2**8, mutatio, 1000, 4)
print "done"
print tryEval.run()


