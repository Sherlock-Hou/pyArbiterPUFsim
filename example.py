import pufsim

#create random generator instance
rndgen = pufsim.RNDStuipdGeneator()
#create pufsim with 2 Multiplexer instances
pufsimu = pufsim.puf(rndgen, 7)
#do a single challenge to the pufsim with a challenge as list
pufsimu.challengeSingle([1,1,1,1,1,1,1])
pufsimu.challengeSingle([1,1,1,1,1,1,0])

pufsimu.challengeSingle([0,1,1,1,1,1,1])

#get a list of all challenges accordingly to the size of the pufsim (number of multiplexer)
# Param: 0 Bit view 1: time difference view
#pufsimu2 = pufsim.puf(rndgen, 5)
#pufsimu2.challengeList(1)


