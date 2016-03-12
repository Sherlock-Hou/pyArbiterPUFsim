import pufsim

rndgen = pufsim.RNDNormal()
k = 16
challengeSampleSize = 2**4
pfSampleSize = 100

res = []
for i in xrange(k):
    mutator = pufsim.MutatorBitSwitch(i)
    stats = pufsim.pufEval(k, rndgen, challengeSampleSize, mutator, pfSampleSize, 4).runStats()
    res.append(sum(stats)/float(len(stats)))

print "for a " + str(k) + "bit arbiter PF:"
print "probability of output flip when kth bit is flipped:"
print
print res
print
print "(determined using a sample of " + str(pfSampleSize) + " arbiter PFs with Gaussian distributed"
print "delay values and a sample of " + str(challengeSampleSize) + " of uniformly distributed"
print "challenges each, for each 1 <= k <= " + str(k) + ".)"
