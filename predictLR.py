# -*- coding: utf-8 -*-

import pufsim
from math import exp

# return the (standard) scalar product of two vectors x and y
def sprod(x, y):
    if len(x) != len(y):
        raise RuntimeError('Both arguments must be vectors of the same length.')
    return sum([ x[i]*y[i] for i in range(len(x)) ])

# logistic function
def h(x, Θ):
    return 1.0 / (1 + exp(-sprod(Θ, x)))

# compute the new value for Θ[j]
def iterateΘj(Θ, j, α, m, tSet):
    return Θ[j] - α/m * sum([ tSet[i][0][j] * ( h(tSet[i][0], Θ) - tSet[i][1] ) for i in range(k)])

# compute input product
def inputProd(c):
    return [
        (-1)**sum([c[j] for j in range(i,k)])
        for i in range(len(c))
    ]

# experiment parameters
k = 32 # pf size
m = 1000 # training set size
n = 10000 # check set size
maxTrainingIteration = 5000

# create pufsim with k multiplexer instances
arbiterpf = pufsim.puf(pufsim.RNDNormal(), k)

# sample training set
tChallenges = pufsim.genChallengeList(k, min([2**k, m]))
tSet = [ ([1] + inputProd(c), (-1)**arbiterpf.challengeBit(c)) for c in tChallenges ]

# train LR
converged = False
i = 0
α = 1.0
Θ = [ 1 for x in range(k+1) ] # note k+1
try:
    while (not converged and i < maxTrainingIteration):
        i += 1
        oldΘ = Θ
        Θ = [ iterateΘj(oldΘ, j, α, m, tSet) for j in range(k+1) ] # note k+1
        converged = oldΘ == Θ
        #print("Θ " + str(i) + "th iteration: " + str([ round(e,3) for e in Θ ]))
except(OverflowError):
    print("OVERFLOW OCCURED, USING LAST KNOWN Θ")

# assess quality
cChallenges = pufsim.genChallengeList(k, min([2**k, n]))
good = 0
bad = 0
for c in cChallenges:
    pfResponse = arbiterpf.challengeBit(c)
    lrResponse = 1 if sprod(Θ, [1] + inputProd(c)) < 0 else 0
    if pfResponse == lrResponse:
        good += 1
    else:
        bad += 1

print("Learned delay values Θ: " + str([round(e,3) for e in Θ]))
print("Predicted correctly: " + str(good))
print("          falsely: " + str(bad))
print("Prediction rate: " + str(float(good)/float(good+bad)))
print("Terminated after " + str(i) + " iterations.")