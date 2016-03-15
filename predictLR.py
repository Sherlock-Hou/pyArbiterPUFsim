#!/cm/shared/apps/python/python3.4/bin/python3
# -*- coding: utf-8 -*-

#SBATCH --mail-user=wisiol@zedat.fu-berlin.de
#SBATCH --job-name=arb-pf-test
#SBATCH --mail-type=end
#SBATCH --mem=2048
#SBATCH --time=08:00:00

import pufsim
from math import exp
from numpy import around, sign, zeros
#from scipy.spatial import distance
import sys
from copy import copy

class ArbiterLR():

    class triGen:

        prev = None
        cur = None
        nxt = None

        def __init__(self, default):
            self.default = default
            self.prev = copy(default)
            self.cur = copy(default)
            self.nxt = copy(default)

        def iterate(self):
            self.prev = self.cur
            self.cur = self.nxt
            self.nxt = copy(self.default)

    # return the (standard) scalar product of two vectors x and y
    def sprod(self, x, y):
        if len(x) != len(y):
            raise RuntimeError('Both arguments must be vectors of the same length.')
        return sum([ x[i]*y[i] for i in range(len(x)) ])

    # logistic function
    def h(self, x, Θ):
        p = -self.sprod(Θ, x)
        try:
            if p > 500:
                # avoid overflow
                return 0
            return 1.0 / (1 + exp(p))
        except OverflowError as e:
            sys.stdout.write("\nh: overflow for p=" + str(p) + "x=" + str(["%8f" % e for e in x]) + ", Θ=" + str(["%8f" % e for e in Θ]))
            raise e

    # compute the new value for Θ[j]
    def iterateΘj(self, Θ, j, α, m, tSet):
        return Θ[j] - α/m * sum([ tSet[i][0][j] * ( self.h(tSet[i][0], Θ) - tSet[i][1] ) for i in range(k)])

    # compute input product
    def inputProd(self, c):
        return [
            (-1)**sum([c[j] for j in range(i,self.k)])
            for i in range(len(c))
        ]

    def __init__(self, k, m, M, n):
        # experiment parameters
        self.k = k # pf size
        self.m = min([2**k, m]) # training set size
        self.M = M # number of wrong CRPs
        self.n = min([2**k, n]) # check set size
        self.convergeDecimals = 8 # number of decimals expected to be equal after one iteration
        self.maxTrainingIteration = 100

        # create pufsim with k multiplexer instances
        self.arbiterpf = pufsim.puf(pufsim.RNDNormal(), k)

    def train(self):
        # sample training set
        tChallenges = pufsim.genChallengeList(self.k, self.m + self.M)
        # add correct challenges
        tSet = [ ([1] + self.inputProd(c), self.arbiterpf.challengeBit(c)) for c in tChallenges[:self.m] ]
        # add wrong challenges
        tSet += [ ([1] + self.inputProd(c), 1-self.arbiterpf.challengeBit(c)) for c in tChallenges[self.m:] ]

        converged = False
        i = 0

        # RPROP parameters
        ηplus = 1.2
        ηminus = 0.5
        Δmin = 10**-6
        Δmax = 50

        # learned delay values
        Θ = self.triGen([ 1 for x in range(self.k+1) ]) # note k+1

        # partial derivatives of error function
        pE = self.triGen(default=[ 0 for x in range(self.k+1) ])

        # update values for Θ
        Δ = self.triGen(default=[ 0 for x in range(self.k+1) ]) # note k+1
        Δ.cur = [ 1 for x in range(self.k+1) ] # init for first iteration
        ΔΘ = self.triGen(default=[ 0 for x in range(self.k+1) ]) # note k+1

        try:
            while (not converged and i < self.maxTrainingIteration):
                # count iterations
                i += 1

                # compute new Θ (RPROP algorithm)
                for j in range(self.k+1):
                    # compute pE.cur[j]
                    pE.cur[j] = 1/float(self.m+self.M) * sum([ tSet[i][0][j] * ( self.h(tSet[i][0], Θ.cur) - tSet[i][1] ) for i in range(self.m+self.M)])

                    # compute Θ.nxt[j]
                    if pE.prev[j]*pE.cur[j] > 0:
                        Δ.cur[j] = min([Δ.prev[j] * ηplus, Δmax])
                        ΔΘ.cur[j] = -sign(pE.cur[j]) * Δ.cur[j]
                        Θ.nxt[j] = Θ.cur[j] + ΔΘ.cur[j]

                    elif pE.prev[j]*pE.cur[j] < 0:
                        Δ.cur[j] = max([Δ.prev[j] * ηminus, Δmin])
                        Θ.nxt[j] = Θ.cur[j] - ΔΘ.prev[j]
                        pE.cur[j] = 0

                    elif pE.prev[j]*pE.cur[j] == 0:
                        ΔΘ.cur[j] = -sign(pE.cur[j]) * Δ.cur[j]
                        Θ.nxt[j] = Θ.cur[j] + ΔΘ.cur[j]

                sys.stdout.write(" ")

                # iterate the triGens Δ, ΔΘ, Θ, pE
                Δ.iterate()
                ΔΘ.iterate()
                Θ.iterate()
                pE.iterate()

                # check for convergence
                converged = (around(Θ.prev,decimals=self.convergeDecimals) == around(Θ.cur,decimals=self.convergeDecimals)).all()
                #sys.stdout.write("Θ(" + str(i) + "): " + str(["%8f" % e for e in Θ.cur]) + "; ")
                #sys.stdout.write(str(i) + "th iteration -- current distance: " + str(round(distance.euclidean(Θ.prev, Θ.cur), convergeDecimals+2)) + "\n")

        except OverflowError as e:
            #print()
            print("OVERFLOW OCCURED, USING LAST KNOWN Θ [" + str(e) + "]")
            Θ.cur = Θ.prev

        self.Θ = Θ.cur
        return self.Θ

    def check(self):
        # assess quality
        cChallenges = pufsim.genChallengeList(self.k, self.n)
        good = 0
        bad = 0
        for c in cChallenges:
            pfResponse = self.arbiterpf.challengeBit(c)
            lrResponse = 0 if self.sprod(self.Θ, [1] + self.inputProd(c)) < 0 else 1
            if pfResponse == lrResponse:
                good += 1
            else:
                bad += 1
                #print("got " + str(lrResponse) + " but expected " + str(pfResponse))

        return float(good)/float(good+bad)

    def run(self):
        self.train()
        return self.check()

result = zeros((10, 6, 10))
for iIdx in range(0, 10):
    for mIdx in range(0, 10):
        m = 100 * (mIdx+1)
        for MIdx in range(6):
            try:
                M = int(m * (MIdx/10.0))
                lr = ArbiterLR(k=64, m=m, M=M, n=10000)
                sys.stdout.write("%s,%s,%s,%s,%s," % (m,M,mIdx,MIdx,iIdx))
                result[mIdx][MIdx][iIdx] = lr.run()
                sys.stdout.write("%s\n" % result[mIdx][MIdx][iIdx])
            except Exception as e:
                sys.stdout.write("%s\n" % str(e))

            #sys.stderr.write(str(result) + "\n\n###################\n\n")
