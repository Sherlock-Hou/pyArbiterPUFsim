# ML attack on arbiter pufs with pybrain library

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.supervised.trainers import RPropMinusTrainer
import pufsim

def changeChallenge(challenge):
    ''' changes challenge to {-1,1} from {0,1} '''
    res = []
    for i in challenge:
        if i == 1:
            res.append(1)
        else:
            res.append(-1)
    return res


class ml_attack(object):

    """docstring for ml_attack"""
    def __init__(self, pufsimu, pufSize, numOfchallenges):
        self.pufSize = pufSize
        self.numOfchallenges = numOfchallenges
        self.pufsimu = pufsimu

    def run(self):

        # create challenge list
        self.chall_list = pufsim.genChallengeList(self.pufSize, self.numOfchallenges)

        # build network
        self.net = buildNetwork(self.pufSize, 2**self.pufSize, 1, bias=True)

        # set dataset
        self.ds = SupervisedDataSet(self.pufSize, 1)

        for challenge in self.chall_list:
            # chg = changeChallenge(challenge)
            # chg_res = 1 if self.pufsimu.challengeBit(challenge) == 1 else -1
            # self.ds.addSample(chg, [chg_res,])

            # add challenges multiple times
            self.ds.addSample(challenge, [self.pufsimu.challengeBit(challenge),])
            self.ds.addSample(challenge, [self.pufsimu.challengeBit(challenge),])
            self.ds.addSample(challenge, [self.pufsimu.challengeBit(challenge),])
            self.ds.addSample(challenge, [self.pufsimu.challengeBit(challenge),])


        # create trainer
        self.trainer = BackpropTrainer(self.net,
                                        dataset=self.ds,
                                        learningrate = 0.01,
                                        momentum = 0.,
                                        weightdecay=0.)
        #self.trainer.setData(self.ds)

        # train network
        a = self.trainer.trainUntilConvergence(dataset=self.ds,
                                    #trainingData=self.ds,
                                    #validationData=self.ds,
                                    maxEpochs=100,
                                    verbose=False,
                                    continueEpochs=10,
                                    validationProportion=0.25)

        return 0


    def validate(self):
        complete_chall_list = pufsim.genChallengeList(self.pufSize, 2 ** self.pufSize)
        corr_count = 0

        for challenge in complete_chall_list:
            pufResult = self.pufsimu.challengeBit(challenge)

            # chg = changeChallenge(challenge)
            chg = challenge

            annResult = -1
            if (self.net.activate(chg)[0] >= 0.5):
                annResult = 1
            else:
                annResult = 0

            #print "challenge:", challenge, "with result", pufResult, "and ANN result", annResult, "from", self.net.activate(chg)[0]
            if (pufResult == annResult):
                corr_count = corr_count + 1
        print "Possible challenges:", 2**self.pufSize, "with correct guesses:", corr_count
        print "Ratio", (float(corr_count) / 2 ** self.pufSize)

        return (float(corr_count) / 2 ** self.pufSize)


ratio_count = 0
num_try = 10
pufSize = 8
# create random generator instance
rndgen = pufsim.RNDUniform()
# create pufsim with pufSize Multiplexer instances
pufsimu = pufsim.puf(rndgen, pufSize)

# compute bias
zero_count = 0
chall_list = pufsim.genChallengeList(pufSize, 2**pufSize)
for challenge in chall_list:
    if (pufsimu.challengeBit(challenge) == 0):
        zero_count = zero_count + 1
bias = zero_count / float(2**pufSize)
b = 0.5 + abs(0.5 - bias)
print bias

# try to break puf num_try times
for i in range(0,num_try):
    ml = ml_attack(pufsimu, pufSize, 20)
    ml.run()
    ratio_count = ratio_count + b + abs(b - ml.validate())
print (ratio_count / num_try)



#################################
# test for 2-input xor
#################################

# build network
# net = buildNetwork(2, 4, 1, bias=True)
#
# # set dataset
# # create dataset with 2 inputs and 1 output
# ds = SupervisedDataSet(2, 1)
# # add samples
# null = 0
# eins = 1
# ds.addSample((0, 0), [null,])
# ds.addSample((0, 1), (1,))
# ds.addSample((1, 0), (1,))
# ds.addSample((1, 1), [null,])
# ds.addSample((0, 0), [null,])
# ds.addSample((0, 1), (1,))
# ds.addSample((1, 0), (1,))
# ds.addSample((1, 1), [null,])
# ds.addSample((0, 0), [null,])
# ds.addSample((0, 1), (1,))
# ds.addSample((1, 0), (1,))
#
#
#
# # create trainer
# trainer = BackpropTrainer(net, ds, learningrate = 0.01, momentum = 0.99)
# # train network
# #trainer.train()
# trainer.trainUntilConvergence(verbose=False,
#                               maxEpochs=1000,
#                               validationData=ds)
#
# # trainer.trainOnDataset(ds, 1000)
# # trainer.testOnData()
#
# # p = net.activateOnDataset( ds )
# #
# # print p
#
# print '0,0->', net.activate([0,0]), round(net.activate([0,0]), 0)
# print '0,1->', net.activate([0,1]), round(net.activate([0,1]), 0)
# print '1,0->', net.activate([1,0]), round(net.activate([1,0]), 0)
# print '1,1->', net.activate([1,1]), round(net.activate([1,1]), 0)
