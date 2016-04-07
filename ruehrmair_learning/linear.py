# attack on linear arbiter puf

from predictor import *
from PUFmodels import *
from scipy import mean, empty, array
import time


def linKnackertester(bitzahl, genauigkeit, genauigkeit2, wieoft, CRParray, timestamp):
    ArbPUFgoal = linArbPUF(bitzahl)

    #file=open(timestamp+'goalParam','w')
    #pickle.dump(ArbPUFgoal.getParam(), file)
    #file.close()
    for i in range(CRParray.size):
        linKnacker(bitzahl, genauigkeit, genauigkeit2, ArbPUFgoal, wieoft, int(CRParray[i]), timestamp)


def linKnacker(bitzahl, genauigkeit, genauigkeit2, ArbPUFgoal, wieoft, CRP, timestamp, challenges=None, bin_r=None):
    res = 0.
    sucess = 0
    oft = 0
    mc_rate = MCError()

    print CRP

    while sucess < wieoft:

        erf = LRError()

        if (challenges is None):
            # create challenges
            features = ArbPUFgoal.calc_features(ArbPUFgoal.generate_challenge(CRP))
        else:
            # challenges are given
            features = ArbPUFgoal.calc_features(challenges)

        if (bin_r is None):
            # create responses
            bin_resp = ArbPUFgoal.bin_response(features)
        else:
            bin_resp = bin_r

        set = TrainData(features, bin_resp)

        testfeatures = ArbPUFgoal.calc_features(ArbPUFgoal.generate_challenge(10000))
        testtargets = ArbPUFgoal.bin_response(testfeatures)

        performanceTrain = 1
        count = 0
        start = time.time()

        while performanceTrain > genauigkeit:

            count += 1

            model = linearPredictor(bitzahl + 1)

            lesson = BasicTrainable(set, model, erf)
            learner = GradLearner(lesson, RProp([bitzahl + 1]), Closures(accuracy=genauigkeit2).grad_performance_stop)
            learner.evaluate_lesson()

            performanceTrain = mc_rate.calc(lesson.trainset.targets, lesson.response()) / set.targets.size

            print count, '.)', 'MCrate(train):', performanceTrain, 'time since start:', start - time.time()
            f = open(timestamp + 'result_' + repr(bitzahl) + '_' + repr(CRP) + '.dat', 'a')
            if performanceTrain > genauigkeit:
                f.write('0 ')
            else:
                f.write('1 ')
            f.close()

            #trials = open(timestamp+'trials_'+repr(bitzahl)+'_'+repr(numxor)+'_'+repr(CRP)+'.dat', 'a')
            #trials.write(repr(count2)+' ')
            #trials.close()

        performanceTest = mc_rate.calc(testtargets, model.response(testfeatures)) / testtargets.shape[0]
        res = performanceTest
        print 'MCrate: (test)', performanceTest, 'time since start:', start - time.time()

        f = open(timestamp + 'mctest_' + repr(bitzahl) + '_'  + repr(CRP) + '.dat', 'a')
        f.write(repr(performanceTest) + ' ')
        f.close()

        ende = time.time()
        sucess += 1

        #file=open(timestamp+'param_'+repr(sucess)+'_CRP_'+repr(CRP),'w')
        #pickle.dump(ArbCopy.getParam(), file)
        #file.close()
        #file=open(timestamp+'features_'+repr(sucess)+'_CRP_'+repr(CRP),'w')
        #pickle.dump(trainingCRP.features, file)
        #file.close()

        zeit = open(timestamp + 'zeit_' + repr(bitzahl) + '_' + repr(CRP) + '.dat', 'a')
        zeit.write(repr(ende - start) + ' ')
        zeit.close()

        oft += 1

    #print 'meanValues','MCrate:', mean(mcrate), 'CRPs:', mean(crps), 'time[s]:', mean(zeit)
    print 'finished'
    return res

if __name__ == '__main__':
    # bitzahl, genauigkeit, genauigkeit2, wieoft, CRParray, timestamp
    linKnackertester(64, 0.05, 0.01, 3, array([1000]), 'Test')
    ArbPUFgoal = linArbPUF(64)
    #linKnacker(64, 0.05, 0.01, ArbPUFgoal, 2, 600, 'Test')
