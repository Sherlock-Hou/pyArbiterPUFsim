# attack on bent arb pufs

from predictor import *
from PUFmodels import *
from scipy import mean, empty, array
import time


def bentKnackertester(bitzahl, num_pufs, genauigkeit, genauigkeit2, wieoft, CRParray, timestamp):
    BentPUFgoal = BentArbPUF(bitzahl, num_pufs)
    #file=open(timestamp+'goalParam','w')
    #pickle.dump(BentPUFgoal.getParam(), file)
    #file.close()
    for i in range(CRParray.size):
        bentKnacker(bitzahl, num_pufs, genauigkeit, genauigkeit2, BentPUFgoal, wieoft, int(CRParray[i]), timestamp)


def bentKnacker(bitzahl, num_pufs, genauigkeit, genauigkeit2, BentPUFgoal, wieoft, CRP, timestamp, challenges=None, bin_resp=None):
    res = 0.
    sucess = 0
    oft = 0
    mc_rate = MCError()

    print CRP

    while sucess < wieoft:

        erf = LRError()

        if (challenges is None):
            # create challenges
            features = BentPUFgoal.calc_features(BentPUFgoal.generate_challenge(CRP))
        else:
            # challenges are given
            features = BentPUFgoal.calc_features(challenges)

        if (bin_resp is None):
            # create response
            bin_resp = BentPUFgoal.bin_response(features)

        set = TrainData(features, bin_resp)

        testfeatures = BentPUFgoal.calc_features(BentPUFgoal.generate_challenge(10000))
        testtargets = BentPUFgoal.bin_response(testfeatures)

        performanceTrain = 1
        count = 0
        start = time.time()

        while performanceTrain > genauigkeit:

            count += 1

            model = bentLinearPredictor(bitzahl + 1, num_pufs)

            lesson = BasicTrainable(set, model, erf)
            learner = GradLearner(lesson, RProp([bitzahl + 1] * num_pufs), Closures(accuracy=genauigkeit2).grad_performance_stop)
            learner.evaluate_lesson()

            performanceTrain = mc_rate.calc(lesson.trainset.targets, lesson.response()) / set.targets.size

            print count, '.)', 'MCrate(train):', performanceTrain, 'time since start:', start - time.time()
            f = open(timestamp + 'result_' + repr(bitzahl) + '_' + repr(num_pufs) + '_' + repr(CRP) + '.dat', 'a')
            if performanceTrain > genauigkeit:
                f.write('0 ')
            else:
                f.write('1 ')
            f.close()

            #trials = open(timestamp+'trials_'+repr(bitzahl)+'_'+repr(num_pufs)+'_'+repr(CRP)+'.dat', 'a')
            #trials.write(repr(count2)+' ')
            #trials.close()

        performanceTest = mc_rate.calc(testtargets, model.response(testfeatures)) / testtargets.shape[0]
        res = performanceTest
        print 'MCrate: (test)', performanceTest, 'time since start:', start - time.time()

        f = open(timestamp + 'mctest_' + repr(bitzahl) + '_' + repr(num_pufs) + '_' + repr(CRP) + '.dat', 'a')
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

        zeit = open(timestamp + 'zeit_' + repr(bitzahl) + '_' + repr(num_pufs) + '_' + repr(CRP) + '.dat', 'a')
        zeit.write(repr(ende - start) + ' ')
        zeit.close()

        oft += 1

    #print 'meanValues','MCrate:', mean(mcrate), 'CRPs:', mean(crps), 'time[s]:', mean(zeit)
    print 'finished'
    return res

if __name__ == '__main__':
    # bitzahl, numpufs, genauigkeit, genauigkeit2, wieoft, CRP Anzahl, timestamp
    bentKnackertester(64, 4, 0.05, 0.01, 10, array([10000]), 'Test')
