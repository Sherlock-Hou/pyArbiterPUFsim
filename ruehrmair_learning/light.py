from predictor import *
from PUFmodels import *
from scipy import mean, empty, array
import time


def xorKnackertester(bitzahl, numxor, genauigkeit, genauigkeit2, wieoft, CRParray, timestamp):
    ArbPUFgoal = XORArbPUF(bitzahl, numxor, 'lightweight')
    #file=open(timestamp+'goalParam','w')
    #pickle.dump(ArbPUFgoal.getParam(), file)
    #file.close()
    for i in range(CRParray.size):
        xorKnacker(bitzahl, numxor, genauigkeit, genauigkeit2, ArbPUFgoal, wieoft, int(CRParray[i]), timestamp)


def xorKnacker(bitzahl, numxor, genauigkeit, genauigkeit2, ArbPUFgoal, wieoft, CRP, timestamp):

    sucess = 0
    oft = 0
    mc_rate = MCError()

    print CRP

    while sucess < wieoft:

        erf = LRError()
        features = ArbPUFgoal.calc_features(ArbPUFgoal.generate_challenge(CRP))
        set = TrainData(features, ArbPUFgoal.bin_response(features))

        testfeatures = ArbPUFgoal.calc_features(ArbPUFgoal.generate_challenge(10000))
        testtargets = ArbPUFgoal.bin_response(testfeatures)

        performanceTrain = 1
        count = 0
        start = time.time()

        while performanceTrain > genauigkeit:

            count += 1

            model = prodLinearPredictor(bitzahl + 1, numxor)

            lesson = BasicTrainable(set, model, erf)
            learner = GradLearner(lesson, RProp([bitzahl + 1] * numxor), Closures(accuracy=genauigkeit2).grad_performance_stop)
            learner.evaluate_lesson()

            performanceTrain = mc_rate.calc(lesson.trainset.targets, lesson.response()) / set.targets.size

            print count, '.)', 'MCrate(train):', performanceTrain, 'time since start:', start - time.time()
            f = open(timestamp + 'result_' + repr(bitzahl) + '_' + repr(numxor) + '_' + repr(CRP) + '.dat', 'a')
            if performanceTrain > genauigkeit:
                f.write('0 ')
            else:
                f.write('1 ')
            f.close()

            #trials = open(timestamp+'trials_'+repr(bitzahl)+'_'+repr(numxor)+'_'+repr(CRP)+'.dat', 'a')
            #trials.write(repr(count2)+' ')
            #trials.close()

        performanceTest = mc_rate.calc(testtargets, model.response(testfeatures)) / testtargets.shape[0]
        print 'MCrate: (test)', performanceTest, 'time since start:', start - time.time()

        f = open(timestamp + 'mctest_' + repr(bitzahl) + '_' + repr(numxor) + '_' + repr(CRP) + '.dat', 'a')
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

        zeit = open(timestamp + 'zeit_' + repr(bitzahl) + '_' + repr(numxor) + '_' + repr(CRP) + '.dat', 'a')
        zeit.write(repr(ende - start) + ' ')
        zeit.close()

        oft += 1

    #print 'meanValues','MCrate:', mean(mcrate), 'CRPs:', mean(crps), 'time[s]:', mean(zeit)
    print 'finished'

if __name__ == '__main__':
    # bitzahl, numxor, genauigkeit, genauigkeit2, wieoft, CRParray, timestamp
    xorKnackertester(64, 5, 0.05, 0.01, 10, array([10000]), 'Test')
