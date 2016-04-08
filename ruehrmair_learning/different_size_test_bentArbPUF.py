import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from PUFmodels import *
from bent import bentKnacker

def plot_bent_puf(number_of_bits, number_of_pufs, results):
    X = np.arange(number_of_bits[0], number_of_bits[1])
    Y = np.arange(number_of_pufs[0], number_of_pufs[1])
    X, Y = np.meshgrid(X, Y)
    Z = results
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    ax.set_zlim(1, np.max(results))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('# of bits', linespacing=1)
    ax.set_ylabel('# of pufs', linespacing=1)
    ax.set_zlabel('Error rate', linespacing=1)
    ax.set_title('Bent puf combination test')
    plt.show()

if __name__ == '__main__':
    #PUF Config
    #range of Puf sizes (bits) {4,8,16,32,64}
    number_of_bits = [2, 8]
    #range of number of Arbiter Pufs {2,4,8,16,32}, just mod 2 allowed
    number_of_pufs = [2, 4]
    #number of instances per Bent Puf to calc average
    number_of_instances = 1

    #Kracker config
    number_of_CRP = 1000
    accuracy_1 = 0.05
    accuracy_2 = 0.01

    result = [[0 for i in range(number_of_pufs[1])] for n in range(number_of_bits[1])]

    #iterate through all combinations of number of pufs and number of bits
    for pufs in range(number_of_pufs[0], number_of_pufs[1]+1, 2):
        for bits in range(number_of_bits[0], number_of_bits[1]+1):
            instance_results = []
            #calculate multiple results for the same combiation
            for instance in range(number_of_instances):
                print "BentPuf dimentsion: bits", bits, "arbiterPufs", pufs, "instance", instance
                BentPUFgoal = BentArbPUF(bits, pufs)
                challenges = BentPUFgoal.generate_challenge(number_of_CRP)
                #result: error rate, time to calc
                instance_results.append(bentKnacker(bits, pufs, accuracy_1, accuracy_2, BentPUFgoal, 1, number_of_CRP, 'Test', challenges))
            #the mean of the error of all results of one combination - time not included
            print instance_results
            result[bits-1][pufs-1] = reduce(lambda (a, b), (x, y): a+x, instance_results)/len(instance_results) if len(instance_results) > 1 else instance_results[0][0]

    print "Result"
    print result

    plot_bent_puf(number_of_bits, number_of_pufs, result)