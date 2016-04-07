import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from PUFmodels import *
from bent import bentKnacker


def plot_results(num_challenges, error_rate_normal, error_rate_interpol, title):
    num_cha = num_challenges
    prc_normal = error_rate_normal
    prc_interpol = error_rate_interpol
    red_patch, = plt.plot(num_cha, prc_normal, 'ro')
    blue_patch, = plt.plot(num_cha, prc_interpol, 'bs')
    plt.axis([num_cha[0], num_cha[-1], 0, 0.5])
    plt.xlabel("# of challenges")
    plt.ylabel("Error rate")
    plt.title(title)

    plt.legend([red_patch, blue_patch], ['Normal Challenges', 'Last Two Bits Flipped'])

    plt.show()







if __name__ == '__main__':
    #PUF Config
    # number_of_bit = 8
    # number_of_pufs = 2
    #
    # #Kracker config
    # number_of_CRP = 1000
    # accuracy_1 = 0.05
    # accuracy_2 = 0.01
    #
    #
    #
    # #create Bent Arbiter PUF
    # BentPUFgoal = BentArbPUF(number_of_bit, number_of_pufs)
    #
    # #create challenges
    # challenges = BentPUFgoal.generate_challenge(number_of_CRP)
    #
    # #train
    # result = bentKnacker(number_of_bit, number_of_pufs, accuracy_1, accuracy_2, BentPUFgoal, 1, number_of_CRP, 'Test')
    # print 'Result', result

    #PUF Config
    #range of Puf sizes
    number_of_bits = [2, 8]
    #range of number of Arbiter Pufs
    number_of_pufs = [2, 4]
    #number of instances per Bent Puf to calc average
    number_of_instances = 1

    #Kracker config
    number_of_CRP = 1000
    accuracy_1 = 0.05
    accuracy_2 = 0.01

    result = []

    for pufs in range(number_of_pufs[1], number_of_pufs[2]+1):
        for bits in range(number_of_bits[1], number_of_bits[2]+1):
            BentPUFgoal = BentArbPUF(bits, pufs)
            challenges = BentPUFgoal.generate_challenge(number_of_CRP)
            result = bentKnacker(bits, pufs, accuracy_1, accuracy_2, BentPUFgoal, 1, number_of_CRP, 'Test', challenges)

