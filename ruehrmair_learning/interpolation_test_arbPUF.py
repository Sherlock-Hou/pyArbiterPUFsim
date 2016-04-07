import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import random

import linear
from PUFmodels import *

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



def flip_last_bit(chall, num_challenges, numbit):
    challenges = chall.copy()
    for i in range(1, num_challenges, 2):
        challenges[:, i:i+1] = challenges[:, i-1:i]
        challenges[numbit - 1:, i:i+1] = 1. if challenges[numbit - 1:, i-1:i] == 0. else 0.
    return challenges


def double_flip_last_bit(chall, num_challenges, numbit):
    c = chall.copy()
    challenges = random.randint(0, 2, [numbit, num_challenges * 2])
    for j in range(num_challenges):
        pos = j*2
        challenges[:, pos:pos+1] = c[:, j:j+1]
        challenges[:, pos+1:pos+2] = c[:, j:j+1]

    for i in range(1, num_challenges * 2, 2):
        challenges[:, i:i+1] = challenges[:, i-1:i]
        challenges[numbit - 1:, i:i+1] = 1. if challenges[numbit - 1:, i-1:i] == 0. else 0.
    return challenges

def flip_last_bit_responses(challenges, num_challenges, ArbPUF, double=False):
    response = ArbPUF.bin_response(ArbPUF.calc_features(challenges))
    num_challenges = 2*num_challenges if double else num_challenges
    for resp in range(1, num_challenges, 2):
        response[resp] = 1. if response[resp-1] == -1. else -1.

    return response

def flip_last_bit_learning(numbit = 64, num_challenges=1000, PUF=None):
    # create Linear Arbiter PUF
    ArbPUF = linArbPUF(numbit) if (PUF is None) else PUF
    # create challenge set and interpolate it
    challenges = ArbPUF.generate_challenge(num_challenges)
    interpolated_challenges = double_flip_last_bit(challenges, num_challenges, numbit)

    # interpolate response of challenges
    interpolated_bin_resp = flip_last_bit_responses(interpolated_challenges, num_challenges, ArbPUF, True)

    # learn with normal set and with interpolated set
    result_normal = linear.linKnacker(numbit, 0.05, 0.01, ArbPUF, 1, num_challenges, 'Test', challenges)
    result_interpol = linear.linKnacker(numbit, 0.05, 0.01, ArbPUF, 1, num_challenges, 'Test', interpolated_challenges, interpolated_bin_resp)

    print "Error rate normal" , result_normal
    print "Error rate interpol" , result_interpol

    return (result_normal, result_interpol)


def flip_two_bits(chall, num_challenges, numbit):
    challenges = chall.copy()
    for i in range(1, num_challenges, 2):
        challenges[:, i:i+1] = challenges[:, i-1:i]
        challenges[numbit-2:numbit-1, i:i+1] = 1. if challenges[numbit-2:numbit-1, i-1:i] == 0. else 0.
        challenges[numbit-1:, i:i+1] = 1. if challenges[numbit-1:, i-1:i] == 0. else 0.
    return challenges

def flip_two_bits_double(chall, num_challenges, numbit):
    c = chall.copy()
    challenges = random.randint(0, 2, [numbit, num_challenges * 2])
    for j in range(num_challenges):
        pos = j*2
        challenges[:, pos:pos+1] = c[:, j:j+1]
        challenges[:, pos+1:pos+2] = c[:, j:j+1]

    for i in range(1, num_challenges * 2, 2):
        challenges[:, i:i+1] = challenges[:, i-1:i]
        challenges[numbit-2:numbit-1, i:i+1] = 1. if challenges[numbit-2:numbit-1, i-1:i] == 0. else 0.
        challenges[numbit - 1:, i:i+1] = 1. if challenges[numbit - 1:, i-1:i] == 0. else 0.
    return challenges

def flip_two_bits_responses(challenges, num_challenges, ArbPUF, double=False):
    response = ArbPUF.bin_response(ArbPUF.calc_features(challenges))
    num_challenges = 2*num_challenges if double else num_challenges
    for resp in range(1, num_challenges, 2):
        response[resp] = 1. if response[resp-1] == 1. else -1.
    return response

def flip_two_bits_learning(numbit=64, num_challenges=1000, PUF=None):
    ArbPUF = linArbPUF(numbit) if (PUF is None) else PUF
    # create challenge set and interpolate it
    challenges = ArbPUF.generate_challenge(num_challenges)
    interpolated_challenges = flip_two_bits_double(challenges, num_challenges, numbit)

    interpolated_bin_resp = flip_two_bits_responses(interpolated_challenges, num_challenges, ArbPUF, True)

    result_normal = linear.linKnacker(numbit, 0.05, 0.01, ArbPUF, 2, num_challenges, 'Test', challenges)
    result_interpol = linear.linKnacker(numbit, 0.05, 0.01, ArbPUF, 1, num_challenges * 2, 'Test', interpolated_challenges, interpolated_bin_resp)

    print "Error rate normal" , result_normal
    print "Error rate interpol" , result_interpol

    return (result_normal, result_interpol)

if __name__ == '__main__':
    num_challenges = []
    error_rate_normal = []
    error_rate_interpol = []

    # how many times each number of challenges is tested
    how_many = 10
    # step between number of challenges
    steps = 50
    max_num_of_challenges = 200
    numbits = 64

    for num_cha in range(50, max_num_of_challenges, steps):
        num_challenges.append(num_cha)
        rate_norm = 0.
        rate_inter = 0.
        for i in range(0,how_many):
            (a, b) = flip_two_bits_learning(numbits, num_cha)
            rate_norm = rate_norm + a
            rate_inter = rate_inter + b

        error_rate_normal.append(rate_norm / how_many)
        error_rate_interpol.append(rate_inter / how_many)

    plot_results(num_challenges, error_rate_normal, error_rate_interpol, "Error rate 64Bit ArbPUF\nDouble CRPs and Flip Last Two Bits")
