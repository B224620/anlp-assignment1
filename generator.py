import sys
from math import log, isclose, prod
from collections import defaultdict

import numpy
#import matplotlib.pyplot as plt
import numpy as np  # numpy provides useful maths and vector operations
from numpy.random import random_sample


def generate_random_sequence(distribution, N):
    ''' generate_random_sequence takes a distribution (represented as a
    dictionary of outcome-probability pairs) and a number of samples N
    and returns a list of N samples from the distribution.
    This is a modified version of a sequence generator by fraxel on
    StackOverflow:
    http://stackoverflow.com/questions/11373192/generating-discrete-random-variables-with-specified-weights-using-scipy-or-numpy
    '''
    #As noted elsewhere, the ordering of keys and values accessed from
    #a dictionary is arbitrary. However we are guaranteed that keys()
    #and values() will use the *same* ordering, as long as we have not
    #modified the dictionary in between calling them.
    outcomes = numpy.array(list(distribution.keys()))
    probs = numpy.array(list(distribution.values()))
    #make an array with the cumulative sum of probabilities at each
    #index (ie prob. mass func)
    bins = numpy.cumsum(probs)
    #create N random #s from 0-1
    #digitize tells us which bin they fall into.
    #return the sequence of outcomes associated with that sequence of bins
    #(we convert it from array back to list first)
    return list(outcomes[numpy.digitize(random_sample(N), bins)])

def normalize_counts(counts):
    ''' normalize_counts takes a dictionary of counts as an argument and
    returns a corresponding dictionary of probabilities by normalizing
    the counts to sum to 1.
    '''
    ## students need to fill in correct function
    totalvalue = sum(counts.values())
    normalcounts = {k:v / totalvalue for k, v in counts.items()}
    #print(str(sum(normalcounts.values())))
    return normalcounts