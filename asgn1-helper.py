#Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
from random import random
from math import log
from collections import defaultdict
from generator import *

tri_counts=defaultdict(int) #counts of all trigrams in input


#function takes line from f output, stores in new string in lower case, checks against vaild set and truncates invalid characters
def preprocess_line(line):
    newline = line.lower()
    validchars = " 0123456789qwertyuiopasdfghjklzxcvbnm."
    for char in line:
        if char not in validchars:
            newline = newline.replace(char, '')
    return newline


#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1] #get input argument: the training file

#This bit of code gives an example of how you might extract trigram counts
#from a file, line by line. If you plan to use or modify this code,
#please ensure you understand what it is actually doing, especially at the
#beginning and end of each line. Depending on how you write the rest of
#your program, you may need to modify this code.
with open('training.es', encoding='utf-8') as f: #added encoding to process utf-8 characters
    for line in f:
        line = preprocess_line(line) #does something!!
        for j in range(len(line)-(3)):
            trigram = line[j:j+3]
            #if trigram not in tri_counts:
                #print('storing ' + trigram + ' to dict....')
            tri_counts[trigram] += 1

#Some example code that prints out the counts. For small input files
#the counts are easy to look at but for larger files you can redirect
#to an output file (see Lab 1).
print("Trigram counts in ", infile, ", sorted alphabetically:")
for trigram in sorted(tri_counts.keys()):
    print(trigram, ": ", tri_counts[trigram])
print("Trigram counts in ", infile, ", sorted numerically:")
for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
    print(tri_count[0], ": ", str(tri_count[1]))


