# pylint: disable=C0103
""" Mapping the AQI to corresponding satellite data"""
import csv
import re
import numpy as np

def out(file):
    """read the datasets from csv files"""
    with open(file, 'r') as f:
        c = csv.reader(f, dialect='excel')
        s = 0
        count = 2014000
        output = []
        next(c, None)
        for row in c:
            if row[1] == "2014":
                if row[4] == "23":
                    output.append([count, s / 24])
                    s = 0
                    count += 1
                if row[5] == "NA":
                    continue
                s += int(row[5])
    return np.array(output[39:])


def inp(file):
    """Read the datasets from csv files"""
    with open(file, 'r') as f:
        band = []
        next(f, None)
        for line in f.readlines():
            # data treatment code here
            wordList = re.sub("[,]", " ", line).split()
            ret = []
            for word in wordList:
                word = float(word)
                ret.append(word)
            band.append(ret)
    return np.array(band)

def matching(a, b):
    """filtering data"""
    temp_a = a
    temp_b = b
    count = 0
    while len(temp_b) != len(temp_a):
        if temp_b[count, 0] != temp_a[count, 0]:
            temp_b = np.delete(temp_b, count, axis=0)
        count += 1
    return temp_a, temp_b
