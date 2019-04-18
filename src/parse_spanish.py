#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:34:11 2019

@author: prycebevan
"""

import string
import glob
import os
import re
import random
from collections import Counter

max_vocab_size = None
UNK = '[UNK]'
PAD = '[PAD]'

def vectorize_sequence(seq, vocab):
    seq = [tok if tok in vocab else UNK for tok in seq]
    return [vocab[tok] for tok in seq]


def unvectorize_sequence(seq, vocab):
    translate = sorted(vocab.keys(),key=lambda k:vocab[k])
    return [translate[i] for i in seq]


def one_hot_encode_label(label):
    vec = [1.0 if l==label else 0.0 for l in labels]
    return np.array(vec)

def retrieve_data(origs, mod):
    traincount = 0
    devcount = 0
    testcount = 0
    
    vocab = Counter()
    traindata = []
    trainlabels = []
    devdata = []
    devlabels = []
    testdata = []
    testlabels = []

    for index, file in enumerate(origs):
        
        #print(file)
        
        ofile = open(file, "r")
        mfile = open(mod[index], "r")
        
        o = ofile.read()
        m = mfile.read()
        
        obody = re.split(r"[\r\n]*\[BODY\]:\s*[\r\n]*", o)
        mbody = re.split(r"[\r\n]*\[BODY\]:\s*[\r\n]*", m)
        
        #I had a reason for this
        obodyNoPunct = obody[1].translate(str.maketrans('', '', string.punctuation))
        mbodyNoPunct = mbody[1].translate(str.maketrans('', '', string.punctuation))
        
        olines = obodyNoPunct.split(' ')
        mlines = mbodyNoPunct.split(' ')
        
        olinestmp = [x for x in olines if (x != '' and x != '\n')]
        mlinestmp = [x for x in mlines if (x != '' and x != '\n')]
        
        olines = []
        mlines = []
        
        for item in olinestmp:
            for word in item.splitlines():
                olines.append(word)
                
        for item in mlinestmp:
            for word in item.splitlines():
                mlines.append(word)
                
        if len(olines) == len(mlines):
            #print(olines)
            #print(mlines)
            rndint = random.randint(0,9)
            if rndint == 8:
                devcount += 1
                for idx, item in enumerate(olines):
                    devdata.append(item.lower())
                    devlabels.append(mlines[idx].lower())
                continue
            if rndint == 9:
                testcount += 1
                for idx, item in enumerate(olines):
                    testdata.append(item.lower())
                    testlabels.append(mlines[idx].lower())
                continue      
            traincount += 1
            for idx, item in enumerate(olines):
                traindata.append(item.lower())
                trainlabels.append(mlines[idx].lower())
                vocab[item.lower()] += 1


    print(traincount, devcount, testcount)
    return traindata, trainlabels, vocab, devdata, devlabels, testdata, testlabels

def retrieve_all_data():
    vocab = Counter()
    traindata = []
    trainlabels = []
    devdata = []
    devlabels = []
    testdata = []
    testlabels = []

    origs = glob.glob("../data/post_scriptum_spanish/original/16th_century/*.txt")
    mod = glob.glob("../data/post_scriptum_spanish/modernized/16th_century/*.txt")

    traindata1, trainlabels1, vocab1, devdata1, devlabels1, testdata1, testlabels1 = retrieve_data(origs, mod)

    origs = glob.glob("../data/post_scriptum_spanish/original/17th_century/*.txt")
    mod = glob.glob("../data/post_scriptum_spanish/modernized/17th_century/*.txt")

    traindata2, trainlabels2, vocab2, devdata2, devlabels2, testdata2, testlabels2 = retrieve_data(origs, mod)

    origs = glob.glob("../data/post_scriptum_spanish/original/18th_century/*.txt")
    mod = glob.glob("../data/post_scriptum_spanish/modernized/18th_century/*.txt")

    traindata3, trainlabels3, vocab3, devdata3, devlabels3, testdata3, testlabels3 = retrieve_data(origs, mod)

    origs = glob.glob("../data/post_scriptum_spanish/original/19th_century/*.txt")
    mod = glob.glob("../data/post_scriptum_spanish/modernized/19th_century/*.txt")

    traindata4, trainlabels4, vocab4, devdata4, devlabels4, testdata4, testlabels4 = retrieve_data(origs, mod)

#    exceldoc = open("myfile.csv", "w")
#    exceldoc.write("TRAIN, LABELS" + "\n")
#    for idx, item in enumerate(traindata1):
#        exceldoc.write(item + ", " + trainlabels1[idx] + "\n")
#    exceldoc.write("DEV, LABELS" + "\n")
#    for idx, item in enumerate(devdata1):
#        exceldoc.write(item + ", " + devlabels1[idx] + "\n")
#    exceldoc.write("TEST, LABELS" + "\n")
#    for idx, item in enumerate(testdata1):
#        exceldoc.write(item + ", " + testlabels1[idx] + "\n")
        
    traindata = traindata1 + traindata2 + traindata3 + traindata4
    trainlabels = trainlabels1 + trainlabels2 + trainlabels3 + trainlabels4
    devdata = devdata1 + devdata2 + devdata3 + devdata4
    devlabels = devlabels1 + devlabels2 + devlabels3 + devlabels4
    testdata = testdata1 + testdata2 + testdata3 + testdata4
    testlabels = testlabels1 + testlabels2 + testlabels3 + testlabels4
    vocab = vocab1 + vocab2 + vocab3 + vocab4
    
    if max_vocab_size:
        vocab = sorted(list([row[0] for row in vocab.most_common(max_vocab_size)]))
    else:
        vocab = sorted(vocab.keys(), key = lambda k: vocab[k], reverse=True)
    vocab = [UNK, PAD] + vocab    
    vocab = {k:v for v,k in enumerate(vocab)}
    return traindata, trainlabels, vocab, devdata, devlabels, testdata, testlabels

random.seed(17)
traindata, trainlabels, vocab, devdata, devlabels, testdata, testlabels = retrieve_all_data()