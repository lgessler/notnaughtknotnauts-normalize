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
from collections import Counter

cwd = os.getcwd()

vocab = Counter()
data = []
labels = []

max_vocab_size = None

UNK = '[UNK]'
PAD = '[PAD]'

origs = glob.glob(cwd + "/original/16th_century/*.txt")
mod = glob.glob(cwd + "/modernized/16th_century/*.txt")

count = 0

for index, file in enumerate(origs):
    
    ofile = open(file, "r")
    mfile = open(mod[index], "r")
    
    o = ofile.read()
    m = mfile.read()
    
    obody = re.split(r"[\r\n]*\[BODY\]:\s*[\r\n]*", o)
    mbody = re.split(r"[\r\n]*\[BODY\]:\s*[\r\n]*", m)
    
#    print(o)
#    print(obody)
#    print(m)
#    print(mbody)
    
    
    #I had a reason for this
    obodyNoPunct = obody[1].translate(str.maketrans('', '', string.punctuation))
    mbodyNoPunct = mbody[1].translate(str.maketrans('', '', string.punctuation))
    
    olines = obodyNoPunct.split(' ')
    mlines = mbodyNoPunct.split(' ')
    
    olines = [x for x in olines if (x != '' and x != '\n')]
    mlines = [x for x in mlines if (x != '' and x != '\n')]
    
    #if index == 295:
    #    print(file)
    #    for num, word in enumerate(olines):
    #        print(word, mlines[num])
        #print(olines)
        #print(mlines)
        
    if len(olines) == len(mlines):
        count += 1
        for idx, item in enumerate(olines):
            data.append(item.lower())
            labels.append(mlines[idx].lower())
            vocab[item.lower()] += 1
    #print(index)
    #print(len(olines))
    #print(len(mlines))
#print (len(origs), count)
if max_vocab_size:
    vocab = sorted(list([row[0] for row in vocab.most_common(max_vocab_size)]))
else:
    vocab = sorted(vocab.keys(), key = lambda k: vocab[k], reverse=True)
vocab = [UNK, PAD] + vocab    
vocab = {k:v for v,k in enumerate(vocab)}





def vectorize_sequence(seq, vocab):
    seq = [tok if tok in vocab else UNK for tok in seq]
    return [vocab[tok] for tok in seq]


def unvectorize_sequence(seq, vocab):
    translate = sorted(vocab.keys(),key=lambda k:vocab[k])
    return [translate[i] for i in seq]


def one_hot_encode_label(label):
    vec = [1.0 if l==label else 0.0 for l in labels]
    return np.array(vec)