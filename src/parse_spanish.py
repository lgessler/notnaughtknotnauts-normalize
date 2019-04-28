#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:34:11 2019

@author: prycebevan
"""

import string
import glob
import re
import random
import numpy as np
import pandas as pd
from collections import Counter

CENTURIES = ["16th_century", "17th_century", "18th_century", "19th_century"]

def get_token_list(file_path):
    """Gets a list of tokens (words) from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        file_text = f.read()
    pattern = r"[\r\n]*\[BODY\]:\s*[\r\n]*"
    body = re.split(pattern, file_text)[1]
    body = body.replace("\n", "")
    body = body.lower()
    translation_map = str.maketrans("", "", string.punctuation)
    body = body.translate(translation_map)
    tokens = body.split(" ")
    tokens = [token for token in tokens if token != ""]
    return tokens

def retrieve_century(century):
    if century not in CENTURIES:
        raise ValueError("Invalid century :" + century
                         + "\nValid centuries are: " + ", ".join(CENTURIES))

    # Dataset contains input (a single original token) in the first column and the label (a single normalized token)
    all_orig_toks = []
    all_norm_toks = []

    print(f"Processing {century} documents...")
    orig_files = glob.glob(f"../data/post_scriptum_spanish/original/{century}/*.txt")
    norm_files = glob.glob(f"../data/post_scriptum_spanish/modernized/{century}/*.txt")
    for index, file in enumerate(orig_files):
        orig_toks = get_token_list(file)
        norm_toks = get_token_list(norm_files[index])
        if len(orig_toks) == len(norm_toks):
            all_orig_toks += orig_toks
            all_norm_toks += norm_toks

    return all_orig_toks, all_norm_toks


def retrieve_tokens(century="all"):
    if century != "all":
        return retrieve_century(century)
    else:
        all_orig_toks = []
        all_norm_toks = []
        for c in CENTURIES:
            orig_toks, norm_toks = retrieve_century(c)
            all_orig_toks += orig_toks
            all_norm_toks += norm_toks
        return all_orig_toks, all_norm_toks

if __name__ == "__main__":
    filepath = '../data/'
    basename = 'all_centuries_toks'
    otoks, ntoks = retrieve_tokens()
    print("Found " + str(len(otoks)) + " tokens.")

    tok_pairs = list(zip(otoks, ntoks))

    dev = len(otoks) // 10
    test = len(otoks) // 10 * 2
    with open(filepath + basename + '.dev.tsv', 'w') as f:
        for orig, norm in tok_pairs[:dev]:
            f.write(f"{orig}\t{norm}\n")
    with open(filepath + basename + '.test.tsv', 'w') as f:
        for orig, norm in tok_pairs[dev:test]:
            f.write(f"{orig}\t{norm}\n")
    with open(filepath + basename + '.train.tsv', 'w') as f:
        for orig, norm in tok_pairs[test:]:
            f.write(f"{orig}\t{norm}\n")


