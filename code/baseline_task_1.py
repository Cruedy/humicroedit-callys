# -*- coding: utf-8 -*-
"""
@author: Nabil Hossain
         nhossain@cs.rochester.edu
         Dept. Computer Science
         University of Rochester, NY
"""

'''
A naive baseline system for task 1

This baseline always predicts the mean funniness grade of all edited headlines in
training set.
'''

import pandas as pd
import numpy as np
import re
import sys
import os

# Taken from https://github.com/rajaswa/bert-humor-semeval-2020/blob/master/task1.py
def get_sentence_pair(sent_orig, edit_word):
    sent_o = re.sub("[</>]", "", sent_orig)
    sent_e = (sent_orig.split("<"))[0] + edit_word + (sent_orig.split(">"))[1]

    return sent_e, sent_o

def baseline_task_1(train_loc, test_loc):
    train = pd.read_csv(train_loc)    
    test = pd.read_csv(test_loc)

    sentences = train.original
    edits = train.edit
    for i in range(len(train)):
        print(get_sentence_pair(sentences[i], edits[i]))
    # can use these two sentences + meanGrade? to train on BERT?

    pred = np.mean(train.meanGrade)    
    test['pred'] = pred
    
    output = test[['id','pred']] 
    out_loc = '../output/task-1-output.csv'
    output.to_csv(out_loc, index=False)
    
    print('Output file created:\n\t- '+os.path.abspath(out_loc))
    
if __name__ == '__main__':
    
    # expect sys.argv[1] = ../data/task-1/train.csv
    # expect sys.argv[2] = ../data/task-1/dev.csv
    baseline_task_1(sys.argv[1], sys.argv[2])

        