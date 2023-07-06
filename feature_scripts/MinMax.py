#!/usr/bin/env python
# _*_coding:utf-8_*_
import re

import numpy as np

normal=[]
def MinMax(encodings, labels,fea,kind):
    #normalized_vector = np.zeros((len(encodings[0])), len(encodings)).astype(str)
    normalized_vector = np.zeros((len(encodings), len(encodings[0]))).astype(str)
    normalized_vector[0, 1:] = encodings[0][1:]
    normalized_vector[:, 0] = ['label'] + labels
    data = np.array(encodings[1:, 1:]).astype(float)
    e = ''
    for i in range(len(data[0])):
        maxValue, minValue = max(data[:, i]), min(data[:, i])
        try:
             data[:, i] = (data[:, i] - minValue) / maxValue
        except ZeroDivisionError as e:
            return 0, e
    normalized_vector[1:, 1:] = data.astype(str)
    normalized_vector=np.array(normalized_vector)
    # for i in range(1, 11, 1):
    #     np.savetxt("features/" + str(kind) +"/mm/" + str(i) + "/f_b/"+str(fea), normalized_vector, fmt='%s', delimiter=',')
    np.savetxt("features/" + str(kind) + "/mm/f_b/" + str(fea), normalized_vector, fmt='%s',delimiter=',')
    return np.array(normalized_vector), e #.tolist()

def read_csv(file):
    encodings = []
    labels = []
    with open(file) as f:
        records = f.readlines()

    ##
    feature = 1
    header = ['#']
    for i in range(1, len(records[0].split(','))):
        header.append('%f' % feature)
        feature = feature + 1
    encodings.append(header)
    ##
    sample = 1
    for line in records:
        array = line.strip().split(',') if line.strip() != '' else None
        encodings.append(['s.%d' % sample] + array[1:])
        labels.append(str(array[0]))
        sample = sample + 1
    return np.array(encodings), labels

def feature_merge(feature_list,kind):
    import numpy as np
    import pandas as pd
    n=0
    for fea in feature_list:
        file='./features/'+str(kind)+'/'+str(fea)
        encodings, labels =read_csv(file)
        MinMax(encodings, labels,fea,kind)


def MinMax_normalized(kind):
    feature_list = ['kmer.csv', 'PseEIIP.csv', 'PseKNC.csv']
    # feature_list = ['kmer.csv', 'PseEIIP.csv', 'PseKNC.csv','w2v_feature.csv']
    # feature_list = ['SCPseDNC.csv',  'PseKNC.csv','SCPseTNC.csv']
    feature_merge(feature_list,kind)
def MinMax_normalized_test(kind):
    feature_list = ['test_kmer.csv', 'test_PseEIIP.csv', 'test_PseKNC.csv']
    #feature_list = ['test_kmer.csv', 'test_PseEIIP.csv', 'test_PseKNC.csv','test_w2v_feature.csv']
    feature_merge(feature_list,kind)


def MinMax_normalized_kmer(kind):
    feature_list = ['kmer.csv']
    feature_merge(feature_list, kind)

def MinMax_normalized_ANF(kind):
    feature_list = ['ANF.csv']
    feature_merge(feature_list, kind)


def MinMax_normalized_PseEIIP(kind):
    feature_list = ['PseEIIP.csv']
    feature_merge(feature_list, kind)

def MinMax_normalized_CKSNAP(kind):
    feature_list = ['CKSNAP.csv']
    feature_merge(feature_list, kind)

def MinMax_normalized_PseKNC(kind):
    feature_list = ['PseKNC.csv']
    feature_merge(feature_list, kind)


def MinMax_normalized_PseEIIP_kmer(kind):
    feature_list = ['kmer.csv', 'PseEIIP.csv', 'PseKNC.csv']
    feature_merge(feature_list, kind)

def MinMax_normalized_PseKNC_PseEIIP(kind):
    feature_list = ['PseEIIP.csv', 'PseKNC.csv']
    feature_merge(feature_list, kind)



def MinMax_normalized_kmer_PseKNC_PSEEIIP(kind):
    feature_list = ['kmer.csv', 'PseKNC.csv','PseEIIP.csv']
    feature_merge(feature_list, kind)


# def MinMax_normalized_kmer_ANF(kind):
#     feature_list = ['kmer.csv', 'ANF.csv']
#     feature_merge(feature_list,kind)
#
# def MinMax_normalized_PseEIIP_ANF(kind):
#     feature_list = ['PseEIIP.csv', 'ANF.csv']
#     feature_merge(feature_list,kind)
# def MinMax_normalized_kmer_pse(kind):
#     feature_list=['kmer.csv', 'PseEIIP.csv']
#     feature_merge(feature_list, kind)
#
#
# def MinMax_normalized_kmer(kind):
#     feature_list = ['kmer.csv']
#     feature_merge(feature_list, kind)
#
# def MinMax_normalized_ANF(kind):
#     feature_list = ['ANF.csv']
#     feature_merge(feature_list, kind)
#
# def MinMax_normalized_PseEIIP(kind):
#     feature_list = ['PseEIIP.csv']
#     feature_merge(feature_list, kind)
#
#
# def MinMax_normalized_CKSNAP(kind):
#     feature_list=['CKSNAP.csv']
#     feature_merge(feature_list, kind)
#
# def MinMax_normalized_PseKNC(kind):
#     feature_list = ['PseKNC.csv']
#     feature_merge(feature_list, kind)

if __name__ == '__main__':
    import os
    import sequence_read_save
    os.chdir('G:/XG‑ac4C/')
    feature_list = ['kmer.csv', 'PseEIIP.csv', 'ANF.csv']  #,
    kind='test'
    feature_merge(feature_list, kind)
    #sequence_read_save.save_to_csv(encodings, "./features/NCP_ND.csv")