#!/usr/bin/env python
# _*_coding:utf-8_*_

import sys, os, platform
from feature_scripts import sequence_read_save
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
father_path = os.path.abspath(
    os.path.dirname(pPath) + os.path.sep + ".") + r'\pubscripts' if platform.system() == 'Windows' else os.path.abspath(
    os.path.dirname(pPath) + os.path.sep + ".") + r'/pubscripts'
sys.path.append(father_path)



def ANF(fastas, **kw):

    AA = 'ACGT'
    encodings = []
    header = ['#', 'label']
    for i in range(1, len(fastas[0][1]) + 1):
        header.append('ANF.' + str(i))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for j in range(len(sequence)):
            code.append(sequence[0: j + 1].count(sequence[j]) / (j + 1))
        encodings.append(code)
    return encodings

def ANF_features(fastas,kind):
    encodings = ANF(fastas)
    sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/ANF.csv")

def ANF_features_test(fastas,kind):
    encodings = ANF(fastas)
    sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/test_ANF.csv")


if __name__ == '__main__':
    import os
    import sequence_read_save
    os.chdir('G:/AC4C/')
    #os.chdir('C:/Users/Administer/Desktop/DeepAc4C-master/')  #os.chdir() 方法用于改变当前工作目录到指定的路径。
    #fastas = sequence_read_save.read_nucleotide_sequences("./sequence/input.fasta")
   #path_pos_data = "./DeepAc4C_Datasets/pos_training_samples_cdhit.fasta"
    #path_neg_data = "./DeepAc4C_Datasets/neg_training_samples_cdhit.fasta"

    fastas = sequence_read_save.read_nucleotide_sequences("./Datasets/pos_training_samples_cdhit.fasta")
    encodings=ANF(fastas)
    sequence_read_save.save_to_csv(encodings, "./features/ANF.csv")