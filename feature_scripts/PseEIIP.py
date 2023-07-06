#三核苷酸的电子-离子相互作用伪势
#数据大小为————————————1148*64（正样本）
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:40:49 2020

@author: Administer
"""

import sys, os, re

from dateutil.parser import parser

from feature_scripts import check_parameters


def TriNcleotideComposition(sequence, base):
    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    tnc_dict = {}
    for triN in trincleotides:
        tnc_dict[triN] = 0
    for i in range(len(sequence) - 2):
        #print(sequence[i:i + 3])
        tnc_dict[sequence[i:i + 3]] += 1 #遍历列表获取每种三元核苷酸的数量
    for key in tnc_dict:
       tnc_dict[key] /= (len(sequence) - 2)  #计算对应三元核苷酸在DNA序列中所占的比例
    return tnc_dict

def PseEIIP(fastas, **kw):
    for i in fastas:
        if re.search('[^ACGT-]', i[1]):
            print('Error: illegal character included in the fasta sequences, only the "ACGT-" are allowed by this PseEIIP scheme.')
            return 0

    base = 'ACGT'

    EIIP_dict = {
        'A': 0.1260,
        'C': 0.1340,
        'G': 0.0806,
        'T': 0.1335,
    }

    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]  #AAA,AAC,AAG,AAT........
    EIIPxyz = {}  #字典
    for triN in trincleotides:
        EIIPxyz[triN] = EIIP_dict[triN[0]] + EIIP_dict[triN[1]] + EIIP_dict[triN[2]]  #当前三元核苷酸电子能量=每个核苷酸电子能量相加

    encodings = []
    header = ['#'] + trincleotides
    encodings.append(header)  #把AAA,AAC,AAG,AAT........加入进encodings列表

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        trincleotide_frequency = TriNcleotideComposition(sequence, base)  #计算每种三元核苷酸出现频率
        code = code + [EIIPxyz[triN] * trincleotide_frequency[triN] for triN in trincleotides]  #最终编码=第triN个核苷酸电子能量*第triN个核苷酸在这条序列中出现的频率
        encodings.append(code)
    return encodings

def PseEIIP_feature(fastas,kind):
    import sequence_read_save
    encodings = PseEIIP(fastas)
    sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/PseEIIP.csv")

def PseEIIP_feature_test(fastas,kind):
    import sequence_read_save
    encodings = PseEIIP(fastas)
    sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/test_PseEIIP.csv")

"""
def PseEIIP_feature(fastas,type):
    import sequence_read_save
    encodings = PseEIIP(fastas)
    sequence_read_save.save_to_csv(encodings, "./features/"+str(type)+"_PseEIIP.csv")
"""
if __name__ == '__main__':
    import os
    import sequence_read_save
    os.chdir('D:/DeepAc4C-main/')
    #os.chdir('C:/Users/Administer/Desktop/DeepAc4C-master/')  #os.chdir() 方法用于改变当前工作目录到指定的路径。
    #fastas = sequence_read_save.read_nucleotide_sequences("./sequence/input.fasta")
   #path_pos_data = "./DeepAc4C_Datasets/pos_training_samples_cdhit.fasta"
    #path_neg_data = "./DeepAc4C_Datasets/neg_training_samples_cdhit.fasta"

    fastas = sequence_read_save.read_nucleotide_sequences("./DeepAc4C_Datasets/pos_training_samples_cdhit.fasta")
    PseEIIP_feature(fastas)
    #PseEIIP_feature(fastas,type)














