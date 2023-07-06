#!/usr/bin/env python
#_*_coding:utf-8_*_
#数据大小为————————————1148*84（正样本）
from feature_scripts import sequence_read_save


def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer
#Kmer(fastas, 3, "DNA", True, False, )
#normalize；规范化：使用此参数值，最终特征向量将基于所有kmer的总出现次数进行规范化
#upto:使用此参数，程序将生成所有kmer：1mer，2mer，…，kmer；
def Kmer(fastas, k=2, type="DNA", upto=False, normalize=True):
    import re
    import itertools
    from collections import Counter
    encoding = []
    header = ['#', 'label']
    NA = 'ACGT'
    if type in ("DNA", 'RNA'):
        NA = 'ACGT'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'

    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0

    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):    #itertools.product()求笛卡尔积。
                header.append(''.join(kmer))
        encoding.append(header)
        for i in fastas:
            name, sequence, = i[0], re.sub('-', '', i[1])  #把编号和序列遍历出来
            count = Counter()   #可以用来统计一个 python 列表、字符串、元组等可迭代对象中每个元素出现的次数(统计 A,T,G,C出现的次数）
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)

                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)  #获得k个核苷酸出现的频率
            code = [name,]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    else:
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))
        encoding.append(header)
        for i in fastas:
            name, sequence = i[0], re.sub('-', '', i[1])
            kmers = kmerArray(sequence, k)
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = [name,]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    return encoding

def kmer_features(fastas,kind):
    encodings = Kmer(fastas, 3, "DNA", True, False)
    sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/kmer.csv")

def kmer_features_test(fastas,kind):
    encodings = Kmer(fastas, 3, "DNA", True, False)
    sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/test_kmer.csv")
"""
def kmer_features(fastas,type):
    import sequence_read_save
    encodings = Kmer(fastas, 3, "DNA", True, False)
    sequence_read_save.save_to_csv(encodings, "./features/"+str(type)+"_kmer.csv")

"""
if __name__ == '__main__':
    import os

    os.chdir('D:/DeepAc4C-main/')
    #os.chdir('C:/Users/Administer/Desktop/DeepAc4C-master/')
    #fastas = sequence_read_save.read_nucleotide_sequences("./sequence/input.fasta")
    #path_pos_data="./DeepAc4C_Datasets/pos_training_samples_cdhit.fasta"
    #path_neg_data="./DeepAc4C_Datasets/neg_training_samples_cdhit.fasta"
    #fastas = sequence_read_save.read_nucleotide_sequences(path_pos_data, path_neg_data)
    fastas = sequence_read_save.read_nucleotide_sequences(
        "./DeepAc4C_Datasets/pos_training_samples_cdhit.fasta")
 