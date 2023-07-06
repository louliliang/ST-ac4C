#!/usr/bin/env python
#_*_coding:utf-8_*_
#CKSNAP编码方案计算由任意核酸分隔的核酸对的出现频率
#数据大小为————————————1148*48
def get_min_sequence_length(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen

def CKSNAP(fastas, gap=5, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if get_min_sequence_length(fastas) < gap + 2:  #序列长度小于gap+2时，报错
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    AA = kw['order'] if kw['order'] != None else 'ACGT'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)   #aaPairs=['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']

    header = ['#', ]
    for g in range(gap + 1):  #range：创建一个整数列表
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1   #myDict存储分隔的核酸对出现次数{'AA': 8, 'AC': 27, 'AG': 33, 'AT': 8, 'CA': 27, 'CC': 43, 'CG': 22, 'CT': 31, 'GA': 35, 'GC': 33, 'GG': 71, 'GT': 14, 'TA': 5, 'TC': 20, 'TG': 28, 'TT': 9}
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings

def cksnap_feature(fastas,kind):
    import sequence_read_save
    kw = {'order': 'ACGT'}
    encodings = CKSNAP(fastas, gap=2, **kw)  #gag表示间隙：CKSNAP描述符的k-space值，应为大于0的整数，默认值为5
    sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/cksnap.csv")

def cksnap_feature_test(fastas,kind):
    import sequence_read_save
    kw = {'order': 'ACGT'}
    encodings = CKSNAP(fastas, gap=2, **kw)  #gag表示间隙：CKSNAP描述符的k-space值，应为大于0的整数，默认值为5
    sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/test_cksnap.csv")
"""
def cksnap_feature(fastas,type):
    import sequence_read_save
    kw = {'order': 'ACGT'}
    encodings = CKSNAP(fastas, gap=2, **kw)  #gag表示间隙：CKSNAP描述符的k-space值，应为大于0的整数，默认值为5
    sequence_read_save.save_to_csv(encodings, "./features/"+str(type)+"_cksnap.csv")
"""
if __name__ == '__main__':
    import os
    import sequence_read_save

    os.chdir('D:/DeepAc4C-main/')
    #os.chdir('C:/Users/Administer/Desktop/DeepAc4C-master/')
    #fastas =sequence_read_save.read_nucleotide_sequences("./sequence/input.fasta")
    #path_pos_data = "./DeepAc4C_Datasets/pos_training_samples_cdhit.fasta"
    #path_neg_data = "./DeepAc4C_Datasets/neg_training_samples_cdhit.fasta"
    #fastas = sequence_read_save.read_nucleotide_sequences(path_pos_data, path_neg_data)
    fastas = sequence_read_save.read_nucleotide_sequences("./DeepAc4C_Datasets/pos_training_samples_cdhit.fasta")
    cksnap_feature(fastas)