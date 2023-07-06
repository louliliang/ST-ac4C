# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:40:49 2020

@author: Administer
"""
import pickle

import numpy as np
from sklearn.model_selection import train_test_split


def read_nucleotide_sequences(file):
    import re, os, sys
    if os.path.exists(file) == False:
        print(os.path)
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None: #:匹配整个字符串,并返回第一个成功的匹配
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:] #获取并分割数据
    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        #获取rna（ACGTU-）序列的序号和把ACGTU-序列用-代替
        header, sequence = array[0].split()[0], re.sub('[^ACGTU-]', '-', ''.join(array[1:]).upper())  #匹配^ACGTU-用-代替
        sequence = re.sub('U', 'T', sequence)  #  replace U as T
        fasta_sequences.append([header, sequence])

    return fasta_sequences

"""
# 读入数据后写入序列列表token_list,返回序列列表和它的最大长度
def transform_token2index(sequences):
    token2index = pickle.load(open('./DeepAc4C_Datasets/residue2idx.pkl', 'rb'))
    print(token2index)

    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)

    token_list = list()
    max_len = 0
    for seq in sequences:
        seq_id = [token2index[residue] for residue in seq]
        token_list.append(seq_id)
        if len(seq) > max_len:
            max_len = len(seq)

    print('-' * 20, '[transform_token2index]: check sequences_residue and token_list head', '-' * 20)
    print('sequences_residue', sequences[0:5])
    print('token_list', token_list[0:5])
    return token_list, max_len
"""

# 对序列列表进行标注格式为[CLS]+序列+[SEP]
"""def make_data_with_unified_length(token_list, max_len):
    token2index = pickle.load(open('./data/residue2idx.pkl', 'rb'))
    data = []
    for i in range(len(token_list)):
        token_list[i] = [token2index['[CLS]']] + token_list[i] + [token2index['[SEP]']]  # 前
        n_pad = max_len - len(token_list[i])
        token_list[i].extend([0] * n_pad)
        data.append(token_list[i])

    print('-' * 20, '[make_data_with_unified_length]: check token_list head', '-' * 20)
    print('max_len + 2', max_len)
    print('token_list + [pad]', token_list[0:5])

    return data

"""


def chunkIt(seq, num):  # 把处理的数据按照10等份分类，添加到out列表中去
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])  # append()方法表示在原来的列表末尾追加新的对象
        last += avg

    return out

#保存数组
def save_to_csv(encodings, file):
    with open(file, 'w') as f:
        for line in encodings[1:]:
            f.write(str(line[0]))
            for i in range(1,len(line)):
                f.write(',%s' % line[i])
            f.write('\n')

def file_remove():
    import os
    dir_list1=os.listdir("./features/") #“os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    for x in dir_list1:
        if x.split(".")[-1]=="csv":
            os.remove("./features/"+str(x))
    for i in range(1,11,1):
        dir_list2=os.listdir("./features/mm/"+str(i))
        for x in dir_list2:
            if x.split(".")[-1]=="csv":
                os.remove("./features/mm/"+str(i)+"/"+str(x)) 
        dir_list3=os.listdir("./features/mm/"+str(i)+"/f_b/")
        for x in dir_list3:
            if x.split(".")[-1]=="csv":
                os.remove("./features/mm/"+str(i)+"/f_b/"+str(x))
                
    dir_list4=os.listdir("./features/combined_features/")
    for x in dir_list4:
        if x.split(".")[-1]=="csv":
            os.remove("./features/combined_features/"+str(x))
        
    # dir_list5=os.listdir("./results/")
    # for x in dir_list5:
        # os.remove("./results/"+str(x))