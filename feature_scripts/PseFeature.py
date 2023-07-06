import re
import itertools

myDiIndex = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
    'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15
}
myTriIndex = {
    'AAA': 0, 'AAC': 1, 'AAG': 2, 'AAT': 3,
    'ACA': 4, 'ACC': 5, 'ACG': 6, 'ACT': 7,
    'AGA': 8, 'AGC': 9, 'AGG': 10, 'AGT': 11,
    'ATA': 12, 'ATC': 13, 'ATG': 14, 'ATT': 15,
    'CAA': 16, 'CAC': 17, 'CAG': 18, 'CAT': 19,
    'CCA': 20, 'CCC': 21, 'CCG': 22, 'CCT': 23,
    'CGA': 24, 'CGC': 25, 'CGG': 26, 'CGT': 27,
    'CTA': 28, 'CTC': 29, 'CTG': 30, 'CTT': 31,
    'GAA': 32, 'GAC': 33, 'GAG': 34, 'GAT': 35,
    'GCA': 36, 'GCC': 37, 'GCG': 38, 'GCT': 39,
    'GGA': 40, 'GGC': 41, 'GGG': 42, 'GGT': 43,
    'GTA': 44, 'GTC': 45, 'GTG': 46, 'GTT': 47,
    'TAA': 48, 'TAC': 49, 'TAG': 50, 'TAT': 51,
    'TCA': 52, 'TCC': 53, 'TCG': 54, 'TCT': 55,
    'TGA': 56, 'TGC': 57, 'TGG': 58, 'TGT': 59,
    'TTA': 60, 'TTC': 61, 'TTG': 62, 'TTT': 63
}

baseSymbol = 'ACGT'

#获取序列出现的比例
def get_kmer_frequency(sequence, kmer):
    myFrequency = {}
    for pep in [''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))]:
        myFrequency[pep] = 0
    for i in range(len(sequence) - kmer + 1):
        myFrequency[sequence[i: i + kmer]] = myFrequency[sequence[i: i + kmer]] + 1  #把序列号是sequence[i: i + kmer]的序列加一，统计AAA——ATT 16种核苷酸出现的次数
    for key in myFrequency:
        myFrequency[key] = myFrequency[key] / (len(sequence) - kmer + 1) #统计AAA——ATT 16种核苷酸出现的频率
    return myFrequency


def correlationFunction(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        CC = CC + (float(myPropertyValue[p][myIndex[pepA]]) - float(myPropertyValue[p][myIndex[pepB]])) ** 2
    return CC / len(myPropertyName)


def correlationFunction_type2(pepA, pepB, myIndex, myPropertyName, myPropertyValue):
    CC = 0
    for p in myPropertyName:
        #print("myPropertyValue[p][myIndex[pepA]]",myPropertyValue[p][myIndex[pepA]])

        CC = CC + float(myPropertyValue[p][myIndex[pepA]]) * float(myPropertyValue[p][myIndex[pepB]])
        #当编码为Rise时候，Rise编码=['-0.109', '1.044', '-0.623', '1.171', '-1.254', '0.242', '-1.389', '-0.623', '0.711', '1.585', '0.242', '1.044', '-1.389', '0.711', '-1.254', '-0.109']
        #第一步的结果=myPropertyValue[Rise][3]*myPropertyValue[Rise][14]=1.171*-1.254=-1.468434
        #第二步的结果myPropertyValue[Rise][14]*myPropertyValue[Rise][8]=-1.254*0.711=-0.891594
    return CC

   #weight=0.1,kmer=2,lamdavalue=10,sequence为序列
def get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):  #
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        theta = 0
        for i in range(len(sequence) - tmpLamada - kmer):
            #print("sequence[i:i + kmer]",sequence[i:i + kmer])
            #print("sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer]",sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer])
            theta = theta + correlationFunction(sequence[i:i + kmer],
                                                sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer], myIndex,
                                                myPropertyName, myPropertyValue) #每次i变换，计算对应序列的6中物理化学编码
        thetaArray.append(theta / (len(sequence) - tmpLamada - kmer)) #对编码进行归一化处理
    return thetaArray


def get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, kmer):
    thetaArray = []
    for tmpLamada in range(lamadaValue):
        for p in myPropertyName:
            theta = 0
            for i in range(len(sequence) - tmpLamada - kmer):
                theta = theta + correlationFunction_type2(sequence[i:i + kmer],
                                                          sequence[i + tmpLamada + 1: i + tmpLamada + 1 + kmer],
                                                          myIndex,
                                                          [p], myPropertyValue)
            thetaArray.append(theta / (len(sequence) - tmpLamada - kmer))
    return thetaArray





def make_PseKNC_vector(fastas, myPropertyName, myPropertyValue, lamadaValue, weight, kmer):  #weight=0.1,kmer=3,lamdavalue=10
    encodings = []
    myIndex = myDiIndex
    header = ['#']
    header = header + sorted([''.join(i) for i in list(itertools.product(baseSymbol, repeat=kmer))])  #join()连接字符串数组。将字符串、元组、列表中的元素以指定的字符(分隔符)连接生成一个新的字符串,
    # itertools.product()根据输入的可遍历对象生成笛卡尔积，与嵌套的for循环类似
    #header=['#', 'AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT', 'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT']
    for k in range(1, lamadaValue + 1):
        header.append('lamada_' + str(k))
    encodings.append(header)
    #encodings=['#', 'AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT', 'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT', 'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT', 'lamada_1', 'lamada_2', 'lamada_3', 'lamada_4', 'lamada_5', 'lamada_6', 'lamada_7', 'lamada_8', 'lamada_9', 'lamada_10']]
    for i in fastas:
        name, sequence, = i[0], re.sub('-', '', i[1])
        code = [name]
        kmerFreauency = get_kmer_frequency(sequence, kmer)  #统计AAA——TTT 64种核苷酸出现的频率

        thetaArray = get_theta_array(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)  #10次计算对应序列的物理化学编码
       #[1.7048674975845395, 1.4586034963503642, 1.3797778321078418, 1.4013733876543202, 1.3338103432835808, 1.4004258395989961, 1.1809423017676761, 1.5297967849872758, 1.3022060089743583, 1.4294466356589144]

       #以下分两种情况计算k元组核苷酸组成
        #1. 1<=u<=4时
        for pep in sorted([''.join(j) for j in list(itertools.product(baseSymbol, repeat=kmer))]):  #sorted排序函数
            code.append(kmerFreauency[pep] / (1 + weight * sum(thetaArray)))  #计算AAA——TTT 64种核苷酸带权频率(64维度）
        # 2. 4^k<=u<=4^k +lamadaValue时:
        #* 表示乘积运算。**表示乘方运算。
        for k in range(len(baseSymbol) ** kmer + 1, len(baseSymbol) ** kmer + lamadaValue + 1):
            code.append((weight * thetaArray[k - (len(baseSymbol) ** kmer + 1)]) / (1 + weight * sum(thetaArray))) #(10维度）
        encodings.append(code)
    return encodings



# def make_SCPseDNC_vector(fastas, myPropertyName, myPropertyValue, lamadaValue, weight):
#     encodings = []
#     myIndex = myDiIndex
#     header = ['#']
#     for pair in sorted(myIndex): #对 'AA'——‘TT'进行排序
#         header.append(pair)
#     for k in range(1, lamadaValue * len(myPropertyName) + 1):
#         header.append('lamada_' + str(k))
# 
#     encodings.append(header)
#     for i in fastas:
#         name, sequence = i[0], re.sub('-', '', i[1])
#         code = [name]
#         dipeptideFrequency = get_kmer_frequency(sequence, 2)  #计算AA-TT核苷酸的频率
#         thetaArray = get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 2)
#         for pair in sorted(myIndex.keys()):
#             code.append(dipeptideFrequency[pair] / (1 + weight * sum(thetaArray)))
#         for k in range(17, 16 + lamadaValue * len(myPropertyName) + 1):
#             code.append((weight * thetaArray[k - 17]) / (1 + weight * sum(thetaArray)))
#         encodings.append(code)
#     return encodings


def make_SCPseTNC_vector(fastas, myPropertyName, myPropertyValue, lamadaValue, weight):
    encodings = []
    myIndex = myTriIndex
    header = ['#']
    for pep in sorted(myIndex):
        header.append(pep)
    for k in range(1, lamadaValue * len(myPropertyName) + 1):
        header.append('lamada_' + str(k))
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        tripeptideFrequency = get_kmer_frequency(sequence, 3)
        thetaArray = get_theta_array_type2(myIndex, myPropertyName, myPropertyValue, lamadaValue, sequence, 3)
        for pep in sorted(myIndex.keys()):
            code.append(tripeptideFrequency[pep] / (1 + weight * sum(thetaArray)))
        for k in range(65, 64 + lamadaValue * len(myPropertyName) + 1):
            code.append((weight * thetaArray[k - 65]) / (1 + weight * sum(thetaArray)))
        encodings.append(code)
    return encodings

import sys, os, platform

import pickle

myDictDefault = {
    'PseKNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
               'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'SCPseDNC': {'DNA': ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist'],
                 'RNA': ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']},
    'SCPseTNC': {'DNA': ['Dnase I', 'Bendability (DNAse)'], 'RNA': []},
}

myDataFile = {
    'PseKNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'SCPseDNC': {'DNA': 'didnaPhyche.data', 'RNA': 'dirnaPhyche.data'},
    'SCPseTNC': {'DNA': 'tridnaPhyche.data', 'RNA': ''},
}

#check_Pse_arguments(fastas,"PseKNC","DNA",0.1,3,10)
def check_Pse_arguments(fastas,method,nctype,weight,kmer,lamadaValue):
    #os.chdir('C:/Users/Administer/Desktop/DeepAc4C-master/')
    #if not os.path.exists("input.fasta"):
     #   print('Error: the input file does not exist.')
    #    sys.exit(1)
    if not 0 < weight < 1:
        print('Error: the weight factor ranged from 0 ~ 1.')
        sys.exit(1)
    if not 0 < kmer < 10:
        print('Error: the kmer value ranged from 1 - 10')
        sys.exit(1)

    fastaMinLength = 100000000
    for i in fastas:
        if len(i[1]) < fastaMinLength:
            fastaMinLength = len(i[1])
    if not 0 <= lamadaValue <= (fastaMinLength - 2): #lamadaValue小于序列长度减二
        print('Error: lamada value error, please see the manual for details.')
        sys.exit(1)


    myIndex = myDictDefault[method][nctype]  #六种物理化学性质
    dataFile = myDataFile[method][nctype]  #六种物理化学性质的值
    if dataFile != '':
        with open('./feature_scripts/' + dataFile, 'rb') as f:
            myProperty = pickle.load(f)

    if len(myIndex) == 0 or len(myProperty) == 0:
        print('Error: arguments is incorrect.')
        sys.exit(1)

    return myIndex, myProperty, lamadaValue, weight, kmer
"""
def Pse_feature(fastas,type):
    import sequence_read_save
    import check_parameters
    
    # PseKNC（74维度）
    #（PseKNC基于三核苷酸出现频率和6个物理化学指标（上升、滚动、移位、滑动、倾斜和扭曲）来表征一个序列的连续局部序列顺序信息和全局序列顺序信息）
    my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(fastas,"PseKNC","DNA",0.1,3,10)
    encodings = make_PseKNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight, kmer) 
    sequence_read_save.save_to_csv(encodings,"./features/"+str(type)+"_PseKNC.csv")
    
    # SCPseDNC   包含两个不同的连续序列-顺序信息 （46维度）
    my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(fastas,"SCPseDNC","DNA",0.1,2,5)
    # print('SCPSEDNC')
    # print(lamada_value, weight, kmer)
    encodings = make_SCPseDNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight) 
    sequence_read_save.save_to_csv(encodings,"./features/"+str(type)+"_SCPseDNC.csv")
    
    # SCPseTNC  （72维度）
    my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(fastas,"SCPseTNC","DNA",0.1,3,4)
    encodings = make_SCPseTNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight) 
    sequence_read_save.save_to_csv(encodings,"./features/"+str(type)+"_SCPseTNC.csv")
"""


def Pse_feature(fastas,kind):
    import sequence_read_save
    import check_parameters

    # PseKNC（74维度）
    # （PseKNC基于三核苷酸出现频率和6个物理化学指标（上升、滚动、移位、滑动、倾斜和扭曲）来表征一个序列的连续局部序列顺序信息和全局序列顺序信息）
    my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(fastas, "PseKNC", "DNA", 0.1,
                                                                                          3, 10)
    encodings = make_PseKNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight, kmer)
    sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/PseKNC.csv")

    # # SCPseDNC   包含两个不同的连续序列-顺序信息 （46维度）
    # my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(fastas, "SCPseDNC", "DNA",
    #                                                                                       0.1, 2, 5)
    # # print('SCPSEDNC')
    # # print(lamada_value, weight, kmer)
    # encodings = make_SCPseDNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight)
    # sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/SCPseDNC.csv")
    # 
    # # SCPseTNC  （72维度）
    # my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(fastas, "SCPseTNC", "DNA",
    #                                                                                       0.1, 3, 4)
    # encodings = make_SCPseTNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight)
    # sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/SCPseTNC.csv")

def Pse_feature_test(fastas,kind):
    import sequence_read_save
    import check_parameters

    # PseKNC（74维度）
    # （PseKNC基于三核苷酸出现频率和6个物理化学指标（上升、滚动、移位、滑动、倾斜和扭曲）来表征一个序列的连续局部序列顺序信息和全局序列顺序信息）
    my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(fastas, "PseKNC", "DNA", 0.1,
                                                                                          3, 10)
    encodings = make_PseKNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight, kmer)
    sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/test_PseKNC.csv")

    # # SCPseDNC   包含两个不同的连续序列-顺序信息 （46维度）
    # my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(fastas, "SCPseDNC", "DNA",
    #                                                                                       0.1, 2, 5)
    # # print('SCPSEDNC')
    # # print(lamada_value, weight, kmer)
    # encodings = make_SCPseDNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight)
    # sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/test_SCPseDNC.csv")
    #
    # # SCPseTNC  （72维度）
    # my_property_name, my_property_value, lamada_value, weight, kmer = check_Pse_arguments(fastas, "SCPseTNC", "DNA",
    #                                                                                       0.1, 3, 4)
    # encodings = make_SCPseTNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight)
    # sequence_read_save.save_to_csv(encodings, "./features/"+str(kind)+"/test_SCPseTNC.csv")

if __name__ == '__main__':
    import os

    os.chdir('D:/DeepAc4C-main/')
    #os.chdir('C:/Users/Administer/Desktop/DeepAc4C-master/')
    import sequence_read_save
    import check_parameters
    #fastas = sequence_read_save.read_nucleotide_sequences("input.fasta")
    #path_pos_data = "./DeepAc4C_Datasets/pos_training_samples_cdhit.fasta"
    #path_neg_data = "./DeepAc4C_Datasets/neg_training_samples_cdhit.fasta"
    #fastas = sequence_read_save.read_nucleotide_sequences(path_pos_data, path_neg_data)
    fastas = sequence_read_save.read_nucleotide_sequences("./DeepAc4C_Datasets/pos_training_samples_cdhit.fasta")
    Pse_feature(fastas)