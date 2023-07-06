# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:40:49 2020

@author: Administer
"""
"""
def feature_merge(feature_list,num_id,save_name):
    import numpy as np
    import pandas as pd
    n=0
    for fea in feature_list:
        # print(fea)
        n=n+1
        if n==1:           
            dfx=pd.read_csv(r'./features/mm/'+str(num_id)+'/f_b/'+str(fea),sep=',',header=0,index_col=0)
            # print((dfx.shape[1]))
        else:
            dfn=pd.read_csv(r'./features/mm/'+str(num_id)+'/f_b/'+str(fea),sep=',',header=0,index_col=0)
            dfx=pd.concat([dfx,dfn],axis=1) #方法用于连接两个或多个数组
        dfx1=pd.DataFrame(dfx)  #二维数组创建
        dfx1.to_csv('./features/combined_features/'+str(save_name)+".csv",sep=",")
        
        
#feature_list=['m_cksnap.csv', 'm_kmer.csv',  'm_SCPseDNC.csv','m_PseEIIP.csv',
#                      'm_PseKNC.csv',  'm_SCPseTNC.csv','w2v_feature.csv']
feature_list=['m_cksnap.csv', 'm_kmer.csv',  'm_SCPseDNC.csv','m_PseEIIP.csv',
                      'm_PseKNC.csv',  'm_SCPseTNC.csv']

def feature_combine():
    for num_id in [1,2,3,4,5,6,7,8,9,10]:
        feature_merge(feature_list,num_id,'feature_'+str(num_id))

"""
def feature_merge(feature_list,save_name,kind):
    import numpy as np
    import pandas as pd
    n=0
    for fea in feature_list:
        # print(fea)
        n=n+1
        if n==1:
            dfx=pd.read_csv(r'./features/'+str(kind)+'/mm/f_b/'+str(fea),sep=',',header=None,index_col=0)
            # print((dfx.shape[1]))
        else:
            dfn=pd.read_csv(r'./features/'+str(kind)+'/mm/f_b/'+str(fea),sep=',',header=None,index_col=0)
            dfx=pd.concat([dfx,dfn],axis=1) #方法用于连接两个或多个数组
        dfx1=pd.DataFrame(dfx)  #二维数组创建
        dfx1.to_csv('./features/'+str(kind)+'/combined_features/'+str(save_name)+".csv",sep=",",header=False)

#feature_listneg=['neg_cksnap.csv','neg_kmer.csv','neg_SCPseDNC.csv','neg_PseEIIP.csv',
#                      'neg_PseKNC.csv','neg_SCPseTNC.csv']

#feature_testlistneg=['test_neg_cksnap.csv','test_neg_kmer.csv','test_neg_SCPseDNC.csv','test_neg_PseEIIP.csv',
#                      'test_neg_PseKNC.csv','test_neg_SCPseTNC.csv']


def feature_combine(kind):
    feature_list = ['kmer.csv', 'PseEIIP.csv', 'PseKNC.csv']
    # feature_list = ['ANF.csv','kmer.csv', 'PseEIIP.csv', 'PseKNC.csv']
    #feature_list=['SCPseDNC.csv','PseKNC.csv','SCPseTNC.csv']
    feature_merge(feature_list,'feature', kind)


def feature_combine_test(kind):
    feature_testlist = ['test_kmer.csv', 'test_PseEIIP.csv', 'test_PseKNC.csv']
    # feature_testlist = ['test_ANF.csv','test_kmer.csv', 'test_PseEIIP.csv', 'test_PseKNC.csv']
    feature_merge(feature_testlist,'testfeature',kind)
    


def feature_combine_kmer(kind):
    feature_list = ['kmer.csv']
    feature_merge(feature_list, 'feature', kind)

def feature_combine_ANF(kind):
    feature_list = ['ANF.csv']
    feature_merge(feature_list, 'feature', kind)


def feature_combine_PseEIIP(kind):
    feature_list = ['PseEIIP.csv']
    feature_merge(feature_list, 'feature', kind)

def feature_combine_CKSNAP(kind):
    feature_list = ['CKSNAP.csv']
    feature_merge(feature_list, 'feature', kind)

def feature_combine_PseKNC(kind):
    feature_list = ['PseKNC.csv']
    feature_merge(feature_list, 'feature', kind)


def feature_combine_PseEIIP_kmer(kind):
    feature_list = ['kmer.csv', 'PseEIIP.csv', 'PseKNC.csv']
    feature_merge(feature_list, 'feature', kind)

def feature_combine_PseKNC_PseEIIP(kind):
    feature_list = ['PseEIIP.csv', 'PseKNC.csv']
    feature_merge(feature_list, 'feature', kind)


def feature_combine_kmer_PseKNC_PSEEIIP(kind):
    feature_list = ['kmer.csv', 'PseKNC.csv','PseEIIP.csv']
    feature_merge(feature_list, 'feature', kind)

# def feature_combine_kmer_pse(kind):
#     feature_list = ['PseEIIP.csv', 'kmer.csv']
#     feature_merge(feature_list, 'feature', kind)
# 
# def feature_combine_PseEIIP_PseKNC(kind):
#     feature_list = ['PseEIIP.csv', 'PseKNC.csv']
#     feature_merge(feature_list, 'feature', kind)

#def feature_combine_test2():
#    feature_merge(feature_testlistneg,'testfeature_neg')