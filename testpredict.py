import re
import joblib
import numpy as np
from random import uniform, random, choice, sample, shuffle

# import sns as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, matthews_corrcoef, average_precision_score, \
    recall_score, precision_recall_curve
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
from statistics import mean
from feature_scripts import sequence_read_save

warnings.filterwarnings("ignore")


#读取处理数据
def read_train(fastas):
    fecode = []
    for i in fastas:
        name, sequence= i[0], re.sub('-', '', i[1])  # 把编号和序列遍历出来
        name = name.split('|')
        name = name[1]
        code = []
        code.append(name)
        code.append(sequence)
        fecode.append(code)
    for i in fecode:
        data_result = []
        for name,seq in fecode:
             data_result.append(seq)
        return data_result

def shuffleData(X, y):
    index = [i for i in range(len(X))]
    shuffle(index)
    X = X[index]
    y = y[index]
    return X, y;

def calc(TN, FP, FN, TP):
    SN = TP / (TP + FN)  # recall
    SP = TN / (TN + FP)
    # Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    # F1 = (2 * TP) / (2 * TP + FP + FN)
    fz = TP * TN - FP * FN
    fm = (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)
    MCC = fz / pow(fm, 0.5)
    return SN, SP, ACC, MCC

def t_test(d1,d2,label):
    # print(stats.levene(d1,d2))#如果返回结果的P值远大于0.05，那么我们认为两总体具有方差齐性
    print("Null Hypothesis: mean(d1) = mean(d2), α= 0.05")
    # 执行 T 检验
    t_statistic, p_value = stats.ttest_ind(d1, d2,equal_var=True)
    #ttest,pval = ttest_rel(d1,d2)
    print("模型 "+str(label)+" 的T statistic:", t_statistic)
    print("模型 "+str(label)+" 的P-value:", p_value)
    if p_value < 0.05:
        print("模型"+str(label)+"与stacking有显著性差异") #表示有显著性差异
    else:
        print("模型"+str(label)+"与stacking没有显著性差异")

if __name__ == '__main__':
    import os
    import feature_scripts.sequence_read_save
    import os, sys
    import pandas as pd
    sys.path.append('feature_scripts/')
    import feature_scripts.Kmer, feature_scripts.CKSNAP, feature_scripts.PseEIIP, feature_scripts.PseFeature,feature_scripts.w2v_kmer_corpus_feature, feature_scripts.ANF
    import feature_scripts.sequence_read_save
    import feature_scripts.MinMax
    import feature_scripts.feature_combine


    print("step_1: sequence checking......")
    pos_fastas= "Datasets/pos_test_samples_cdhit.fasta"
    neg_fastas = 'Datasets/neg_test_samples_cdhit.fasta'
    #pos_fastas = "Datasets/iddata/ac4c_pos_test.fasta"
    #neg_fastas = 'Datasets/iddata/ac4c_neg_test.fasta'
    fastas_pos = sequence_read_save.read_nucleotide_sequences(pos_fastas)
    fastas_neg = sequence_read_save.read_nucleotide_sequences(neg_fastas)
    #fastas="Datasets/ac4c_testing.fasta"
    #data_train= sequence_read_save.read_nucleotide_sequences(fastas)
    #x_train_pos = read_train(fastas_pos)
    #x_train_neg = read_train(fastas_neg)
    kind='test'
    data_test = fastas_neg + fastas_pos
    print("step_1: embedding ......")
    feature_scripts.Kmer.kmer_features_test(data_test,kind)
    # feature_scripts.ANF.ANF_features_test(data_test, kind)
    #feature_scripts.CKSNAP.cksnap_feature_test(data_train,kind)
    feature_scripts.PseEIIP.PseEIIP_feature_test(data_test,kind)
    feature_scripts.PseFeature.Pse_feature_test(data_test,kind)
    # print("step_3: feature scale and selection......")
    # feature_scripts.w2v_kmer_corpus_feature.w2v_kmer_corpus_test()
    # feature_scripts.w2v_kmer_corpus_feature.w2v_features_test(kind)
    # feature_scripts.w2v_kmer_corpus_feature.word2_vectest(kind)
    feature_scripts.MinMax.MinMax_normalized_test(kind)
    feature_scripts.feature_combine.feature_combine_test(kind)
    # dfn = pd.read_csv("./features/"+str(kind)+"/combined_features/word2vectest.csv", sep=',', header=None).to_numpy()
    # data_test = np.concatenate([dfx, dfn], axis=1)
    data_test= pd.read_csv("./features/" + str(kind) + "/combined_features/testfeature.csv", sep=',',
                             header=None).to_numpy()
    # np.savetxt("./features/"+str(kind)+"/test_result.csv", data_test, fmt="%s", delimiter=',')
    data_test = np.array(data_test)
    y_test = data_test[1:, 0]
    x_test = data_test[1:, 1:]
    code = []
    for i in y_test:
        name = i.split('|')
        name = name[1]
        code.append(name)
    y_test = np.array(code)
    x_test = np.array(x_test)
    test_x, test_y = shuffleData(x_test, y_test);
    under_sample = ClusterCentroids(random_state=20)
    # under_sample = RandomUnderSampler()
    X_test, y_test = under_sample.fit_resample(test_x, test_y)

    list = ['stacking','Logistic','knn','svm','RF','MLP']    #,'GBDT','Logistic',
    colorlist = ['red','pink','blue','green','orange','black']  #,
    k = 0
    t_stacking = []
    t_LR = []
    t_knn = []
    t_svm = []
    t_RF = []
    t_MLP = []
    for label in list:
        pred_proba = []
        #记录模型运行开始时间
        start_time = time.time()
        for mode_num in range(1,11):
            model = joblib.load('./model/test/' + label + '/'+str(mode_num)+'predict_protein.pkl', 'r+')
            prediction = model.predict_proba(X_test)[:, 1]
            pred_proba.append(prediction)
        pred_proba_ave = np.mean(np.array(pred_proba).T, axis=1)
        pred_proba_ave = [np.round(x, 4) for x in pred_proba_ave]
        tempLabel1 = np.zeros(shape=y_test.shape, dtype=np.int32)
        for i in range(len(y_test)):
            if pred_proba_ave[i] < 0.5:
                tempLabel1[i] = 0;
            else:
                tempLabel1[i] = 1;
        test_y = [int(i) for i in y_test]

        # 混淆矩阵（y:是样本真实分类结果，tempLabel：是样本预测分类结果）
        confusion = confusion_matrix(test_y, tempLabel1)
        TN, FP, FN, TP = confusion.ravel()
        sn, sp, acc, mcc = calc(TN, FP, FN, TP)
        # 定义计算t-test
        if (str(label) == 'stacking'):
            t_stacking.extend(pred_proba_ave)
        if (str(label) == 'Logistic'):
            t_LR.extend(pred_proba_ave)
        if (str(label) == 'knn'):
            t_knn.extend(pred_proba_ave)
        if (str(label) == 'svm'):
            t_svm.extend(pred_proba_ave)
        if (str(label) == 'RF'):
            t_RF.extend(pred_proba_ave)
        if (str(label) == 'MLP'):
            t_MLP.extend(pred_proba_ave)

        # 计算AUC
        au = roc_auc_score(test_y, pred_proba_ave)
        prc = average_precision_score(test_y, pred_proba_ave)
        print("sn: %0.4f [%s]" % (sn, label))
        print("sp: %0.4f [%s]" % (sp, label))
        print("ACC: %0.4f [%s]" % (acc, label))
        print("MCC: %0.4f [%s]" % (mcc, label))
        print("AUC: %0.4f [%s]" % (au, label))
        print("PRC: %0.4f [%s]" % (prc, label))
        # 保存最优参数
        font = {'style': 'italic'}
        #计算模型运行时间
        end_time = time.time()
        print("模型"+str(label)+"运行时间：", end_time - start_time, "秒")

        #-----画acc,sp,mcc,sn图
        plt.subplot(1, 2, 1)
        plt.xticks(fontproperties='Times New Roman', fontsize=8)
        plt.yticks(fontproperties='Times New Roman', fontsize=20)
        plt.text(str(label),acc,'%.4f' %acc, ha='center', fontproperties='Times New Roman', fontsize=10,
                 zorder=10)
        # 设置图片名称
        plt.title("Independent test acc")
        # 设置x轴标签名
        plt.xlabel("Algorithm type")
        # 设置y轴标签名
        plt.ylabel("ACC")
        plt.bar(str(label), acc, color=colorlist[k])


        plt.subplot(1, 2, 2)
        plt.xticks(fontproperties='Times New Roman', fontsize=8)
        plt.yticks(fontproperties='Times New Roman', fontsize=20)
        plt.text(str(label), mcc,'%.4f' %mcc, ha='center', fontproperties='Times New Roman', fontsize=10,
                 zorder=10)
        # 设置图片名称
        plt.title("Independent test mcc")
        # 设置x轴标签名
        plt.xlabel("Algorithm type")
        # 设置y轴标签名
        plt.ylabel("MCC")
        plt.bar(str(label), mcc, color=colorlist[k])
        k = k + 1
    plt.show()
    # 计算t-test
    # 比较模型stacking和LR差异
    t_test(t_stacking, t_LR, 'LR')
    # 比较模型stacking和knn的差异
    t_test(t_stacking, t_knn, 'knn')
    # 比较模型stacking和svm的差异
    t_test(t_stacking, t_svm, 'svm')
    # 比较模型stacking和RF的差异
    t_test(t_stacking, t_RF, 'RF')
    # 比较模型stacking和MLP的差异
    t_test(t_stacking, t_MLP, 'MLP')
    q = 0
    for label in list:
        pred_prob = []
        for mode_num in range(1, 11):
            model1 = joblib.load('./model/test/' + label + '/' + str(mode_num) + 'predict_protein.pkl', 'r+')
            #############画图部分
            #1.计算auc
            prediction = model1.predict_proba(X_test)[:,1]
            pred_prob.append(prediction)
        pred_proba_av = np.mean(np.array(pred_prob).T, axis=1)
        pred_proba_av = [np.round(x, 4) for x in pred_proba_av]
        test= [int(i) for i in y_test]
        fpr, tpr, threshold = metrics.roc_curve(test, pred_proba_av)
        roc_auc = auc(fpr, tpr)
        # 2.绘制曲线
        plt.title('ROC')
        plt.plot(fpr, tpr, color=colorlist[q],label=str(label) + ' AUC = %0.4f' %roc_auc)
        #plt.plot(mean(fp), mean(tp), color=colorlist[k], label=str(label) + ' AUC = %0.3f' % mean(ro_auc))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # 设置图片名称
        plt.title("Independent test results")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        q = q + 1
    plt.show()
    t=0
    for label in list:
        pred_prob = []
        for mode_num in range(1, 11):
            model2 = joblib.load('./model/test/' + label + '/' + str(mode_num) + 'predict_protein.pkl', 'r+')
            #############画图部分
            # 1.计算auc
            prediction = model2.predict_proba(X_test)[:, 1]
            pred_prob.append(prediction)
        #二.计算auprc图
        pred_proba_av = np.mean(np.array(pred_prob).T, axis=1)
        pred_proba_av = [np.round(x, 4) for x in pred_proba_av]
        test = [int(i) for i in y_test]
        precision, recall, _ = precision_recall_curve(test, pred_proba_av)
        # average_precision值的计算
        auPRC = average_precision_score(test, pred_proba_av)
        # PRC曲线绘制
        # plt.figure(figsize=(6, 6))
        plt.title('auPRC')
        plt.plot(recall, precision, color=colorlist[t], label=str(label) + ' auPRC = %0.4f' %auPRC)
        #plt.plot(mean(fp), mean(tp), color=colorlist[k], label=str(label) + ' AUC = %0.3f' % mean(ro_auc))
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title('auPRC')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot([0, 1], [1, 0], color='m', linestyle='--')
        plt.legend(loc='lower right')
        t=t+1
        # 保存图片(常用格式如下)
        # plt.savefig('auPRC curves.jpg', dpi=300)
        # plt.savefig('auPRC curves.pdf', dpi=300)
        # plt.savefig('auPRC curves.png', dpi=300)
    plt.show()
