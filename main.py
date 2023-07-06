import re

from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import joblib
import numpy as np
from random import uniform, random, choice, sample, shuffle
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, RepeatedEditedNearestNeighbours, ClusterCentroids
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
# import tensorflow as tf
# from tensorflow import keras
import warnings
from statistics import mean
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, recall_score
from feature_scripts import sequence_read_save

warnings.filterwarnings("ignore")


# 读取处理数据
def read_train(fastas):
    fecode = []
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])  # 把编号和序列遍历出来
        name = name.split('|')
        name = name[1]
        code = []
        code.append(name)
        code.append(sequence)
        fecode.append(code)
    for i in fecode:
        data_result = []
        for name, seq in fecode:
            data_result.append(seq)
        return data_result


def shuffleData(X, y):
    index = [i for i in range(len(X))]
    shuffle(index)
    X = X[index]
    y = y[index]
    return X, y;


# 交叉验证
def get_k_fold_data(k, i, X, y):
    fold_size = X.shape[0] // k

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = np.concatenate([X[0:val_start], X[val_end:]], 0)
        y_train = np.concatenate([y[0:val_start], y[val_end:]], 0)
    else:
        X_valid, y_valid = X[val_start:], y[val_start:]
        X_train = X[0:val_start]
        y_train = y[0:val_start]

    return X_train, y_train, X_valid, y_valid


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


def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sn, sp, acc, mcc = calc(tn, fp, fn, tp)
    return {'sn': sn, 'sp': sp, 'acc': acc, 'mcc': mcc}


if __name__ == '__main__':
    import os
    import feature_scripts.sequence_read_save
    import os, sys
    import pandas as pd

    sys.path.append('feature_scripts/')
    np.random.seed(42)
    import feature_scripts.Kmer, feature_scripts.ANF, feature_scripts.PseEIIP, feature_scripts.MinMax, \
        feature_scripts.w2v_kmer_corpus_feature
    import feature_scripts.sequence_read_save
    import feature_scripts.feature_combine

    # os.chdir('G:/AC4C/')

    fastas_pos = "Datasets/pos_training_samples_cdhit.fasta"
    fastas_neg = 'Datasets/neg_training_samples_cdhit.fasta'
    kind = 'ANF'
    print("step_1: sequence checking......")
    fastas_pos = sequence_read_save.read_nucleotide_sequences(fastas_pos)
    fastas_neg = sequence_read_save.read_nucleotide_sequences(fastas_neg)
    print("fastas_pos", type(fastas_neg))
    data_train = fastas_neg + fastas_pos
    print("step_2: embedding ......")
    #feature_scripts.Kmer.kmer_features(data_train, kind)
    #feature_scripts.ANF.ANF_features(data_train, kind)
    #feature_scripts.PseEIIP.PseEIIP_feature(data_train, kind)
    print("step_3: feature scale and selection......")
    feature_scripts.MinMax.MinMax_normalized_ANF(kind)
    # feature_scripts.w2v_kmer_corpus_feature.w2v_kmer_corpus()
    # feature_scripts.w2v_kmer_corpus_feature.w2v_features(kind)
    feature_scripts.feature_combine.feature_combine_ANF(kind)
    data_train = pd.read_csv("./features/" + str(kind) + "/combined_features/feature.csv", sep=',',
                             header=None).to_numpy()
    y_train = data_train[1:, 0]
    x_train = data_train[1:, 1:]
    code = []
    for i in y_train:
        name1=str(i)
        name = name1.split('|')
        name = name[1]
        code.append(name)
    train_y = np.array(code)
    train_x = np.array(x_train)
    # GBDT作为基模型的特征选择,选择出最重要的20个特征
    # 1.画图（画出最重要的20个特征）
    # x_new = SelectFromModel(GradientBoostingClassifier(), max_features=200, threshold=-np.inf).fit_transform(train_x,
    #                                                                                                         train_y)  # .transform(train_x)  #
    # # print(train_x[x.get_support()])
    # clf = GradientBoostingClassifier(n_estimators=50)
    # clf = clf.fit(x_new, train_y)
    # # print("特征重要性", clf.feature_importances_)
    # importances = clf.feature_importances_
    # # 计算标准差
    # indices = np.argsort(importances)[::-1]
    # # 获得特征排名结果
    # print("Feature ranking:")
    # for f in range(x_new.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #
    # # Plot the impurity-based feature importances of the forest
    # # 画出各个特征的重要性
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(x_new.shape[1]), importances[indices],
    #         color="r", align="center", width=1, linewidth=0.1, edgecolor='b')
    # plt.xticks(range(x_new.shape[1]), indices)
    # plt.xlim([-1, x_new.shape[1]]);
    # plt.xlabel("Features in feature importance value order")
    # plt.ylabel("Feature importance values")

    # 2.选择出最重要的300个特征
    #x_new = SelectFromModel(GradientBoostingClassifier(), max_features=350, threshold=-np.inf).fit_transform(train_x,train_y)
    # np.savetxt("C:/Users/lou/Desktop/result.csv", x, fmt='%s', delimiter=',')
    train_x, train_y = shuffleData(train_x, train_y)
    print("------------------------------------------------------------")
    under_sample = ClusterCentroids(random_state=42)
    x_train, y_train = under_sample.fit_resample(train_x, train_y)

    level0 = list()
    level0.append(('Logistic', LogisticRegression(random_state=0, max_iter=1300)))
    level0.append(('knn', KNeighborsClassifier(n_neighbors=80, leaf_size=17)))
    level0.append(('DecisionTree',
                   DecisionTreeClassifier(max_depth=40, max_leaf_nodes=None, min_samples_leaf=20, min_samples_split=78,
                                          random_state=30)))
    level0.append(('GaussianNB', GaussianNB(var_smoothing=0.007965430072059344)))
    # level0.append(('svm', SVC(C= 1.6134, kernel='rbf', degree=0.2651, gamma="auto", coef0=0.0, shrinking=True, probability=True,
    #                           tol=0.078, class_weight=None, random_state=30)))
    level0.append(
        ('RandomForestClassifier',
         RandomForestClassifier(n_estimators=40, min_samples_split=78, min_samples_leaf=20, random_state=30)))
    level0.append(('MLPClassifier', MLPClassifier(activation='relu', alpha=1e-05, batch_size=37, beta_1=0.9,
                                                  beta_2=0.999, epsilon=1e-08,
                                                  hidden_layer_sizes=(11,), learning_rate='constant',
                                                  learning_rate_init=0.021, max_iter=8532, momentum=0.58)))

    level1 = LogisticRegression(random_state=0, max_iter=1300)
    clf1 = StackingClassifier(estimators=level0, final_estimator=level1)
    clf2 = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
                                  random_state=30)
    clf3 = KNeighborsClassifier(n_neighbors=5)
    clf4 = GaussianNB(var_smoothing=1e-09)
    clf6 = RandomForestClassifier(n_estimators=10, min_samples_split=10, min_samples_leaf=5, random_state=30)
    clf7 = MLPClassifier(activation='relu', alpha=1e-05, batch_size=5, beta_1=0.9,
                         beta_2=0.999, epsilon=1e-08,
                         hidden_layer_sizes=(3,), learning_rate='constant',
                         learning_rate_init=0.001, max_iter=1000, momentum=0.9)

    # 参数寻优：
    param_grid1 = {'max_iter': [1100, 1150, 1200, 1250, 1300, 1350]}

    param_grid2 = {'n_neighbors': [1, 2, 3, 5, 8, 10, 20]}
    param_grid3 = {'max_depth': [5, 10, 15, 20, 30, 40, 50],
                   'min_samples_leaf': [1, 2, 3, 4, 5, 6],
                   'min_samples_split': [2, 5, 8, 10, 12]}
    param_grid4 = {'var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02]}
    param_grid5 = {'C': [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1, 1.2],
                   'degree': [2, 3, 4, 5, 6, 7, 8, 9],
                   'tol': [0.01, 0.1, 0.001, 0.0001]}

    param_grid6 = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 10],
                   'min_samples_split': [2, 3, 5, 8, 10],
                   'min_samples_leaf': [1, 2, 3, 5, 10]}
    param_grid7 = {'batch_size': [2, 5, 8, 10, 15, 20, 25, 30],
                   'learning_rate_init': [0.001, 0.01, 0.1],
                   'max_iter': [1000, 2000, 5000],
                   'momentum': [0.8, 0.9, 0.6, 0.3, 0.1],
                   'hidden_layer_sizes': [1, 3, 5, 10, 12]}

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42)
    # define the search
    grid_Logistic = BayesSearchCV(estimator=LogisticRegression(), search_spaces=param_grid1, n_jobs=-1, cv=cv)
    grid_KNeighbors = BayesSearchCV(estimator=KNeighborsClassifier(), search_spaces=param_grid2, n_jobs=-1, cv=cv)
    grid_DecisionTree = BayesSearchCV(estimator=DecisionTreeClassifier(), search_spaces=param_grid3, n_jobs=-1, cv=cv)
    grid_GaussianNB = BayesSearchCV(estimator=GaussianNB(), search_spaces=param_grid4, n_jobs=-1, cv=cv)
    grid_SVM = BayesSearchCV(estimator=SVC(probability=True), search_spaces=param_grid5, n_jobs=-1, cv=cv)
    grid_Forest = BayesSearchCV(estimator=RandomForestClassifier(), search_spaces=param_grid6, n_jobs=-1, cv=cv)
    grid_MLP = BayesSearchCV(estimator=MLPClassifier(), search_spaces=param_grid7, n_jobs=-1, cv=cv)
    SN = []
    SP = []
    ACC = []
    MCC = []
    AUC = []
    PRC = []
    k = 1
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for clf, label in zip(
            [clf1, grid_Logistic, grid_KNeighbors, grid_DecisionTree, grid_GaussianNB, grid_Forest, grid_MLP],
            ['stacking', 'Logistic', 'knn', 'DecisionTree', 'GaussianNB', 'RandomForestClassifier', 'MLPClassifier']):
        # 保存最优参数
        if (str(label) != 'stacking'):
            if (str(label) != 'VotingClassifier'):
                print("输出最优参数" + str(label) + "模型")
                #print("using parms: %s" % (clf.best_estimator_))
        for train_index, test_index in kf.split(x_train):
            this_train_x, this_train_y = x_train[train_index], y_train[train_index]  # 本组训练集
            this_test_x, this_test_y = x_train[test_index], y_train[test_index]  # 本组验证集
            # 训练本组的数据，并计算准确率
            clf.fit(this_train_x, this_train_y)
            prediction = clf1.predict_proba(this_test_x)[:, 1]
            # test_pred = [int(i) for i in predict_proba]
            tempLabel = np.zeros(shape=this_test_y.shape, dtype=np.int32)
            this_test_y = [int(i) for i in this_test_y]
            for i in range(len(this_test_y)):
                if prediction[i] < 0.5:
                    tempLabel[i] = 0;
                else:
                    tempLabel[i] = 1;
            # 混淆矩阵（y:是样本真实分类结果，tempLabel：是样本预测分类结果）
            confusion = confusion_matrix(this_test_y, tempLabel)
            TN, FP, FN, TP = confusion.ravel()
            sn, sp, acc, mcc = calc(TN, FP, FN, TP)
            # 计算AUC
            auc = roc_auc_score(this_test_y, prediction)
            prc = average_precision_score(this_test_y, prediction)
            # print("sn个: %0.5f (+/- %0.4f) [%s]" % (sn, sn.std(), label))
            # print("sp个: %0.5f (+/- %0.4f) [%s]" % (sp, sn.std(), label))
            # print("acc个: %0.5f (+/- %0.4f) [%s]" % (acc, acc.std(), label))
            # print("mcc个: %0.5f (+/- %0.4f) [%s]" % (mcc, mcc.std(), label))
            # print("auc个: %0.5f (+/- %0.4f) [%s]" % (auc, auc.std(), label))
            # print("PRC个: %0.5f (+/- %0.4f) [%s]" % (prc, prc.std(), label))
            SN.append(sn)
            SP.append(sp)
            MCC.append(mcc)
            ACC.append(acc)
            AUC.append(auc)
            PRC.append(prc)
            joblib.dump(clf1, './model/ANF/' + str(label) +'/'+str(k) + 'predict_protein.pkl')
            k = k + 1
        print("SN: %0.5f [%s]" % (mean(SN), label))
        print("SP: %0.5f [%s]" % (mean(SP), label))
        print("ACC: %0.5f [%s]" % (mean(ACC), label))
        print("MCC: %0.5f [%s]" % (mean(MCC), label))
        print("AUC: %0.5f [%s]" % (mean(AUC), label))
        print("PRC: %0.5f [%s]" % (mean(PRC), label))

    # clf.fit(x_train, y_train)
        # #输出评价指标
        # AUC = cross_val_score(clf, x_train, y_train, cv=10, scoring='roc_auc')
        # cv_results = cross_validate(clf, x_train, y_train, cv=10, scoring=confusion_matrix_scorer)
        # print("sn: %0.5f (+/- %0.4f) [%s]" % (cv_results['test_sn'].mean(), cv_results['test_sn'].std(), label))
        # print("sp: %0.5f (+/- %0.4f) [%s]" % (cv_results['test_sp'].mean(), cv_results['test_sp'].std(), label))
        # print("acc: %0.5f (+/- %0.4f) [%s]" % (cv_results['test_acc'].mean(), cv_results['test_acc'].std(), label))
        # print("mcc: %0.5f (+/- %0.4f) [%s]" % (cv_results['test_mcc'].mean(), cv_results['test_mcc'].std(), label))
        # print("AUC: %0.5f (+/- %0.4f) [%s]" % (AUC.mean(), AUC.std(), label))
    plt.show()






