###########################################################
#  # -*- coding: utf-8 -*-
#  # !/usr/bin/python
#  Created on 2020年2月15日
#  @author: FrankLee
#  @email: oubasongalee@gmail.com
#  @description: 
###########################################################

import torch
import numpy as np
import torch.utils.data as Data
import time
from sklearn.metrics import classification_report
from sklearn import metrics
from models.DNNUse import loadDataVec, DNN
from parameters.settings.path_setting import ProPath
import pickle

split_id = ProPath.split_id
split_dirpath = ProPath.split_dirpath
modelpath = ProPath.modelpath


def loadModel(modelpath):
    dnn_model = torch.load(modelpath)
    return dnn_model


def loadDataLoader():
    testSam_list, X_test, y_test = loadDataVec('test')
    Xt_np = np.asarray(X_test)
    yt_np = np.asarray(y_test)
    Xt_tensor = torch.from_numpy(Xt_np)
    yt_tensor = torch.from_numpy(yt_np)
    test_data = Data.TensorDataset(Xt_tensor, yt_tensor)
    testloader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=50000,
            shuffle=True)
    return testSam_list, testloader


def DNNTest():
    dnn_model = loadModel(modelpath + 'DNN_model.pkl')
    testSam_list, testloader = loadDataLoader()
    
    # 被预测为恶意样本，也确实为恶意样本的列表
    true_maldict = {}
    # 存放被预测为恶意样本，也确实为恶意样本列表的路径
    true_mallistpath = split_dirpath + split_id + '\\DNN_true_maldict.npy'
    dnn_resultpath = split_dirpath + split_id + '\\DNN_result.txt'
    test_Y_list = [[0]]
    sampled_Y_list = [[1]]

    for (tx, ty) in testloader:
        t_x = tx  
        t_y = ty
        t_out = dnn_model(t_x.float())
        acc = (np.argmax(t_out.data.numpy(), axis=1) == t_y.data.numpy())
            
        print('acc:' + str(acc))
            
        y_true1 = t_y.data.numpy()
        y_pred1 = np.argmax(t_out.data.numpy(), axis=1)
        test_Y_list = test_Y_list + y_true1
        sampled_Y_list = sampled_Y_list + y_pred1
        for i in range(len(y_true1)):
            if(y_true1[i] == 1 and y_pred1[i] == 1):
                true_maldict[testSam_list[i]] = 1
                print(testSam_list[i])
    with open(dnn_resultpath, 'wb') as dnn_resultfile:
        y_true = test_Y_list
        y_pred = sampled_Y_list
        print('testModel: ------------------------------------------------------------classification_report: ')
        dnn_resultfile.write('testModel: ------------------------------------------------------------classification_report: \n')
        print(classification_report(y_true, y_pred, digits=4))
        dnn_resultfile.write(classification_report(y_true, y_pred, digits=4))
        print('------------------------------------------------------------confusion_matrix:')
        dnn_resultfile.write('\n------------------------------------------------------------confusion_matrix:\n')
        confusion = metrics.confusion_matrix(y_true, y_pred)
        print(confusion)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        print('------------------------------classfication accuracy:')
        dnn_resultfile.write('\n------------------------------classfication accuracy:\n')
        print((TP + TN) / float(TP + TN + FN + FP))
        print('------------------------------Classification Error:')
        dnn_resultfile.write('\n------------------------------Classification Error:\n')
        print((FP + FN) / float(TP + TN + FN + FP))
        print('------------------------------Sensitivity:Recall:')
        dnn_resultfile.write('\n------------------------------Sensitivity:Recall:\n')
        print(TP / float(TP + FN))
        print('------------------------------Specificity:')
        dnn_resultfile.write('\n------------------------------Specificity:\n')
        print(TN / float(TN + FP))
        print('------------------------------False Positive Rate:')
        dnn_resultfile.write('\n------------------------------False Positive Rate:\n')
        print(FP / float(TN + FP))
        print('------------------------------Precision:')
        dnn_resultfile.write('\n------------------------------Precision:\n')
        print(TP / float(TP + FP))
        print('------------------------------auc:')
        dnn_resultfile.write('\n------------------------------auc:\n')
        print(metrics.roc_auc_score(y_true, y_pred))
        print("---------risk assessment distribution------------------")
        dnn_resultfile.write("\n---------risk assessment distribution------------------\n")
    with open(true_mallistpath, 'wb') as true_mallistfile:
        pickle.dump(true_maldict, true_mallistfile)


def getTestPMalVec():
    # testpmal_vec:是mal且被DNN模型判定为mal的样本的向量。
    testpmal_vecpath = split_dirpath + split_id + '\\test\\DNN_pmal'
    testmal_vecpath = split_dirpath + split_id + '\\test\\mal'
    true_maldictpath = split_dirpath + split_id + '\\DNN_true_maldict.npy'
    with open(true_maldictpath, 'rb') as true_maldictfile:
        true_maldict = pickle.load(true_maldictfile)
    
    with open(testmal_vecpath, 'r') as testmal_vecfile:
        with open(testpmal_vecpath, 'w') as testpmal_vecfile:
            line = testmal_vecfile.readline()
            while(line):
                index = line.index(' ')
                mal = line[0:index]
                if(mal in true_maldict):
                    testpmal_vecfile.write(line)
                line = testmal_vecfile.readline()

        
if __name__ == '__main__':
    DNNTest()
    getTestPMalVec()
    
