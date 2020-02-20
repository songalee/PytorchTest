###########################################################
#  # -*- coding: utf-8 -*-
#  # !/usr/bin/python
#  Created on 2019年12月26日
#  @author: FrankLee
#  @email: oubasongalee@gmail.com
#  @description: 
###########################################################

from parameters.settings.path_setting import ProPath
import pickle
import random
import numpy as np
import torch
from models.DNNModelTest import loadModel


split_dirpath = ProPath.split_dirpath
split_id = ProPath.split_id

api_perlist_dictpath = ProPath.api_perlist_dictpath
feaType_lidictpath = ProPath.feaType_lidictpath

api_perlist_dict1path = ProPath.api_perlist_dict1path
modelpath = ProPath.modelpath


def getFeatureIDDict():
    newfeature_dict = {}
    feature_dictpath = split_dirpath + split_id + '\\feature_dict.npy'
    with open(feature_dictpath, 'rb') as feature_dictfile:
        feature_dict = pickle.load(feature_dictfile)
    counter = 0
    for key in feature_dict:
        newfeature_dict[key] = counter
        counter = counter + 1
    return newfeature_dict


def getAPIPerlistDict():
    with open(api_perlist_dict1path, 'rb') as file:
        api_perlist_dict = pickle.load(file)
    return api_perlist_dict


def getFeaType_lidict():
    with open(feaType_lidictpath, 'rb') as file:
        feaType_lidict = pickle.load(file)
    return feaType_lidict


newfeature_dict = getFeatureIDDict()
api_perlist_dict = getAPIPerlistDict()
feaType_lidict = getFeaType_lidict()


def changeVec(vec_list, feature_seedlist):
    newvec_list = vec_list.copy()
    for feature_seed in feature_seedlist:
        index = newfeature_dict[feature_seed]
        newvec_list[index] = 1
    # 根据newfeature_dict修改某些特征的值
    return newvec_list


def prelabel(vecs_list, model):
    label_list = model.predict(vecs_list)
    return label_list


def genRanFeaSeedList(seed_num):
    feature_seedlist = []
    feature_list = list(newfeature_dict.keys())
    for i in range(seed_num):
        ran_index = random.randint(0, 6000)
        feature_seedlist.append(feature_list[ran_index])
        
    return feature_seedlist


def genEditFeaSeedList(seed_num):
    # newfeature_dict
    # api_perlist_dict
    
    feature_seedlist = []
    permission_list = feaType_lidict['permission']
    apiCall_perlist = feaType_lidict['api_call']
    
    # 获取的api对应的permission的列表
    permission_mapping_list = []
    
    # 需要修改api的数量
    api_randomNum = random.randint(0, seed_num - 1)

    # 随机找到需要修改的api的索引
    for i in range(api_randomNum):
        ran_index = random.randint(0, len(apiCall_perlist) - 1)
        feature_seedlist.append('api_call::' + apiCall_perlist[ran_index])
        # 获取api对应的permission
        if(apiCall_perlist[ran_index] not in api_perlist_dict):
            continue
        per_list = api_perlist_dict[apiCall_perlist[ran_index]]
        # if('android.permission.DUMP' not in per_list):
        #    print('bbbbbbb:' + str(per_list))
        for per in per_list:
            permission_mapping_list.append(per)
        
    # 随机找到需要修改的permission
    for j in range(seed_num - api_randomNum):
        ran_index = random.randint(0, len(permission_list) - 1)
        permission = permission_list[ran_index]
        if(permission not in permission_mapping_list):
            feature_seedlist.append('permission::' + permission_list[ran_index])
            
    return feature_seedlist


def main():
    DNN_model = loadModel(modelpath + 'DNN_model.pkl')
    testpmal_vecpath = split_dirpath + split_id + '\\test\\DNN_pmal'
    with open(testpmal_vecpath, 'r') as testpmal_vecfile:
        line = testpmal_vecfile.readline()
        line_clips = line.split(' ')
        mal = line_clips[0]
        line_clips = line_clips[1:len(line_clips) - 1]
        line_clips = list(map(int, line_clips))
        
        while(True):
            print('Seeding--------------------------------------')
            feature_seedlist = genEditFeaSeedList(300)
            newvec_list = changeVec(line_clips, feature_seedlist)
            print('features:' + str(feature_seedlist))
            
            #预测样本的恶意与否
            newvec_np = np.asarray([newvec_list])
            newvec_tensor = torch.from_numpy(newvec_np)
            t_out = DNN_model(newvec_tensor)
            print('predict:' + str(np.argmax(t_out.data.numpy(), axis=1)))


if __name__ == '__main__':
    main()
    
