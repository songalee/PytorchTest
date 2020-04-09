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
from models.CNNTest import loadModel
from models.CNNUse import CNN

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
    CNN_model = loadModel(modelpath + 'CNN_model.pkl')
    testpmal_vecpath = split_dirpath + split_id + '\\test\\CNN_pmal'
    attack_resultpath = split_dirpath + split_id + '\\test\\CNN_attack_result'
    # setting
    mutant_size = 300  # 变种向量的数量
    mutant_times = 1000  # 变种的次数
    # 样本个数的限制
    sIndex_size = 3
    
    with open(attack_resultpath, 'w') as attack_resultfile:
        attack_resultfile.write('##############Settings##################\n')
        attack_resultfile.write('##mutant_size:' + str(mutant_size) + '\n')
        attack_resultfile.write('##mutant_times:' + str(mutant_times) + '\n')
        attack_resultfile.write('##sIndex_size:' + str(sIndex_size) + '\n')
        attack_resultfile.write('##############Settings##################\n')
        # 样本的个数。
        sIndex_count = 0
        
        # 变种成功的总数量
        total_success_count = 0
        # 变种的次数计数器
        total_mutant_counter = 0
        
        with open(testpmal_vecpath, 'r') as testpmal_vecfile:
            line = testpmal_vecfile.readline()
            while(line):
                sIndex_count = sIndex_count + 1
                print('The ' + str(sIndex_count) + '\'s sample####################')
                attack_resultfile.write('The ' + str(sIndex_count) + '\'s sample####################\n')
                if(sIndex_count > sIndex_size):
                    break
                line_clips = line.split(' ')
                mal = line_clips[0]
                line_clips = line_clips[1:len(line_clips) - 1]
                line_clips = list(map(int, line_clips))
                
                # 变种成功的数量
                success_count = 0
                # 变种的次数计数器
                mutant_counter = 0
                
                # 预统计攻击效果
                while(mutant_counter < mutant_times):
                    mutant_counter = mutant_counter + 1
                    total_mutant_counter = total_mutant_counter + 1
                    print('Seeding--------------------------------------')
                    attack_resultfile.write('Seeding--------------------------------------\n')
                    print('mutant_counter:' + str(mutant_counter))
                    attack_resultfile.write('mutant_counter:' + str(mutant_counter) + '\n')
                    feature_seedlist = genEditFeaSeedList(mutant_size)
                    newvec_list = changeVec(line_clips, feature_seedlist)
                    print('features:' + str(feature_seedlist))
                    attack_resultfile.write('features:' + str(feature_seedlist) + '\n')
                    
                    # 预测样本的恶意与否
                    newvec_np = np.asarray([newvec_list])
                    newvec_tensor = torch.from_numpy(newvec_np)
                    t_out, _ = CNN_model(newvec_tensor.float())
                    result_out = np.argmax(t_out.data.numpy(), axis=1)
                    print('predict:' + str(result_out))
                    attack_resultfile.write('predict:' + str(result_out) + '\n')

                    if(result_out[0] == 0):
                        success_count = success_count + 1
                        total_success_count = total_success_count + 1
                success_rate = success_count / mutant_counter
                print('success_count:' + str(success_count) + ';mutant_counter:' + str(mutant_counter))
                attack_resultfile.write('success_count:' + str(success_count) + ';mutant_count:' + str(mutant_counter) + '\n')
                print('one sample\'s success_rate:' + str(success_rate))
                attack_resultfile.write('one sample\'s success_rate:' + str(success_rate) + '\n')
                line = testpmal_vecfile.readline()
            total_success_rate = total_success_count / total_mutant_counter   
            print('total_success_count:' + str(total_success_count) + ';total_mutant_counter:' + str(total_mutant_counter))
            attack_resultfile.write('total_success_count:' + str(total_success_count) + ';total_mutant_counter:' + str(total_mutant_counter) + '\n')
            print('samples\' total_success_rate:' + str(total_success_rate))
            attack_resultfile.write('samples\' total_success_rate:' + str(total_success_rate) + '\n')


if __name__ == '__main__':
    main()
    
