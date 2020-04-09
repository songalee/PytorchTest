###########################################################
#  # -*- coding: utf-8 -*-
#  # !/usr/bin/python
#  Created on 2020年4月9日
#  @author: FrankLee
#  @email: oubasongalee@gmail.com
#  @description: 
###########################################################

from parameters.settings.path_setting import ProPath

flag_folder = 'train'
split_id_dirpath = ProPath.split_id_dirpath
modelpath = ProPath.modelpath

def loadDataVec(data_flag):
    
    flagvec_dirpath = split_id_dirpath + data_flag + '\\'
    X_list = []
    y_list = []
    # 存放样本名的列表
    sam_list = []
    malvec_path = flagvec_dirpath + 'mal'
    benvec_path = flagvec_dirpath + 'ben'
    with open(malvec_path, 'r') as malvec_file:
        line = malvec_file.readline()
        while(line):
            line = line.replace('\n', '')
            line_clips = line.split(' ')
            sam_list.append(line_clips[0])
            line_clips = line_clips[1:len(line_clips) - 1]
            X_list.append(list(map(float, line_clips)))
            y_list.append(1)
            line = malvec_file.readline()
            print(line)
            
    with open(benvec_path, 'r') as benvec_file:
        line = benvec_file.readline()
        while(line):
            line = line.replace('\n', '')
            line_clips = line.split(' ')
            sam_list.append(line_clips[0])
            line_clips = line_clips[1:len(line_clips) - 1]
            X_list.append(list(map(float, line_clips)))
            y_list.append(0)
            line = benvec_file.readline()
            print(line)
    return sam_list, X_list, y_list