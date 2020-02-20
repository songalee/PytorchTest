###########################################################
#  # -*- coding: utf-8 -*-
#  # !/usr/bin/python
#  Created on 2019年12月24日
#  @author: FrankLee
#  @email: oubasongalee@gmail.com
#  @description: 
###########################################################


class ProPath():

    '''源数据相关路径'''
    
    split_id = 'split_1'
    data_dirpath = 'D:\\lisong\\Experiments1\\4_1\\'
    
    # 数据集按照数量被分割成了10份，按照app名字存放的路径
    appname_dirpath = data_dirpath + '\\dataset_splits\\all\\'
    
    # 某部分数据集的文件
    train_cs_path = appname_dirpath + split_id + '\\' + 'train_cs'
    test_cs_path = appname_dirpath + split_id + '\\' + 'test_cs'
    validate_cs_path = appname_dirpath + split_id + '\\' + 'validate_cs'
    # 所有数据集提取的特征存放的路径
    feature_filepath = data_dirpath + '\\feature_vectors\\'
    
    '''生成的数据相关路径'''
    split_dirpath = data_dirpath + 'vec_data\\split\\'
    
    api_per_mappingpath = data_dirpath + 'mapping_5.1.1.csv'
    
    api_perlist_dictpath = data_dirpath + 'api_perlist_dictpath.npy'
    
    # 特征类型与其对应的列表
    feaType_lidictpath = appname_dirpath + split_id + '\\' + 'feaType_lidict.npy'
    
    # 数据集生成的向量表示
    split_id_dirpath = split_dirpath + split_id + '\\'
    
    '''项目相关路径'''
    pro_dirpath = 'D:\\lisong\\ProgramFiles\\eclipse\\workspace\\PytorchTest\\'
    # pro_dirpath = 'D:\\lisong\\ProgramFiles\\eclipse\\workspace\\AdversarialExamAttack\\'
    
    data_dict_path = pro_dirpath + 'parameters\\drebin\\data_dict.npy'
    
    mal_dict_path = pro_dirpath + 'parameters\\drebin\\mal_dict.npy'
    
    family_list_dictpath = pro_dirpath + 'parameters\\drebin\\family_list_dict.npy'
    
    '''生成模型存放的路径'''
    modelpath = split_id_dirpath + 'models\\'
    
    # 从android官网获取的对应关系
    parse_relation_path = pro_dirpath + 'constants\\code_dictlib\\api_dictlibs\\all'
    
    api_perlist_dict1path = pro_dirpath + 'constants\\code_dictlib\\api_dictlibs\\api_perlist_dict1.npy'

