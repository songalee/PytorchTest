###########################################################
#  # -*- coding: utf-8 -*-
#  # !/usr/bin/python
#  Created on 2020年2月12日
#  @author: FrankLee
#  @email: oubasongalee@gmail.com
#  @description: 
###########################################################

import torch as t
import torchvision as tv
import numpy as np
import torch.utils.data as Data
import time
from parameters.settings.path_setting import ProPath

# 超参数
EPOCH = 10
BATCH_SIZE = 100
DOWNLOAD_MNIST = True  # 下过数据的话, 就可以设置成 False
N_TEST_IMG = 10  # 到时候显示 5张图片看效果, 如上图一

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


class DNN(t.nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        # 用模拟的数据试试
        sam_list, X_list, y_list = loadDataVec('train')
        # X_list = [[1.0, 2.0], [1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
        # y_list = [1, 1, 1, 1, 0]
        
        X_np = np.asarray(X_list)
        y_np = np.asarray(y_list)
        X_tensor = t.from_numpy(X_np)
        y_tensor = t.from_numpy(y_np)
        train_data = Data.TensorDataset(X_tensor, y_tensor)
        
        testSam_list, X_test, y_test = loadDataVec('test')
        # X_test = [[1.0, 2.0], [1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
        # y_test = [1, 1, 1, 1, 0]
        
        Xt_np = np.asarray(X_test)
        yt_np = np.asarray(y_test)
        Xt_tensor = t.from_numpy(Xt_np)
        yt_tensor = t.from_numpy(yt_np)
        test_data = Data.TensorDataset(Xt_tensor, yt_tensor)

        '''train_data = tv.datasets.FashionMNIST(
        root="./fashionmnist/",
        train=True,
        transform=tv.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
        )'''

        '''test_data = tv.datasets.FashionMNIST(
        root="./fashionmnist/",
        train=False,
        transform=tv.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
        )'''

        # print(test_data)

        # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
        self.train_loader = t.utils.data.DataLoader(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            shuffle=True)

        self.test_loader = t.utils.data.DataLoader(
            dataset=test_data,
            batch_size=1000,
            shuffle=True)

        self.dnn = t.nn.Sequential(
            t.nn.Linear(6096, 512),
            t.nn.Dropout(0.5),
            t.nn.ELU(),
            t.nn.Linear(512, 128),
            t.nn.Dropout(0.5),
            t.nn.ELU(),
            t.nn.Linear(128, 2),
        )

        self.lr = 0.001
        self.loss = t.nn.CrossEntropyLoss()
        self.opt = t.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):

        nn1 = x.view(-1, 6096)
        # print(nn1.shape)
        out = self.dnn(nn1)
        # print(out.shape)
        return(out)


def train():
    use_gpu = t.cuda.is_available()
    model = DNN()
    if(use_gpu):
        model.cuda()
    print(model)
    loss = model.loss
    opt = model.opt
    dataloader = model.train_loader
    testloader = model.test_loader
    
    for e in range(EPOCH):
        step = 0
        ts = time.time()
        for (x, y) in (dataloader):

            model.train()  # train model dropout used
            step += 1
            b_x = x  # batch x, shape (batch, 28*28)
            # print(b_x.shape)
            b_y = y
            if(use_gpu):
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            out = model(b_x.float())
            losses = loss(out, b_y.long())
            opt.zero_grad()
            losses.backward()
            opt.step()
            if(step % 100 == 0):
                if(use_gpu):
                    print(e, step, losses.data.cpu().numpy())
                else:
                    print(e, step, losses.data.numpy())
                
                model.eval()  # train model dropout not use
                for (tx, ty) in testloader:
                    t_x = tx  # batch x, shape (batch, 28*28)
                    t_y = ty
                    if(use_gpu):
                        t_x = t_x.cuda()
                        t_y = t_y.cuda()
                    t_out = model(t_x.float())
                    if(use_gpu):
                        acc = (np.argmax(t_out.data.cpu().numpy(), axis=1) == t_y.data.cpu().numpy())
                    else:
                        acc = (np.argmax(t_out.data.numpy(), axis=1) == t_y.data.numpy())

                    print(time.time() - ts , np.sum(acc) / 1000)
                    ts = time.time()
                    break  # 只测试前1000个

    t.save(model, modelpath + 'DNN_model.pkl')  # 保存整个网络
    t.save(model.state_dict(), modelpath + 'DNN_model_params.pkl')  # 只保存网络中的参数 (速度快, 占内存少)


if __name__ == "__main__":
    train()
