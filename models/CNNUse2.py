###########################################################
#  # -*- coding: utf-8 -*-
#  # !/usr/bin/python
#  Created on 2020年4月9日
#  @author: FrankLee
#  @email: oubasongalee@gmail.com
#  @description: 
###########################################################

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
from parameters.settings.path_setting import ProPath
from data_factory.seneledataLoader import loadDataVec
 
torch.manual_seed(1)  # reproducible

flag_folder = 'train'
split_id_dirpath = ProPath.split_id_dirpath
modelpath = ProPath.modelpath

# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001  # learning rate


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        
        # 用模拟的数据试试
        _, X_list, y_list = loadDataVec('train')
        # X_list = [[1.0, 2.0, 1.0, 2.0], [1.0, 1.0, 1.0, 2.0], [1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        # y_list = [1, 1, 1, 1, 0]
        del_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        X_np = np.asarray(X_list)
        X_np = np.delete(X_np, del_list, axis=1)
        y_np = np.asarray(y_list)
        X_tensor = torch.from_numpy(X_np)
        y_tensor = torch.from_numpy(y_np)
        X_tensor = X_tensor.reshape(1, 78, 78)
        train_data = Data.TensorDataset(X_tensor, y_tensor)
        
        _, X_test, y_test = loadDataVec('test')
        # X_test = [[1.0, 2.0], [1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
        # y_test = [1, 1, 1, 1, 0]
        
        Xt_np = np.asarray(X_test)
        Xt_np = np.delete(Xt_np, del_list, axis=1)
        yt_np = np.asarray(y_test)
        Xt_tensor = torch.from_numpy(Xt_np)
        Xt_tensor = Xt_tensor.reshape(1, 78, 78)
        yt_tensor = torch.from_numpy(yt_np)
        test_data = Data.TensorDataset(Xt_tensor, yt_tensor)

        # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=1000,
            shuffle=True)
        
        self.conv1 = nn.Sequential(# input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(# input shape (1, 28, 28)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x  # return x for visualization

 
def train():
    
    cnn = CNN()
    print(cnn)  # net architecture
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    
    train_loader = cnn.train_loader
    testloader = cnn.test_loader
    
    # training and testing
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            b_x = Variable(x)  # batch x
            b_y = Variable(y)  # batch y
    
            output = cnn(b_x)[0]  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
     
            '''if step % 50 == 0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = sum(pred_y == test_y) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)'''
            
    torch.save(cnn, modelpath + 'CNN_model.pkl')  # 保存整个网络
    torch.save(cnn.state_dict(), modelpath + 'CNN_model_params.pkl')  # 只保存网络中的参数 (速度快, 占内存少)


if __name__ == '__main__':
    train()
    
