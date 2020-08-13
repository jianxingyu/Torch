# -*- coding: utf-8 -*-
# @Time    : 2020/8/13 下午 2:38
# @Author  : jianxingyu
# @FileName: cifarnet.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    '''
    [b,28,28,1]---->[b,28,28,6]---->[b,14,14,6]---->[b,10,10,16]---->[b,5,5,16]---->[b,5*5*16]---->[b,120]---->[b,84]---->[b,10]
        原始           卷积            pooling            卷积          pooling          flatten       全连接      全连接      输出层
    '''

    def __init__(self):
        super(Net, self).__init__()
        print("进入一个类中")
        self.conv1 = nn.Conv2d(28 * 28, 6, 3)
        self.conv2 = nn.Conv2d(14 * 14 * 6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 去掉第一个维度
        nu = 1
        for i in size:
            nu *= i
        return i

