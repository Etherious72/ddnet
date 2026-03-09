# -*- coding: utf-8 -*-
"""
参数配置

Created on Feb 2023

@author: Xing-Yi Zhang (Zhangzxy20004182@163.com)

"""

####################################################
####                 主参数                    ####
####################################################

# 可选数据集: SEGSalt|SEGSimulation|FlatVelA|CurveFaultA|FlatFaultA|CurveVelA)
dataset_name = 'CurveFaultA'
learning_rate = 0.001                               # 学习率
classes = 1                                         # 输出通道数
display_step = 2                                    # 打印一次 loss 所需训练步数
model_type = 'DDNet70'
device_mode = 'cpu'                                # auto|cpu|gpu（训练/测试脚本通用）

####################################################
####               数据集参数                 ####
####################################################

####    如果数据集是.mat，size值为文件个数   ####
####    如果数据集是.npyt，size值为训练数据条数   ####

if dataset_name  == 'SEGSimulation':
    data_dim = [400, 301]                           # 单炮地震数据尺寸
    model_dim = [201, 301]                          # 单个速度模型尺寸
    inchannels = 29                                 # 输入通道数
    train_size = 800                                # 训练集样本数
    test_size = 100                                 # 测试集样本数

    firststage_epochs = 40
    secondstage_epochs = 30
    thirdstage_epochs = 30
    loss_weight = [1, 1e6]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 10                           # 每个训练 epoch 的 batch 大小
    test_batch_size = 2

elif dataset_name  == 'SEGSalt':
    data_dim = [400, 301]
    model_dim = [201, 301]
    inchannels = 29
    train_size = 30
    test_size = 10

    firststage_epochs = 0
    secondstage_epochs = 0
    thirdstage_epochs = 50                          # SEGSalt 用于迁移学习，不需要课程学习任务
    loss_weight = [1, 1e6]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 5
    test_batch_size = 2

elif dataset_name == 'FlatVelA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 24000
    test_size = 6000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 120
    loss_weight = [1, 0.01]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 64
    test_batch_size = 5

elif dataset_name == 'CurveVelA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 24000
    test_size = 6000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 100
    loss_weight = [1, 0.1]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 64
    test_batch_size = 5

elif dataset_name == 'FlatFaultA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    train_size = 48000
    test_size = 6000

    firststage_epochs = 10
    secondstage_epochs = 10
    thirdstage_epochs = 100
    loss_weight = [1, 0.01]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 64
    test_batch_size = 5

elif dataset_name == 'CurveFaultA':
    data_dim = [1000, 70]
    model_dim = [70, 70]
    inchannels = 5
    # train_size = 48000
    train_size = 3
    # test_size = 6000
    test_size = 1000

    firststage_epochs = 5
    secondstage_epochs = 5
    thirdstage_epochs = 5
    loss_weight = [1, 0.1]
    epochs = firststage_epochs + secondstage_epochs + thirdstage_epochs

    train_batch_size = 64
    test_batch_size = 5

else:
    print('The selected dataset is invalid')
    exit(0)
