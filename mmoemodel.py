# -*-coding: utf-8 -*-
# @Time    : 2024/7/12 17:28
# @File    : mmoemodel.py
# @Software: PyCharm

'''MMOE结构参考论文
Ma J, Zhao Z, Yi X, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts[C]
//Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018: 1930-1939.

https://www.zhihu.com/question/475434348
'''
import random
import tensorflow as tf
'''
MMOEModel重要参数
experts_num: 5
'''

class MMOEModel(tf.keras.Model):
    def __init__(self, task_num=4, share_feat_dim=200, feat_dim=100, output_dim=64, dim=32, class_num=1):
        super(MMOEModel, self).__init__()
        self.task_num = task_num
        self.class_num = class_num
        self.experts_num = self.task_num * 2
        self.output_dim = output_dim
        self.dim = dim
        # 共享层
        self.share_layer = tf.keras.layers.Dense(share_feat_dim)

        # 门控层
        self.gate_layers1 = tf.keras.layers.Dense(self.experts_num, use_bias=False)
        self.gate_layers2 = tf.keras.layers.Dense(self.experts_num, use_bias=False)
        self.gate_layers3 = tf.keras.layers.Dense(self.experts_num, use_bias=False)
        self.gate_layers4 = tf.keras.layers.Dense(self.experts_num, use_bias=False)

        # expert层
        self.expert_layer = [[] for _ in range(self.experts_num)]
        for id in range(self.experts_num):
            self.expert_layer[id].append(tf.keras.layers.Dense(feat_dim, kernel_regularizer=tf.keras.regularizers.l1(0.01)))
            self.expert_layer[id].append(tf.keras.layers.Dense(output_dim, kernel_regularizer=tf.keras.regularizers.l1(0.01)))

        # task tower层
        self.task_layer1 = tf.keras.layers.Dense(self.dim)
        self.task_layer2 = tf.keras.layers.Dense(self.dim)
        self.task_layer3 = tf.keras.layers.Dense(self.dim)
        self.task_layer4 = tf.keras.layers.Dense(self.dim)

        # task output tower层
        self.task_output_layer1 = tf.keras.layers.Dense(self.class_num)
        self.task_output_layer2 = tf.keras.layers.Dense(self.class_num)
        self.task_output_layer3 = tf.keras.layers.Dense(self.class_num)
        self.task_output_layer4 = tf.keras.layers.Dense(self.class_num)

    def call(self, feat):
        h = self.share_layer(feat)
        # 计算门控权值
        h_weight1 = self.gate_layers1(h)
        h_weight1 = tf.nn.softmax(h_weight1, axis=1)

        h_weight2 = self.gate_layers2(h)
        h_weight2 = tf.nn.softmax(h_weight2, axis=1)

        h_weight3 = self.gate_layers3(h)
        h_weight3 = tf.nn.softmax(h_weight3, axis=1)

        h_weight4 = self.gate_layers4(h)
        h_weight4 = tf.nn.softmax(h_weight4, axis=1)


        expert_output = []
        # 计算专家输出值
        for id in range(self.experts_num):
            h_data = self.expert_layer[id][0](h)
            h_data = tf.nn.leaky_relu(h_data)
            h_data = self.expert_layer[id][1](h_data)
            h_data = tf.nn.leaky_relu(h_data)
            expert_output.append(h_data)

        # 融合专家值
        # 任务1
        h = 0
        for i_exp in range(self.experts_num):
            h = h + tf.reshape(h_weight1[:, i_exp], (-1, 1)) * expert_output[i_exp]
        h = self.task_layer1(h)
        h = tf.nn.leaky_relu(h)
        h = self.task_output_layer1(h)
        h = tf.nn.sigmoid(h)
        res = h

        # 任务2
        h = 0
        for i_exp in range(self.experts_num):
            h = h + tf.reshape(h_weight2[:, i_exp], (-1, 1)) * expert_output[i_exp]
        h = self.task_layer2(h)
        h = tf.nn.leaky_relu(h)
        h = self.task_output_layer2(h)
        h = tf.nn.sigmoid(h)
        res = tf.concat([res, h], axis=1)

        # 任务3
        h = 0
        for i_exp in range(self.experts_num):
            h = h + tf.reshape(h_weight3[:, i_exp], (-1, 1)) * expert_output[i_exp]
        h = self.task_layer3(h)
        h = tf.nn.leaky_relu(h)
        h = self.task_output_layer3(h)
        h = tf.nn.sigmoid(h)
        res = tf.concat([res, h], axis=1)

        # 任务4,该任务为回规预测任务
        h = 0
        for i_exp in range(self.experts_num):
            h = h + tf.reshape(h_weight4[:, i_exp], (-1, 1)) * expert_output[i_exp]
        h = self.task_layer4(h)
        h = tf.nn.leaky_relu(h)
        h = self.task_output_layer4(h)
        h = tf.nn.relu(h)
        res = tf.concat([res, h], axis=1)
        return res
