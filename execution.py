# -*-coding: utf-8 -*-
# @Time    : 2024/7/19 11:26
# @File    : execution.py
# @Software: PyCharm

import os
os.system("pip install \"dask[dataframe]\"")
import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import mmoemodel
import dask.dataframe as dd


def train_step(model, args, epoch):
    # 读取数据
    path = args.data_input.split(',')[0]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    count = 0
    print("开始读取数据! {}".format(datetime.datetime.now()))
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    n = len(input_files)
    loss_all = 0
    loss = 0
    auc_metric = tf.metrics.AUC()
    for file in input_files:
        count += 1
        print("epoch:{}一共{}个文件,当前正在处理第{}个文件,文件路径:{}......".format(epoch, len(input_files), count, os.path.join(path, file)))
        # 读取训练数据
        # data = pd.read_csv(os.path.join(path, file), sep=',', header=None).astype('float32')
        data = dd.read_csv(os.path.join(path, file), sep=',', header=None).astype('float32')
        data = data.compute()  # 将 Dask DataFrame 转换为 Pandas DataFrame
        label = tf.convert_to_tensor(data.iloc[:, args.split_dim::], dtype=tf.float32)
        data = tf.convert_to_tensor(data.iloc[:, 0:args.split_dim], dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((data, label)).shuffle(buffer_size=10000).batch(args.batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        count_batch = 0
        for batch_data, batch_label in dataset:  # 取一个 batch 进行检查
            count_batch += 1
            with tf.GradientTape() as tape:
                res = model(batch_data, training=True)
                loss = loss_function(res, batch_label, args)
            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            if count_batch % args.batch_during == 0:
                print("第{}个epoch第{}文件第{}个batch的loss:{} {}".format(epoch, count, count_batch, loss, datetime.datetime.now()))
                auc_metric.update_state(batch_label[:, 2], res[:, 2])
                print("此时回流测试AUC为:{}".format(auc_metric.result().numpy()))
                auc_metric.reset_states()
                auc_metric.update_state(batch_label[:, 0], res[:, 0])
                print("此时被邀请测试AUC为:{}".format(auc_metric.result().numpy()))
                auc_metric.reset_states()
                auc_metric.update_state(batch_label[:, 1], res[:, 1])
                print("此时次日留存测试AUC为:{}".format(auc_metric.result().numpy()))
                auc_metric.reset_states()
        loss_all = loss_all + loss.numpy() / n
    return model, loss_all


def loss_function(res, label, args):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # 任务1的loss
    loss1 = bce(label[:, 0], res[:, 0])

    # 任务2的loss
    loss2 = bce(label[:, 1], res[:, 1])

    # 任务3的loss
    loss3 = bce(label[:, 2], res[:, 2])

    # 回归任务的loss
    if args.regression_loss == "Log-Cosh Loss":
        log_cosh_per_sample = tf.reduce_mean(tf.math.log(tf.cosh(res[:, -1] - label[:, -1])))
    else:
        mae_loss = tf.keras.losses.MeanAbsoluteError()
        log_cosh_per_sample = mae_loss(res[:, -1], label[:, -1])

    loss = args.task1_weight * loss1 + args.task2_weight * loss2 + args.task3_weight * loss3 + args.task4_weight * log_cosh_per_sample

    return loss


def train(args):
    # 定义模型
    if args.env == "train":
        model = mmoemodel.MMOEModel(task_num=args.task_num, share_feat_dim=args.share_feat_dim, feat_dim=args.feat_dim, output_dim=args.output_dim, dim=args.dim, class_num=args.class_num)
    else:
        # 装载训练好的模型
        cmd = "s3cmd get -r  " + args.model_output + "mmoemodel"
        os.system(cmd)
        model = tf.keras.models.load_model("./mmoemodel", custom_objects={'tf': tf}, compile=False)
        print("mmmoemodel is loaded!")
        # 读取数据
    beforeLoss = 2 ** 23
    stop_num = 0
    for epoch in range(args.epoch):
        model, loss = train_step(model, args, epoch)
        if beforeLoss > loss:
            beforeLoss = loss
            stop_num = 0
            # 保存model
            # model.summary()
            model.save("./mmoemodel", save_format="tf")
            cmd = "s3cmd put -r ./mmoemodel " + args.model_output
            os.system(cmd)
            print("epoch:{} mmoemodel模型已保存! {}".format(epoch, datetime.datetime.now()))
        else:
            stop_num += 1
            if stop_num > args.stop_num:
                print("epoch:{} Early stop! {}".format(epoch, datetime.datetime.now()))
                model.save("./mmoemodel", save_format="tf")
                cmd = "s3cmd put -r ./mmoemodel " + args.model_output
                os.system(cmd)
                print("epoch:{} mmoemodel模型已保存! {}".format(epoch, datetime.datetime.now()))
                break


# 执行推理过程
def inference(args):
    # 读取训练好的模型
    cmd = "s3cmd get -r  " + args.model_output + "mmoemodel"
    os.system(cmd)
    model = tf.keras.models.load_model("./mmoemodel", custom_objects={'tf': tf}, compile=False)
    print("mmoemodel is loaded!")
    # 读取数据
    path = args.data_input.split(',')[0]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    count = 0
    for file in input_files:
        count += 1
        print("一共{}个文件,当前正在处理第{}个文件,文件路径:{}......".format(len(input_files), count, os.path.join(path, file)))
        # 读取训练数据
        # data的第1-args.ID_dim列为ID,第args.ID_dim列之后为特征
        data = pd.read_csv(os.path.join(path, file), sep=',', header=None).astype(str)
        N = data.shape[0]
        result = np.zeros((N, args.task_num + args.ID_dim)).astype(str)
        result[:, 0:args.ID_dim] = data.iloc[:, 0:args.ID_dim].values
        data = tf.convert_to_tensor(data.iloc[:, args.ID_dim::], dtype=tf.float32)
        pred = model(data)
        result[:, args.ID_dim::] = pred.numpy().astype(str)
        # 写入结果
        n = data.shape[0]  # 获取行数
        output_file = os.path.join(args.data_output, 'pred_{}.csv'.format(count))

        # 使用 numpy.savetxt 写入 CSV 文件
        with open(output_file, mode="a") as resultfile:
            # 写入数据
            np.savetxt(resultfile, result, delimiter=',', fmt='%s')  # 使用 %s 以支持字符串和数字
        print("第{}个数据文件已经写入完成,写入数据的行数{} {}".format(count, n, datetime.datetime.now()))
        # write(result.tolist(), count, args)


# 写数据到文件系统
def write(data, count, args):
    # data是一个二维列表,第一列为ID,之后的列为特征
    n = len(data)
    with open(os.path.join(args.data_output, 'pred_{}.csv'.format(count)), mode="a") as resultfile:
        # 说明此时的data是[[],[],...]的二级list形式
        if n > 1:
            line = ""
            for j in range(n):
                line = line + ",".join(map(str, data[j])) + "\n"
        elif n == 1:
            line = ",".join(map(str, data[0])) + "\n"
        else:
            data = []
            line = ",".join(map(str, data)) + "\n"
        resultfile.write(line)
    print("第{}个数据文件已经写入完成,写入数据的行数{} {}".format(count, n, datetime.datetime.now()))
