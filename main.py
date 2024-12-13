# -*-coding: utf-8 -*-
# @Time    : 2024/7/19 11:13
# @File    : main.py
# @Software: PyCharm

import time
import argparse
import execution

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or inference)", type=str, default='train_incremental')
    parser.add_argument("--lr", help="学习率", type=float, default=0.00001)
    parser.add_argument("--stop_num", help="early stop机制的触发次数", type=int, default=30)
    parser.add_argument("--batch_size", help="batch的大小", type=int, default=2048)
    parser.add_argument("--batch_during", help="打印batch loss的周期", type=int, default=100)
    parser.add_argument("--epoch", help="迭代次数", type=int, default=200)
    parser.add_argument("--task_num", help="任务数", type=int, default=4)
    parser.add_argument("--share_feat_dim", help="共享层隐含层输出特征的维度", type=int, default=200)
    parser.add_argument("--feat_dim", help="隐含层输出特征的维度2", type=int, default=100)
    parser.add_argument("--output_dim", help="隐含层输出特征的维度3", type=int, default=64)
    parser.add_argument("--dim", help="输出特征的维度", type=int, default=32)
    parser.add_argument("--ID_dim", help="在inference阶段前几列为ID", type=int, default=2)
    parser.add_argument("--class_num", help="类别数目", type=int, default=1)
    parser.add_argument("--task1_weight", help="任务1(分类)loss权重", type=float, default=1.5)
    parser.add_argument("--task2_weight", help="任务2(分类)loss权重", type=float, default=1)
    parser.add_argument("--task3_weight", help="任务3(分类)loss权重", type=float, default=2)
    parser.add_argument("--task4_weight", help="任务4(回归)loss权重", type=float, default=0.01)
    parser.add_argument("--split_dim", help="输入数据的特征维度(不含类标签信息)", type=int, default=183)
    parser.add_argument("--regression_loss", help="regression任务的loss(MeanAbsoluteError或Log-Cosh Loss)", type=str, default='MeanAbsoluteError')
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置",
                        type=str, default='models/apgame/recall/mmoemodel/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    if args.env == "train" or args.env == "train_incremental":
        execution.train(args)
    elif args.env == "inference":
        execution.inference(args)
    else:
        raise TypeError("args.env必需是train或train_incremental或inference！")
    end_time = time.time()
    print("算法总共耗时:{}".format(end_time - start_time))