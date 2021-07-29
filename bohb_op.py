'''
Author: Xiang Pan
Date: 2021-07-09 23:55:21
LastEditTime: 2021-07-29 17:53:06
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/bohb_op.py
xiangpan@nyu.edu
'''

import wandb
from bohb import BOHB
import bohb.configspace as cs
from option import *
from env import *
from milvus import Milvus, MetricType, IndexType
import sys
import pandas as pd

args = get_option()
table_dict = dict()
# df =

# def objective(step, alpha, beta):
#     return 1 / (alpha * step + 0.1) + beta


# def evaluate(params, n_iterations):
#     loss = 0.0
#     for i in range(int(n_iterations)):
#         loss += objective(**params, step=i)
#     return loss/n_iterations

def sign(x, threshold):
    if x > threshold:
        return x - threshold
    else:
        return 100000 * (threshold - x)


min_loss = 9999999999999
min_loss_recall = -1
min_loss_query_per_sec = -1
threshold = 90


gCurIndexType = 0
gCurIndexParam = None
gCurSearchParam = None



def build_type_evaluate(params, n_iterations):
    env.set_build_index_type(params['index_type'])
    
    global gCurIndexType
    gCurIndexType = params # enum type

    opt = BOHB(env.get_build_configspace(), build_evaluate, max_budget = n_iterations, min_budget=1)
    logs = opt.optimize()

    return logs.best['loss']

def build_evaluate(params, n_iterations):
    env.env_build_input(params = params)

    global gCurIndexParam
    gCurIndexParam = params
    
    search_opt = BOHB(env.search_configspace, search_evaluate, max_budget=n_iterations, min_budget=1)
    logs = search_opt.optimize()
    return logs.best['loss']


# def build_default_log_data():



def search_evaluate(params, n_iterations):
    recall, query_per_sec = env.env_search_input(params = params)

    if args.wandb_log:
        wandb.log({"recall": recall})
        wandb.log({"query_per_sec": query_per_sec})

        # index
        wandb.log({"gCurIndexType": gCurIndexType['index_type'].value})
        # global gTestTable
 
        # build
        global gCurIndexParam 
        for k,v in gCurIndexParam.items():
            wandb.log({k: v}) 

        # search
        for k,v in params.items():
            wandb.log({k: v})
        
        data = list(gCurIndexParam.values()) + list(params.values()) + [recall, query_per_sec]
        print(data)
    
        global table_dict

        if gCurIndexType['index_type'] not in table_dict.keys():
            print("create_index")
            cols = list(gCurIndexParam.keys()) + list(params.keys()) + ["recall", "query_per_sec"]
            table_dict[gCurIndexType['index_type']] = wandb.Table(columns =cols)
            table_dict[gCurIndexType['index_type']].add_data(*data)
        else:
            table_dict[gCurIndexType['index_type']].add_data(*data)
        # wandb.log({str(gCurIndexType['index_type']) : table_dict[gCurIndexType['index_type']]})

        
    # threshold = 95 
    loss = sign(recall, threshold) 
    global min_loss
    global min_loss_recall
    global min_loss_query_per_sec
    # if loss < min_loss:
    #     min_loss = loss
    #     min_loss_recall = recall
    #     min_loss_query_per_sec = query_per_sec
    print(params, recall, query_per_sec, loss)
    return loss


def get_exp_name(args):
    if args.op == "build_type":
        name = "build_type_threshold_" + str(args.threshold)
    if args.op == "build_index":
        name = "build_"+args.index_type + "_threshold_" + str(args.threshold)
    if args.op == "search":
        name = "search_"+ args.index_type + "_threshold_" + str(args.threshold)
    return name 

if __name__ == '__main__':
    env = ENV(args = args)
    threshold = args.threshold
    if args.wandb_log:
        name = get_exp_name(args)
        run = wandb.init()
        wandb.run.name = name
        
        # global gTestTable
        # gTestTable = wandb.Table(columns = ["index_type", "query_per_sec", "accuracy"])
        
    if args.op == "build_type":
        index_type = cs.CategoricalHyperparameter('index_type',[IndexType.IVF_FLAT, IndexType.IVF_PQ, IndexType.IVF_SQ8, IndexType.HNSW])
        index_type_configspace = cs.ConfigurationSpace([index_type], seed=123)
        type_opt = BOHB(index_type_configspace, build_type_evaluate, max_budget=5, min_budget=1)
        type_logs = type_opt.optimize()
        print(type_logs)


        # env.set_build_index_type(type_logs.best['hyperparameter'].to_dict())

        # search_opt = BOHB(env.search_configspace, search_evaluate, max_budget=10, min_budget=1)
        # search_logs = search_opt.optimize()
        # recall, query_per_sec = env.env_search_input(search_logs.best['hyperparameter'].to_dict())
        # print(recall, query_per_sec)

    elif args.op == "build_param":
        opt = BOHB(env.build_configspace, build_evaluate, max_budget=10, min_budget=1)
        logs = opt.optimize()
        print(logs)

        # reimplement best op
        env.env_build_input(logs.best['hyperparameter'].to_dict())
        search_opt = BOHB(env.search_configspace, search_evaluate, max_budget=10, min_budget=1)
        search_logs = search_opt.optimize()
        recall, query_per_sec = env.env_search_input(search_logs.best['hyperparameter'].to_dict())
        print(recall, query_per_sec)
        
    else:
        env.build_default_index()
        opt = BOHB(env.search_configspace, search_evaluate, max_budget=10, min_budget=1)
        logs = opt.optimize()
        print(logs)
        # reimplement best op
        recall, query_per_sec = env.env_search_input(logs.best['hyperparameter'].to_dict())
        print(recall,query_per_sec)
    # run.log({"gTestTable": gTestTable})
    print(table_dict)
    for k,v in table_dict.items():
        run.log({str(k): v})