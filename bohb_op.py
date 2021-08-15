'''
Author: Xiang Pan
Date: 2021-07-09 23:55:21
LastEditTime: 2021-08-14 20:18:12
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
from utils import *

args = get_option()
table_dict = dict()

min_loss = 9999999999999
min_loss_recall = -1
min_loss_query_per_sec = -1
threshold = 95


gCurIndexType = 0
gCurIndexParam = None
gCurSearchParam = None



def build_type_evaluate(params, n_iterations):

    env.target_index_type = params['index_type']
    # env.check_index()
    env.refresh_status()

    opt = BOHB(get_build_configspace(env.target_index_type), build_evaluate, max_budget = n_iterations, min_budget=1,  eta = 10)
    logs = opt.optimize()

    return logs.best['loss']

def build_evaluate(params, n_iterations):
    
    env.env_build_input(build_type = env.target_index_type, params = params)

    global gCurIndexParam
    gCurIndexParam = params
    
    search_opt = BOHB(env.search_configspace, search_evaluate, max_budget=n_iterations, min_budget=1)
    logs = search_opt.optimize()
    return logs.best['loss']


def search_evaluate(params, n_iterations):
    recall, query_per_sec = env.env_search_input(params = params)

    loss = sign(recall, threshold) - query_per_sec

    if args.wandb_log:
        wandb.log({"recall": recall})
        wandb.log({"query_per_sec": query_per_sec})

        # index
        env.refresh_status()
        wandb.log({"index_type_enum": int(env.index_type.value)})
 
        # build
        for k,v in env.index_params.items():
            wandb.log({k: v}) 

        # search
        for k,v in env.search_params.items():
            wandb.log({k: v})
        
        data = list(env.index_params.values()) + list(env.search_params.values()) + [recall, query_per_sec, loss]
    
        global table_dict

        if env.index_type not in table_dict.keys():
            print("create_index")
            cols = list(env.index_params.keys()) + list(env.search_params.keys()) + ["recall", "query_per_sec", "loss"] 
            table_dict[env.index_type] = wandb.Table(columns = cols)
        table_dict[env.index_type].add_data(*data)
        if 'best' not in table_dict.keys() or loss < table_dict['best'].get_column("loss")[0]:
            cols = ["index_type"] + list(env.index_params.keys()) + list(env.search_params.keys()) + ["recall", "query_per_sec", "loss"] 
            # data = list(env.index_params.values()) + list(env.search_params.values()) + [recall, query_per_sec, loss]
            table_dict['best'] = wandb.Table(columns = cols)
            data = [str(env.index_type)] + data 
            table_dict['best'].add_data(*data)

    return loss


def get_exp_name(args):
    if args.op == "build_type":
        name = "build_type_threshold_" + str(args.threshold)
    elif args.op == "build_params":
        name = "build_"+args.index_type + "_threshold_" + str(args.threshold)
    elif args.op == "search_params":
        name = "search_"+ args.index_type + "_threshold_" + str(args.threshold)
    else:
        print("op error!")
        name = "error_run"
    return name


if __name__ == '__main__':
    env = ENV(args)
    threshold = args.threshold
    if args.wandb_log:
        name = get_exp_name(args)
        run = wandb.init()
        wandb.run.name = name
    if args.op == "build_type":
        build_type_search_spcae = [IndexType.IVF_FLAT, IndexType.IVF_PQ, IndexType.IVF_SQ8, IndexType.HNSW]
        build_type_search_method = "NOT BO"
        if build_type_search_method == "BO":
            index_type = cs.CategoricalHyperparameter('index_type',[IndexType.IVF_FLAT, IndexType.IVF_PQ, IndexType.IVF_SQ8, IndexType.HNSW])
            index_type_configspace = cs.ConfigurationSpace([index_type], seed=123)
            type_opt = BOHB(index_type_configspace, build_type_evaluate, max_budget=10, min_budget=1)
            type_logs = type_opt.optimize()
            print(type_logs)
        else:
            print("a")
        
        for ndx, row in table_dict['best'].iterrows():
            print(row)

    elif args.op == "build_params":
        
        opt = BOHB(env.build_configspace, build_evaluate, max_budget=10, min_budget=1)
        logs = opt.optimize()
        # print(logs.best['hyperparameter'].to_dict())
        for ndx, row in table_dict['best'].iterrows():
            print(row)
        # env.env_build_input(env.index_type, logs.best['hyperparameter'].to_dict())
        # search_opt = BOHB(env.search_configspace, search_evaluate, max_budget=10, min_budget=1)
        # search_logs = search_opt.optimize()
        # recall, query_per_sec = env.env_search_input(search_logs.best['hyperparameter'].to_dict())
        # print(logs.best['hyperparameter'].to_dict(),search_logs.best['hyperparameter'].to_dict(), recall, query_per_sec)
        
    else:
        # serch parmas
        # env.set_target_index_type()
        # TODO: fix this by decoupling target and current
        env.index_type = get_index_type(args.index_type) # from str to enum
        env.build_default_index()
        opt = BOHB(env.search_configspace, search_evaluate, max_budget=5, min_budget=1)
        logs = opt.optimize()
        # print(logs)
        for ndx, row in table_dict['best'].iterrows():
            print(row)
        # print()

        # reimplement best op
        recall, query_per_sec = env.env_search_input(logs.best['hyperparameter'].to_dict())
        print(recall,query_per_sec)

    # print(table_dict)
    for k,v in table_dict.items():
        run.log({str(k): v})