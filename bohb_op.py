'''
Author: Xiang Pan
Date: 2021-07-09 23:55:21
LastEditTime: 2021-09-25 18:51:58
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/bohb_op.py
xiangpan@nyu.edu
'''

from bohb import BOHB
import bohb.configspace as cs
from option import *
from env import *
from milvus import Milvus, MetricType, IndexType
import sys
import pandas as pd
from utils import *
from generate_markdown import *

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
    if args.build_search_share_space:
        opt = BOHB(get_build_search_shared_configspace(env.target_index_type), build_search_share_space_evaluate, max_budget = n_iterations, min_budget=1,  eta = 10)
        logs = opt.optimize()
    
    else:
        opt = BOHB(get_build_configspace(env.target_index_type), build_evaluate, max_budget = n_iterations, min_budget=1,  eta = 10)
        logs = opt.optimize()
    return logs.best['loss']

# def build_type_togethe
def build_search_share_space_evaluate(params, n_iterations):
    if 'nlist' in params.keys() and 'nprobe' in params.keys():
        if params['nlist'] <= params['nprobe']:
            return np.inf
    config = dict()
    config['index_type'] = env.target_index_type
    config['index_params'] = dict((key, value) for key, value in params.items() if key not in ['nprobe', 'ef'])
    config['search_params'] = dict((key, value) for key, value in params.items() if key in ['nprobe', 'ef'])
    recall , query_per_sec = env.config_input(config)
    loss = loss_opertation(recall, query_per_sec)
    return loss

def build_evaluate(params, n_iterations):
    
    env.env_build_input(build_type = env.target_index_type, params = params)

    global gCurIndexParam
    gCurIndexParam = params
    
    # the search configuration already refreshed
    search_opt = BOHB(env.search_configspace, search_evaluate, max_budget=n_iterations, min_budget=1)
    logs = search_opt.optimize()
    return logs.best['loss']

def loss_opertation(recall, query_per_sec):
    
    loss = sign(recall, threshold) - query_per_sec
    env.refresh_status()
    
    data = list(env.index_params.values()) + list(env.search_params.values()) + [recall, query_per_sec, loss]
    
    global table_dict

    if env.index_type not in table_dict.keys():
        cols = list(env.index_params.keys()) + list(env.search_params.keys()) + ["recall", "query_per_sec", "loss"] 
        table = pd.DataFrame([data], columns=cols)
        table_dict[env.index_type] = table
    else:
        table_dict[env.index_type].loc[len(table_dict[env.index_type].index)] = data
    
    if 'best' not in table_dict.keys():
        cols = ["index_type"] + list(env.index_params.keys()) + list(env.search_params.keys()) + ["recall", "query_per_sec", "loss"] 
        data = [str(env.index_type)] + data 
        table_dict['best'] = pd.DataFrame([data], columns=cols)  
        table_dict['best'].style.set_caption("best op")
    elif loss < table_dict['best']["loss"][0]:
        data = [str(env.index_type)] + data 
        table_dict['best'].loc[0] = data
        if args.wandb_log:
            wandb.log({"best_loss" :loss})

    if args.wandb_log:
        wandb.log({"recall": recall})
        wandb.log({"query_per_sec": query_per_sec})
        wandb.log({"loss": loss})

        # index
        wandb.log({"index_type_enum": int(env.index_type.value)})

        # build
        for k,v in env.index_params.items():
            wandb.log({k: v}) 

        # search
        for k,v in env.search_params.items():
            wandb.log({k: v})

    return loss

def search_evaluate(params, n_iterations):
    recall, query_per_sec = env.env_search_input(params = params)
    loss = loss_opertation(recall, query_per_sec)
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
    name = get_exp_name(args)
    if args.wandb_log:
        import wandb
        run = wandb.init(entity="milvus")
        wandb.run.name = name
    if args.op == "build_type":
        build_type_search_spcae = [IndexType.IVF_FLAT, IndexType.IVF_PQ, IndexType.IVF_SQ8, IndexType.HNSW]
        if args.build_type_op_method == "BO":  # BO
            index_type = cs.CategoricalHyperparameter('index_type', build_type_search_spcae)
            index_type_configspace = cs.ConfigurationSpace([index_type], seed=123)
            type_opt = BOHB(index_type_configspace, build_type_evaluate, max_budget=10, min_budget=1)
            type_logs = type_opt.optimize()
        else:                                 # Loop
            for index_type in build_type_search_spcae:
                env.target_index_type = index_type
                env.refresh_status()
                opt = BOHB(get_build_configspace(env.target_index_type), build_evaluate, max_budget=10, min_budget=1)
                logs = opt.optimize()

    elif args.op == "build_params":
        opt = BOHB(get_build_configspace(env.target_index_type), build_evaluate, max_budget=10, min_budget=1)
        logs = opt.optimize()
        
    else:
        # serch parmas
        env.index_type = get_index_type(args.index_type) # from str to enum
        env.build_default_index()
        opt = BOHB(env.search_configspace, search_evaluate, max_budget=5, min_budget=1)
        logs = opt.optimize()
    print(table_dict['best'])

    path = "./outputs/pandas_logs/" + name + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    for name, table in table_dict.items():
        table.to_csv(path+str(name)+".csv")
    if args.wandb_log:
        for k,v in table_dict.items():
            run.log({str(k): v})