'''
Author: Xiang Pan
Date: 2021-07-09 23:55:21
LastEditTime: 2021-07-23 22:35:45
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/bohb_op.py
xiangpan@nyu.edu
'''
import wandb
from bohb import BOHB
import bohb.configspace as cs
from option import *
from env import ENV
import sys

args = get_option()



# def objective(step, alpha, beta):
#     return 1 / (alpha * step + 0.1) + beta
    # loss = 0.0
    # for i in range(int(n_iterations)):
    #     loss += objective(**params, step=i)
    # return loss/n_iterations

def sign(x, threshold):
    if x > threshold:
        return x - threshold
    else:
        return 100000 * (threshold - x)


min_loss = 9999999999999
min_loss_recall = -1
min_loss_query_per_sec = -1
threshold = 90


def build_evaluate(params, n_iterations):
    env.env_build_input(params = params)
    search_opt = BOHB(env.search_configspace, search_evaluate, max_budget=n_iterations, min_budget=1)
    logs = search_opt.optimize()
    # print(logs)
    return logs.best['loss']





def search_evaluate(params, n_iterations):
    recall, query_per_sec = env.env_search_input(params = params)

    if args.wandb_log:
        wandb.log({"recall": recall})
        wandb.log({"query_per_sec": query_per_sec})
        for k,v in params.items():
            wandb.log({k: v})
        
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

if __name__ == '__main__':

    
    env = ENV(args = args)
    threshold = args.threshold
    if args.wandb_log:
        wandb.init()
    if args.op == "build":
        opt = BOHB(env.build_configspace, build_evaluate, max_budget=10, min_budget=1)
        logs = opt.optimize()
        print(logs)

        # reimplement best op
        env.env_build_input(logs.best['hyperparameter'].to_dict())
        search_opt = BOHB(env.search_configspace, search_evaluate, max_budget=10, min_budget=1)
        search_logs = search_opt.optimize()
        recall, query_per_sec = env.env_search_input(search_logs.best['hyperparameter'].to_dict())
        print(recall,query_per_sec)
        
    else:
        env.build_default_index()
        opt = BOHB(env.search_configspace, search_evaluate, max_budget=10, min_budget=1)
        logs = opt.optimize()
        print(logs)
        # reimplement best op
        recall, query_per_sec = env.env_search_input(logs.best['hyperparameter'].to_dict())
        print(recall,query_per_sec)