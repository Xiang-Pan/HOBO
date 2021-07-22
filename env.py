'''
Author: Xiang Pan
Date: 2021-07-09 23:56:14
LastEditTime: 2021-07-22 18:37:53
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/env.py
xiangpan@nyu.edu
'''
import os
import time
import numpy as np
import random
import bohb.configspace as cs
import json
from milvus import Milvus, MetricType, IndexType


class IVF_FLAT_default_build_config(object):
    def __init__(self):
        self.nlist = 2048


class IVF_FLAT_build_config(object):
    nlist =  cs.IntegerUniformHyperparameter('nlist', 0, 65535)
    configspace = cs.ConfigurationSpace([nlist], seed=123)

    # nprobe = 16
class IVF_FLAT_search_config(object):
    nprobe =  cs.IntegerUniformHyperparameter('nprobe', 0, 2048)
    configspace = cs.ConfigurationSpace([nprobe], seed=123)
    

class IVF_PQ_default_build_config(object):
    def __init__(self):
        self.nlist  = 2048
        self.M = 16
    # nbits = 8

class IVF_PQ_build_config(object):
    nlist =  cs.IntegerUniformHyperparameter('nlist', 0, 65535)
    M =  cs.IntegerUniformHyperparameter('M', 0, 16)
    configspace = cs.ConfigurationSpace([nlist, M], seed=123)

class IVF_PQ_search_config(object):
    nprobe =  cs.IntegerUniformHyperparameter('nprobe', 0, 20)
    configspace = cs.ConfigurationSpace([nprobe], seed=123)




class HNSW_default_build_config(object):
    M = 24
    efConstruction = 128
    # ef = 64

class HNSW_build_config(object):
    M =  cs.IntegerUniformHyperparameter('M', 4, 64)
    efConstruction =  cs.IntegerUniformHyperparameter('efConstruction', 8, 512)
    configspace = cs.ConfigurationSpace([M, efConstruction], seed=123)

class HNSW_search_config(object):
    top_k = 64
    nprobe =  cs.IntegerUniformHyperparameter('nprobe', 0, 32768)
    configspace = cs.ConfigurationSpace([nprobe], seed=123)

class IVF_SQ8_default_build_config(object):
    nlist = 2048
    # QuantizerType = 8
    nprobe = 16

class IVF_SQ8_build_config(object):
    nlist =  cs.IntegerUniformHyperparameter('nlist', 1, 65535)
    configspace = cs.ConfigurationSpace([nlist], seed=123)
    

class IVF_SQ8_search_config(object):
    nprobe =  cs.IntegerUniformHyperparameter('nprobe', 0, 32768)
    configspace = cs.ConfigurationSpace([nprobe], seed=123)
    


def get_index_type(index_type):
    if index_type == "IVF_FLAT":
        return IndexType.IVF_FLAT
    if index_type == "IVF_PQ":
        return IndexType.IVF_PQ 
    if index_type == "IVF_SQ8":
        return IndexType.IVF_SQ8
    if index_type == "HNSW":
        return IndexType.HNSW

def get_default_build_config(index_type):
    if index_type == "IVF_FLAT":
        return IVF_FLAT_default_build_config().__dict__
    if index_type == "IVF_PQ":
        return IVF_PQ_default_build_config().__dict__
    if index_type == "IVF_SQ8":
        return IVF_SQ8_default_build_config().__dict__
    if index_type == "HNSW":
        return HNSW_default_build_config().__dict__

def get_search_configspace(index_type):
    if index_type == "IVF_FLAT":
        return IVF_FLAT_search_config().configspace
    if index_type == "IVF_PQ":
        return IVF_PQ_search_config().configspace
    if index_type == "IVF_SQ8":
        return IVF_SQ8_search_config().configspace
    if index_type == "HNSW":
        return HNSW_search_config().configspace

def get_default_build_configspace(index_type):
    if index_type == "IVF_FLAT":
        return IVF_FLAT_default_build_config().configspace
    if index_type == "IVF_PQ":
        return IVF_PQ_default_build_config().configspace
    if index_type == "IVF_SQ8":
        return IVF_SQ8_default_build_config().configspace
    if index_type == "HNSW":
        return HNSW_default_build_config().configspace


def get_build_configspace(index_type):
    if index_type == "IVF_FLAT":
        return IVF_FLAT_build_config().configspace
    if index_type == "IVF_PQ":
        return IVF_PQ_build_config().configspace
    if index_type == "IVF_SQ8":
        return IVF_SQ8_build_config().configspace
    if index_type == "HNSW":
        return HNSW_build_config().configspace

    

class ENV():
    def __init__(self, args):
        print("ENV")
        host = '127.0.0.1'
        port = '19530'
        self.client = Milvus(host, port)
        self.collection_name = 'siftsmall'
        self.query_groundtruth = self.get_groundtruth()
        self.query_vectors = self.get_query()
        self.top_k = 100
        self.index_type = get_index_type(args.index_type)
        print(args.index_type)
        self.default_build_config = get_default_build_config(args.index_type)
        self.search_configspace = get_search_configspace(args.index_type)
        self.build_configspace = get_build_configspace(args.index_type)
        

    def build_default_index(self):
        build_info = self.client.create_index(self.collection_name, self.index_type, self.default_build_config)
        print(build_info)


    def get_groundtruth(self):
        groundtruth = np.load("./cached_datasets/siftsmall_numpy/siftsmall_groundtruth.npy") #! NEED modified to name-based
        return groundtruth

    def get_query(self):
        siftsmall_query = np.load("./cached_datasets/siftsmall_numpy/siftsmall_query.npy")
        query_vectors = siftsmall_query
        return query_vectors

    def get_avg_recall(self, preds, targets):
        recall_total = 0
        for i in range(len(targets)):
            recall_single = len(set(preds[i]) & set(targets[i]))
            recall_total += recall_single
        avg_recall = recall_total/len(preds)
        return avg_recall
    
    def env_build_input(self, params):
        build_info = self.client.create_index(self.collection_name, self.index_type, params)
        # print(build_info)

    def env_search_input(self, params):
        start_time = time.time()
        statue, res = self.client.search(self.collection_name, top_k = self.top_k, query_records = self.query_vectors, params = params)
        end_time = time.time()
        query_time = (end_time - start_time)
        query_num = self.query_vectors.shape[0]
        # print(query_time)
        # print()
        query_per_sec = query_num/query_time
        # query_per_sec = 0
        # print(statue)
        if len(res) == 0:
            return 0, query_per_sec
        else:
            converted_res = np.zeros(res.shape)
            for i in range(len(res)):
                for j in range(len(res[i])):
                    converted_res[i][j] = res[i][j].id
            return self.get_avg_recall(converted_res, self.query_groundtruth), query_per_sec

        