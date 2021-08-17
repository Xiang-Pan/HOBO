'''
Author: Xiang Pan
Date: 2021-08-16 20:19:02
LastEditTime: 2021-08-16 20:57:23
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/build_siftsmall.py
xiangpan@nyu.edu
'''
import os
import time
#  
import numpy as np
import random
from milvus import Milvus,MetricType, IndexType
import torch
SERVER_ADDR = '127.0.0.1'
SERVER_PORT = '19530'
client = Milvus(host=SERVER_ADDR, port=SERVER_PORT)

collection_name = 'siftsmall_64'
client.drop_collection(collection_name)
collection_param = {
    "collection_name": collection_name,
    "dimension": 64,
    "metric_type": MetricType.L2
}
client.create_collection(collection_param)
print(client.list_collections())

path = "./cached_datasets/"+collection_name+"/"
siftsmall_base = np.load(path + "siftsmall_64_base.npy")
print(siftsmall_base)
entity_ids = [i for i in range(len(siftsmall_base))]
entities = siftsmall_base
status, ids = client.insert(collection_name, entities, entity_ids)
client.create_index(collection_name, IndexType.IVF_FLAT, {"nlist": 100})