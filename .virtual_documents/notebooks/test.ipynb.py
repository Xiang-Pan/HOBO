import os
import time
#  
import numpy as np
import random
from milvus import Milvus,MetricType, IndexType
SERVER_ADDR = '127.0.0.1'
SERVER_PORT = '19530'


client = Milvus(host=SERVER_ADDR, port=SERVER_PORT)
# create_milvus_collection(milvus)


#  ➜ sudo docker run -d --name milvus_cpu_1.1.0 \
# -p 19530:19530 \
# -p 19121:19121 \
# -v /home/$USER/milvus/db:/var/lib/milvus/db \
# -v /home/$USER/milvus/conf:/var/lib/milvus/conf \
# -v /home/$USER/milvus/logs:/var/lib/milvus/logs \
# -v /home/$USER/milvus/wal:/var/lib/milvus/wal \
# milvusdb/milvus:1.1.0-cpu-d050721-5e559c


client.list_collections()


collection_name = 'demo_film_tutorial'
collection_param = {
    "collection_name": collection_name,
    "dimension": 8,
    "index_file_size": 2048,
    "metric_type": MetricType.L2
}
client.create_collection(collection_param)


client.list_collections()


client.list_partitions(collection_name)


client.create_partition(collection_name, "films")


import random


film_A = [random.random() for _ in range(8)]


film_A


entity_ids = [0, 1, 2]
entities = [[random.random() for _ in range(8)] for _ in range(3)]
status, ids = client.insert(collection_name, entities, entity_ids)


status, results = client.search(collection_name, 2, [film_A])


results


ids


client.create_index(collection_name, IndexType.FLAT)


query_vectors = film_A


query_vectors = [[1,2,3,4,5,6,7,8]]


# FLAT
client.search(collection_name,1,query_vectors)


status, info = client.get_collection_info(collection_name)


info


status, results = client.search(collection_name, 2, [film_A])


status


results


from bohb import *


from bohb import BOHB
import bohb.configspace as cs


def objective(step, alpha, beta):
    return 1 / (alpha * step + 0.1) + beta


def evaluate(params, n_iterations):
    loss = 0.0
    for i in range(int(n_iterations)):
        loss += objective(**params, step=i)
    return loss/n_iterations


alpha = cs.CategoricalHyperparameter('alpha', [0.001, 0.01, 0.1])
beta = cs.CategoricalHyperparameter('beta', [1, 2, 3])
configspace = cs.ConfigurationSpace([alpha, beta], seed=123)

opt = BOHB(configspace, evaluate, max_budget=10, min_budget=1)
logs = opt.optimize()
print(logs)



get_ipython().getoutput("pwd")





import numpy as np
fv = numpy.fromfile("./cached_datasets/siftsmall/siftsmall_base.fvecs", dtype="int32")
dim = fv.view(numpy.int32)[0]
new = fv.reshape(-1, dim + 1)[:,1:] # reshapes file
f_new = new.view(numpy.float32)
np.save("./cached_datasets/siftsmall_numpy/siftsmall_base",f_new)


f_new


import numpy as np
fv = numpy.fromfile("./cached_datasets/siftsmall/siftsmall_groundtruth.ivecs", dtype="int32")
# dim = fv.view(numpy.int32)[0]
dim = 100
f_new = fv.reshape(-1, dim + 1)[:,1:] # reshapes file
# f_new = new.view(numpy.float32)
np.save("./cached_datasets/siftsmall_numpy/siftsmall_groundtruth",f_new)


fv = numpy.fromfile("./cached_datasets/siftsmall/siftsmall_groundtruth.ivecs", dtype="int32")


new


import numpy as np
fv = numpy.fromfile("./cached_datasets/siftsmall/siftsmall_learn.fvecs", dtype="int32")
dim = fv.view(numpy.int32)[0]
new = fv.reshape(-1, dim + 1)[:,1:] # reshapes file
f_new = new.view(numpy.float32)
np.save("./cached_datasets/siftsmall_numpy/siftsmall_learn",f_new)


import numpy as np
fv = numpy.fromfile("./cached_datasets/siftsmall/siftsmall_query.fvecs", dtype="int32")
dim = fv.view(numpy.int32)[0]
new = fv.reshape(-1, dim + 1)[:,1:] # reshapes file
f_new = new.view(numpy.float32)
np.save("./cached_datasets/siftsmall_numpy/siftsmall_query",f_new)


f_new


docker


import os
import time
#  
import numpy as np
import random
from milvus import Milvus,MetricType, IndexType
SERVER_ADDR = '127.0.0.1'
SERVER_PORT = '19530'

host = '127.0.0.1'
port = '19530'
client = Milvus(host, port)


collection_name = 'siftsmall'
collection_param = {
    "collection_name": collection_name,
    "dimension": 128,
#     "index_file_size": 2048,
    "metric_type": MetricType.L2
}
client.create_collection(collection_param)


client.list_collections()


siftsmall_base = np.load("./cached_datasets/siftsmall_numpy/siftsmall_base.npy")


siftsmall_base.shape


entity_ids = [i for i in range(len(siftsmall_base))]
entities = siftsmall_base
collection_name = "siftsmall"


status, ids = client.insert(collection_name, entities, entity_ids)


client.create_index(collection_name, IndexType.IVF_FLAT, {"nlist": 100})
# int. 1~65536


siftsmall_query = np.load("./cached_datasets/siftsmall_numpy/siftsmall_query.npy")
query_vectors = siftsmall_query


collection_name = 'siftsmall'
statue, res = client.search(collection_name, top_k = 200, query_records = query_vectors, params={"nprobe": 1 })
# int. 1~nlist(cpu), 1~min[2048, nlist](gpu)


res





converted_res = np.zeros(res.shape)
or i in range(len(res)):
    for j in range(len(res[i])):
#         print(i,j,res[i][j].id)
        converted_res[i][j] = res[i][j].id


siftsmall_groundtruth = np.load("./cached_datasets/siftsmall_numpy/siftsmall_groundtruth.npy")
siftsmall_groundtruth.shape
# query_vectors = siftsmall_query


converted_res.shape





from torchmetrics import RetrievalRecall
# import torch.tensor as tensor
indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
preds = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
target = torch.tensor([False, False, True, False, True, False, True])
r2 = RetrievalRecall(k=100)
r2(preds, target, indexes=indexes)



preds = converted_res


target = siftsmall_groundtruth


preds


class IVF_FLAT_build_config(object):
    nlist  = 2048


import json
a = IVF_FLAT_build_config()


json.dumps(a.__dict__)


a = IVF_PQ_build_config()
a.nlist


a


gDataDim = 512


a = [i for i in range(1,16) if 512get_ipython().run_line_magic("i", " == 0]")


a


import wandb
run = wandb.init()


gTestTable = wandb.Table(columns = ["index_type", "query_per_sec", "accuracy"])


d = {i:0 for i in gTestTable.columns}


parmas = dict()


parmas["index_type"] = 1


parmas["query_per_sec"] = 1


for k,v in parmas.items():
    d[k] = v


list(d.keys())


d["sss"] = 1


for k,v in d.items():
    if k not in gTestTable.columns:
        gTestTable.add_column(name = k, data = 0)


gTestTable.add_data([1,1,0,1])


gTestTable.columns


import wandb
run = wandb.init()
my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
# run.log({"table_key": my_table})


my_table = wandb.Table(columns=["a", "b"], data=[["1a", "1b"], ["2a", "2b"]])
my_table.add_data("c","d")
run.log({"table_key": my_table})


from milvus import *


collection_name = "siftsmall"
status, stats = client. get_index_info(collection_name)


stats._index_type


stats._params





from option import *

















hyperparameters['index_type'] = args.index_type
index_params = dict()
search_params = dict()
for b in build_keys:
    index_params[b.replace("build_","")] = vars(args)[b]
for s in search_keys:
    search_params[s.replace("search_","")] = vars(args)[s]

hyperparameters["index_params"] = index_params
hyperparameters["search_params"] = search_params


hyperparameters


import yaml


def dfs(a, l, depth):
    if len(a) == 1: # end condition
        for i in a[0]:
            l.append(i)
            print(l, depth)
            l.remove(i)
    else:
        for i in a[0]:
            print(i,depth)
            l.append(i)
            for j in a[1:]:
                dfs(a[1:],l, depth+1)
            l.remove(i)








dfs(params_list,[],0)


from utils import convert_config


config





a = [range(1,10),range(1,3)]


# def get_params(config):

name_list





params_list


eval("range(1,nlist,10)")


nlist = 20





def constuct_for_loop():
    name = 'ef'
    i = 'range(1,1024,10)'
    s = 'for ' + name + ' in '+ i
    print(s)
    
    


for name in config['parameters']['index_params']:
    r = config['parameters']['index_params'][name]['values']

    if type(r) == 'str':
        r = list(eval(r))
    



name = 'ef'
i = 'range(1,1024,10)'
s = 'for ' + name + ' in '+ i + ':\n print(1)'
eval(s)


iter('abc')


list(range(1,1024,10))





import pandas as pd


df = pd.read_csv("./grid_search_results/IVF_FLAT.csv")


df


data = df.to_numpy()


data.shape



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
 
 
# 数据
x = data[:, 1]  
y = data[:, 2]  
z = data[:, -1]  
 
 
# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
 
 
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
 
 
# 数据
 
# 数据１
data1 = np.arange(24).reshape((8, 3))
# data的值如下：
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]
#  [12 13 14]
#  [15 16 17]
#  [18 19 20]
#  [21 22 23]]
x1 = data1[:, 0]  # [ 0  3  6  9 12 15 18 21]
y1 = data1[:, 1]  # [ 1  4  7 10 13 16 19 22]
z1 = data1[:, 2]  # [ 2  5  8 11 14 17 20 23]
 
# 数据２
data2 = np.random.randint(0, 23, (6, 3))
x2 = data2[:, 0]
y2 = data2[:, 1]
z2 = data2[:, 2]
 
 
# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1, y1, z1, c='r', label='顺序点')
ax.scatter(x2, y2, z2, c='g', label='随机点')
 
 
# 绘制图例
ax.legend(loc='best')
 
 
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
 
 
# 展示
plt.show()


a = [[1,2],[3,4]]





configs = dfs(a,[],0)


for i in configs:
    print(i)


configs[0]


cd HOBO


import numpy as np


collection_name = 'sift'
collection_param = {
    "collection_name": collection_name,
    "dimension": 128,
#     "index_file_size": 2048,
    "metric_type": MetricType.L2
}
client.create_collection(collection_param)


client.drop_collection("sift")


client.list_collections()


mkdir ./cached_datasets/sift_numpy


import numpy as np
fv = np.fromfile("./cached_datasets/sift/sift_query.fvecs", dtype="int32")
dim = fv.view(np.int32)[0]
new = fv.reshape(-1, dim + 1)[:,1:] # reshapes file
f_new = new.view(np.float32)
np.save("./cached_datasets/sift_numpy/sift_query",f_new)


sift_base = np.load("./cached_datasets/sift_numpy/sift_base.npy")


siftsmall_base.shape


id_count = 0
def get_batch():
    entity_ids = [i for i in range(len(siftsmall_base))]
    entities = siftsmall_base
    collection_name = "sift"
    id_count += batch_size


import os
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.data = sift_base
        self.labels = [i for i in range(len(siftsmall_base))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x,y


from torch.utils.data import DataLoader

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)


for batch in dataloader:
    entities, entity_ids = batch
    status, ids = client.insert(collection_name, entities.numpy(), entity_ids.numpy().tolist())


for i in 





status


sift_groundtruth = np.load("./cached_datasets/sift_numpy/sift_groundtruth.npy")
sift_groundtruth.shape


query_vectors = np.load("./cached_datasets/sift_numpy/sift_query.npy")
query_vectors.shape


collection_name = 'sift'
statue, res = client.search(collection_name, top_k = 100, query_records = query_vectors, params={"nprobe": 512 })
# int. 1~nlist(cpu), 1~min[2048, nlist](gpu)


res


converted_res = np.zeros(res.shape)
for i in range(len(res)):
    for j in range(len(res[i])):
        converted_res[i][j] = res[i][j].id


converted_res


preds = converted_res


targets = sift_groundtruth


recall_total = 0
for i in range(len(targets)):
    recall_single = len(set(preds[i]) & set(targets[i]))
    recall_total += recall_single
avg_recall = recall_total/len(preds)
avg_recall


preds


cd ..


import numpy as np
import pandas as pd


df = pd.read_csv("./grid_search_results/HNSW.csv")


df


index = df['loss'].idxmin()


df.iloc[index,:]


class IVF_PQ_default_build_config(object):
    def __init__(self):
        self.nlist  = 2048
        self.M = 16 


IVF_PQ_default_build_config().__dict__



