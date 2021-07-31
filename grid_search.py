'''
Author: Xiang Pan
Date: 2021-07-29 21:18:11
LastEditTime: 2021-07-30 22:09:12
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/grid_search.py
xiangpan@nyu.edu
'''
import wandb
from milvus import IndexType
from env import *
# sweep_config = {"name" : "grid-search",
#                 "method" : "grid",
#                 "parameters" : 
#                 {
#                     "index_type" : 
#                     {
#                         "values" : [IndexType.IVF_FLAT]
#                     },
#                     "build_nlist" :
#                     {
#                         "min": 1,
#                         "max": 2048
#                     },
#                     "search_nprobe":
#                     {
#                         "min": 1,
#                         "max": 2048
#                     }
#                 }
# }


# # Set up your default hyperparameters
hyperparameter_defaults = dict(
    index_type      = IndexType.IVF_FLAT,
    build_nlist     = 2048,
    search_nprobe   = 1024,
)

# # Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults)
# Access all hyperparameter values through wandb.config
config = wandb.config

# # Set up your model
model = ENV()
# env = make_env(config)

# # Log metrics inside your training loop
# # for epoch in range(config["epochs"]):
# #     val_acc, val_loss = model.fit()
# #     metrics = {"validation_accuracy": val_acc,
# #                "validation_loss": val_loss}
# #     wandb.log(metrics)