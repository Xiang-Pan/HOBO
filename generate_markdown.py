'''
Author: Xiang Pan
Date: 2021-08-14 13:35:23
LastEditTime: 2021-08-14 22:22:45
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/generate_markdown.py
xiangpan@nyu.edu
'''
import wandb
import os
import json
import numpy as np
import pandas as pd
import sys

# get run name
def print_markdown_table(run_name):
    # run_name = 
    HOME_DIR = os.getcwd()

    # get artifacts
    api = wandb.Api()
    run = api.run("xiang-pan/HOBO/"+run_name)
    artifacts = run.logged_artifacts()

    # download artifacts
    file_name = []
    for i in artifacts:
        file_name.append(i.download())
        print("Downloading...",file_name[-1])

    # load artifacts
    file_name.sort(reverse=True)
    json_files = {}
    for d in file_name:
        # change dir
        artifact_dir = d.split('.')[-1]
        os.chdir(HOME_DIR+artifact_dir)

        # load json
        file_str = os.listdir()[0]
        file = open(file_str).read()
        json_files[file_str.split('.')[0]] = json.loads(file)

    # generate df
    os.chdir(HOME_DIR)
    pdlist = []
    for k, j in json_files.items():
        # check data required available
        if 'data' not in j.keys() or 'columns' not in j.keys():
            print('Error: fail to locate data for dataframe construction!')
            break
        data =  np.array(j['data'])
        col = j['columns']
        
        # construct dict
        dic = {}
        for i in range(len(col)):
            dic[col[i]] = data[:,i]
        # construct df
        df = pd.DataFrame.from_dict(dic)
        df_selected = df.sort_values('loss').head(1)
        if k != 'best':
            df_selected['index_type'] = k
        pdlist.append(df_selected)

    # output df result
    out_df = pd.concat(pdlist)
    out_df.loss = out_df.loss.astype('float')
    out_df = out_df.sort_values('loss')
    #print(out_df)

    # TODO: convert df to markdown
    print('\n')
    print(out_df.to_markdown())

if __name__ == '__main__':
    print_markdown_table(sys.argv[-1])