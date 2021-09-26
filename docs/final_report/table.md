<!--
 * @Author: Xiang Pan
 * @Date: 2021-09-25 20:59:19
 * @LastEditTime: 2021-09-25 21:10:00
 * @LastEditors: Xiang Pan
 * @Description: 
 * @FilePath: /HOBO/docs/final_repot/table.md
 * xiangpan@nyu.edu
-->
---
title: "HOBO Results"
<!-- author: Xiang Pan -->
date: Set 25, 2021
geometry: margin=2cm
output: table
---
# Results
## Index Type Optimization


| Method                                                                 | index_type | M  | efConstruction | ef  | recall | query_per_sec | loss     |
|------------------------------------------------------------------------|------------|----|----------------|-----|--------|---------------|----------|
| [BOHB(Index Type Loop)](https://wandb.ai/xiang-pan/HOBO/runs/14pnimgi) | 'HNSW'     | 17 | 445            | 114 | 99.68  | 16782.6       | -16777.9 |
| [BOHB(Index Type BO)](https://wandb.ai/xiang-pan/HOBO/runs/13jdknsy)   | 'HNSW'     | 22 | 274            | 106 | 99.75  | 18522         | -18517.2 |

|                                                              | index_type | nlist | M   | nprobe | recall | query_per_sec | loss          |
|--------------------------------------------------------------|------------|-------|-----|--------|--------|---------------|---------------|
| [Grid Search](https://wandb.ai/xiang-pan/HOBO/runs/3vdvm6gs) | 'HNSW'     | 4     | 158 | 200    | 97.11  | 18331.74  | -18329.63 |



## Index Parameters Optimization


### IVF_FLAT

|                                                              | index_type | nlist | nprobe | recall | query_per_sec | loss          |
|--------------------------------------------------------------|------------|-------|--------|--------|---------------|---------------|
| [BOHB](https://wandb.ai/xiang-pan/HOBO/runs/9ughlu3l)        | 'IVF_FLAT' | 2883  | 54     | 99.68  | 14911         | -14906.3      |
| [Grid Search](https://wandb.ai/xiang-pan/HOBO/runs/22n2lk07) | 'IVF_FLAT' | 14601 | 101    | 100.0  | 14402.03  | -14397.03 |

### IVF_SQ8
|                                                              | index_type | nlist | nprobe | recall | query_per_sec | loss         |
|--------------------------------------------------------------|------------|-------|--------|--------|---------------|--------------|
| [BOHB](https://wandb.ai/xiang-pan/HOBO/runs/2hnt39qn)        | 'IVF_SQ8'  | 8405  | 46     | 98.86  | 13827.5       | -13823.7     |
| [Grid Search](https://wandb.ai/xiang-pan/HOBO/runs/26z6cea5) | 'IVF_SQ8'  | 5401  | 101    | 99.49  | 13080.62   | -13076.13 |

### IVF_PQ
|                                                                                                             | index_type | m   | nlist | nprobe | recall | query_per_sec      | loss                |
|-------------------------------------------------------------------------------------------------------------|------------|-----|-------|--------|--------|--------------------|---------------------|
| [BOHB](https://wandb.ai/xiang-pan/HOBO/runs/2hh95hjr)                                                       | 'IVF_PQ'   | 128 | 3800  | 205    | 98.1   | 1289.00 | -1285.90 |
| [Grid Search](https://wandb.ai/xiang-pan/HOBO/runs/2hh95hjr)  | 'IVF_PQ'   | 64  | 1     | 1      | 95.08  | 1733.67        | -7629.25        |


### HNSW
| Method                                                       | index_type | M  | efConstruction | ef  | recall | query_per_sec | loss          |
|--------------------------------------------------------------|------------|----|----------------|-----|--------|---------------|---------------|
| [BOHB](https://wandb.ai/xiang-pan/HOBO/runs/1gkilnbh)        | 'HNSW'     | 18 | 92             | 157 | 99.85  | 17868.6       | -17863.8      |
| [Grid Search](https://wandb.ai/xiang-pan/HOBO/runs/3vdvm6gs) | 'HNSW'     | 4  | 158            | 200 | 97.11  | 18331.74  | -18329.63 |
