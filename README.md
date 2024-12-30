# Engeneering hw3 | DistributedDataParallel

Group 17

report: https://sii-czxy.feishu.cn/wiki/Flu7wMvuCisUTekVYBccFWnKnbg?from=from_copylink

implemented version & terminal command:
1. single card: `python main_singlecard.py -n official_singlecard`
2. torch.DDP: `python -m torch.distributed.launch --nproc_per_node 4 main_ddp.py -n official_ddp`
3. parameter-server: `python main_paramserver.py --name official_paramserver`
4. all-reduce tree: `python main_allreduce_tree.py --name official_allreduce_tree`
5. all-reduce ring: `python main_allreduce_ring.py --name official_allreduce_ring`
6. vanilla selfDDP: `python -m torch.distributed.launch --nproc_per_node 4 main_selfddp.py -n official_selfddp`
7. syncBN selfDDP: `python -m torch.distributed.launch --nproc_per_node 4 main_selfddp_syncbn.py -n official_selfddp_syncbn`

all the experiment logs used in the report can be found in dir `outputs`.
