source activate /home/zhufl/anaconda3/bin/activate base

env CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_task_player_multifeat.py