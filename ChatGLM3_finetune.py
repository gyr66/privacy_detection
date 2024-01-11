import os
from datetime import datetime
import random

model_path = r"/home/platform/GeWei/Projects/ChatGLM3/models"
# 定义变量
lr = 2e-5
num_gpus = 4
lora_rank = 8
lora_alpha = 32
lora_dropout = 0.1
max_source_len = 512
max_target_len = 128
dev_batch_size = 1
grad_accumularion_steps = 2
max_step = 10000
save_interval = 100
max_seq_len = 512
logging_steps=1

run_name = "news"
dataset_path = "data/GLM_train.jsonl"
datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f"output/{run_name}-{datestr}-{lr}"
master_port = random.randint(10000, 65535)

os.makedirs(output_dir, exist_ok=True)
# 构建命令
# --standalone --nnodes=1 --nproc_per_node={num_gpus} \
command = f"\
CUDA_VISIBLE_DEVICES=4,5,6,7 python /home/platform/GeWei/Projects/ChatGLM3/finetune_basemodel_demo/finetune.py\
    --train_format input-output \
    --train_file {dataset_path} \
    --lora_rank {lora_rank} \
    --lora_alpha {lora_alpha} \
    --lora_dropout {lora_dropout} \
    --max_seq_length {max_seq_len} \
    --preprocessing_num_workers 4 \
    --model_name_or_path {model_path} \
    --output_dir {output_dir} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_steps {max_step} \
    --logging_steps {logging_steps} \
    --save_steps {save_interval} \
    --learning_rate {lr}        \
"
print('pre is ok ! now activate command')
os.system(command)
print("Done!")