#!/bin/bash
set -e
# 基础配置
MASTER_ADDR="10.192.23.158"
MASTER_PORT="29200"
SSH_PORT="12233"
NPROC_PER_NODE="4"
TRAIN_SCRIPT="/opt/megatron-lm/pretrain_gpt.py"
NETWORK_IF="enP5p9s0"
TOTAL_NODES=16
# 节点配置列表（IP:node_rank）
NODE_CONFIGS=(
    "10.192.23.142:1"
    "10.192.23.157:2"
    "10.192.23.156:3"
    "10.192.23.154:4"
    "10.192.23.153:5"
    "10.192.23.155:6" 
    "10.192.23.151:7"
    "10.192.23.150:8"
    "10.192.23.149:9"
    "10.192.23.148:10"
    "10.192.23.147:11"
    "10.192.23.146:12"
    "10.192.23.145:13"
    "10.192.23.144:14"
    "10.192.23.152:15"
)

# 检查节点数量
if [ ${#NODE_CONFIGS[@]} -ne $((TOTAL_NODES - 1)) ]; then
    echo "错误：节点配置数量不匹配！"
    echo "配置了 ${#NODE_CONFIGS[@]} 个从节点，但总节点数应该是 ${TOTAL_NODES}"
    exit 1
fi

# 在主节点执行命令（带日志）
echo "在主节点 ${MASTER_ADDR} 启动训练..."
export GLOO_SOCKET_IFNAME="${NETWORK_IF}"
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO 
export PYTHONPATH=$PYTHON_PATH:./megatron
export CUDA_DEVICE_MAX_CONNECTIONS=1
cd /mnt/nas/minih/nemotron_340b/training-script
mkdir -p ./tensorboard_logs/llama3_1_405b_fp8
# 启动主节点
nohup torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${TOTAL_NODES}" \
    --node_rank 0 \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
        --use-mcore-models \
        --num-layers 120 \
        --hidden-size 12288 \
        --ffn-hidden-size 49152 \
        --num-attention-heads 96 \
        --group-query-attention \
        --num-query-groups 8 \
        --kv-channels 128 \
        --seq-length 8192 \
        --max-position-embeddings 8192 \
        --position-embedding-type rope \
        --rotary-base 1000000 \
        --rotary-percent 1.0 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --swiglu \
        --init-method-std 0.0055 \
        --attention-backend fused \
        --apply-layernorm-1p \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --micro-batch-size 1 \
        --global-batch-size 64 \
        --train-samples 1953125000 \
        --lr-decay-samples 1949218748 \
        --lr-warmup-samples 3906252 \
        --lr 0.00008 \
        --min-lr 0.000008 \
        --decoupled-lr 2.0e-4 \
        --decoupled-min-lr 2.0e-5 \
        --lr-decay-style cosine \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --bf16 \
        --grad-reduce-in-bf16 \
        --cross-entropy-loss-fusion \
        --calculate-per-token-loss \
        --manual-gc \
        --empty-unused-memory-level 2 \
        --exit-duration-in-mins 360 \
        --use-distributed-optimizer \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --fp8-format hybrid \
        --fp8-amax-history-len 1024 \
        --fp8-amax-compute-algo max \
        --fp8-param-gather \
        --tensor-model-parallel-size 8 \
        --context-parallel-size 2 \
        --pipeline-model-parallel-size 4 \
        --sequence-parallel \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 128256 \
        --tiktoken-pattern v2 \
        --split '99,1,0' \
        --no-create-attention-mask-in-dataloader \
        --no-mmap-bin-files \
        --num-workers 1 \
        --log-interval 1 \
        --eval-iters 3200000 \
        --eval-interval 100000000 \
        --save-interval 100000000 \
        --log-throughput \
        --ckpt-format torch_dist \
        --save checkpoints/llama3_1_405b_fp8_new \
        --load checkpoints/llama3_1_405b_fp8_new  \
        --distributed-timeout-minutes 60 \
        --tensorboard-dir ./tensorboard_logs/llama3_1_405b_fp8 > ./master_llama405b_train.log 2>&1 &

MASTER_PID=$!
echo "主节点启动成功，PID: ${MASTER_PID}"

sleep 5  # 等待主节点初始化

# 批量启动从节点
echo "开始启动从节点..."
for config in "${NODE_CONFIGS[@]}"; do
    NODE_IP=$(echo "${config}" | cut -d':' -f1)
    NODE_RANK=$(echo "${config}" | cut -d':' -f2)

    echo "启动节点 ${NODE_IP}（rank ${NODE_RANK}）..."
    
    # 检查SSH连接
    if ! ssh -p "${SSH_PORT}" -o ConnectTimeout=10 "${NODE_IP}" "echo 'SSH连接成功'" 2>/dev/null; then
        echo "警告：无法连接到节点 ${NODE_IP}，跳过..."
        continue
    fi
    
    # 通过ssh在远程节点执行命令
    ssh -p "${SSH_PORT}" "${NODE_IP}" "
        cd /mnt/nas/minih/nemotron_340b/training-script
        mkdir -p ./benchmark_cache_llama3_1_405b_fp8_empty
        mkdir -p ./tensorboard_logs/llama3_1_405b_fp8
        export GLOO_SOCKET_IFNAME=${NETWORK_IF}
        export NCCL_NVLS_ENABLE=0
        export MASTER_ADDR=${MASTER_ADDR}
        export MASTER_PORT=${MASTER_PORT}
	export NCCL_DEBUG=INFO 
	export PYTHONPATH=$PYTHON_PATH:./megatron
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        
        nohup torchrun \
            --nproc_per_node ${NPROC_PER_NODE} \
            --nnodes ${TOTAL_NODES} \
            --node_rank ${NODE_RANK} \
            --master_addr ${MASTER_ADDR} \
            --master_port ${MASTER_PORT} \
            ${TRAIN_SCRIPT} \
	    --use-mcore-models \
		--num-layers 120 \
		--hidden-size 12288 \
		--ffn-hidden-size 49152 \
		--num-attention-heads 96 \
		--group-query-attention \
		--num-query-groups 8 \
		--kv-channels 128 \
		--seq-length 8192 \
		--max-position-embeddings 8192 \
		--position-embedding-type rope \
		--rotary-base 1000000 \
		--rotary-percent 1.0 \
		--attention-dropout 0.0 \
		--hidden-dropout 0.0 \
		--swiglu \
		--init-method-std 0.0055 \
		--attention-backend fused \
		--apply-layernorm-1p \
		--untie-embeddings-and-output-weights \
		--disable-bias-linear \
		--micro-batch-size 1 \
		--global-batch-size 64 \
		--train-samples 1953125000 \
		--lr-decay-samples 1949218748 \
		--lr-warmup-samples 3906252 \
		--lr 0.00008 \
		--min-lr 0.000008 \
		--decoupled-lr 2.0e-4 \
		--decoupled-min-lr 2.0e-5 \
		--lr-decay-style cosine \
		--clip-grad 1.0 \
		--weight-decay 0.1 \
		--adam-beta1 0.9 \
		--adam-beta2 0.95 \
		--bf16 \
		--grad-reduce-in-bf16 \
		--cross-entropy-loss-fusion \
		--calculate-per-token-loss \
		--manual-gc \
		--empty-unused-memory-level 2 \
		--exit-duration-in-mins 360 \
		--use-distributed-optimizer \
		--overlap-grad-reduce \
		--overlap-param-gather \
		--fp8-format hybrid \
		--fp8-amax-history-len 1024 \
		--fp8-amax-compute-algo max \
		--fp8-param-gather \
		--tensor-model-parallel-size 8 \
		--context-parallel-size 2 \
		--pipeline-model-parallel-size 4 \
		--sequence-parallel \
		--mock-data \
		--tokenizer-type NullTokenizer \
		--vocab-size 128256 \
		--tiktoken-pattern v2 \
		--split '99,1,0' \
		--no-create-attention-mask-in-dataloader \
		--no-mmap-bin-files \
		--num-workers 1 \
		--log-interval 1 \
		--eval-iters 3200000 \
		--eval-interval 10000000 \
		--save-interval 10000000 \
		--log-throughput \
		--ckpt-format torch_dist \
                --save checkpoints/llama3_1_405b_fp8_new \
                --load checkpoints/llama3_1_405b_fp8_new \
		--distributed-timeout-minutes 60 \
		--tensorboard-dir ./tensorboard_logs/llama3_1_405b_fp8  > ./${NODE_RANK}_llama405b_train.log 2>&1 &
        
        echo '节点 ${NODE_IP} 启动成功'
    " &
    
    sleep 2  # 错开启动时间，避免网络拥堵
done

# 等待所有后台任务完成
wait

echo "所有节点启动命令已下发完成！"
echo "主节点日志：master_train.log"

# 显示进程状态
echo "检查训练进程状态..."
ps aux | grep torchrun | grep -v grep || echo "未找到torchrun进程"

echo "脚本执行完成！"
