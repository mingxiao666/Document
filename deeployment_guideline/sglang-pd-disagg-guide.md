# Docker
```bash
export HF_TOKEN=${YOUR_HUGGIFACE_TOKEN}
docker run -it  --ipc=host --privileged --gpus all --network=host --env "HF_TOKEN=$HF_TOKEN" -v  /raid/minih/:/root/.cache/huggingface/  --ipc=host --name deepseek_r1_host lmsysorg/sglang:v0.4.10.post2-cu128-gb200  bash 
```

# Patch
```bash
pip install netifaces
python3 -m pip install --no-cache-dir https://github.com/sgl-project/whl/releases/download/v0.2.8/sgl_kernel-0.2.8+cu128-cp39-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps
pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend

cd /root/.cache/huggingface/
wget https://github.com/user-attachments/files/20036217/attachment_ep_statistics.zip
unzip attachment_ep_statistics
```

# P nodes
- 假设使用 `10.255.240.108` 作为预填充主节点（prefill master node），`10.255.240.107` 作为预填充工作节点（prefill worker node）。
- 将 `--cuda-graph-max-bs` 设置为 768，然后将 `--max-running-request` 设置为 6144；计算逻辑：6144 = 768 * 8，其中 8 是预填充 GPU 的总数。
- deepep.json 见 https://github.com/ai-dynamo/dynamo/blob/4a11fb264b14dbaa67a872f24767828ed17db7e9/examples/sglang/configs/deepseek_r1/wideep/deepep.json

### 在 10.255.240.108 上执行：
```bash
MC_FORCE_MNNVL=1 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048 MC_TE_METRIC=true SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True SGLANG_LOCAL_IP_NIC=enP5p9s0 GLOO_SOCKET_IFNAME=enP5p9s0 NCCL_SOCKET_IFNAME=enP5p9s0 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 PYTHONUNBUFFERED=1 python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --trust-remote-code --disaggregation-mode prefill --dist-init-addr 10.255.240.108:5757 --nnodes 2 --node-rank 0 --tp-size 8 --dp-size 8 --enable-dp-attention --host 0.0.0.0 --decode-log-interval 1 --max-running-requests 6144 --context-length 2176 --disable-radix-cache --moe-dense-tp-size 1 --enable-dp-lm-head --disable-shared-experts-fusion --ep-num-redundant-experts 32 --eplb-algorithm deepseek --attention-backend cutlass_mla --watchdog-timeout 1000000  --init-expert-location /root/.cache/huggingface/attachment_ep_statistics/decode_in1000out1000.json --disable-cuda-graph --chunked-prefill-size 16384 --max-total-tokens 32768 --moe-a2a-backend deepep --deepep-mode low_latency --deepep-config /root/.cache/huggingface/deepep.json --ep-dispatch-algorithm dynamic
```

### 在 10.255.240.107 上执行：
```bash
MC_FORCE_MNNVL=1 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048 MC_TE_METRIC=true SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True SGLANG_LOCAL_IP_NIC=enP5p9s0 GLOO_SOCKET_IFNAME=enP5p9s0 NCCL_SOCKET_IFNAME=enP5p9s0 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 PYTHONUNBUFFERED=1 python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --trust-remote-code --disaggregation-mode prefill --dist-init-addr 10.255.240.108:5757 --nnodes 2 --node-rank 1 --tp-size 8 --dp-size 8 --enable-dp-attention --host 0.0.0.0 --decode-log-interval 1 --max-running-requests 6144 --context-length 2176 --disable-radix-cache --moe-dense-tp-size 1 --enable-dp-lm-head --disable-shared-experts-fusion --ep-num-redundant-experts 32 --eplb-algorithm deepseek --attention-backend cutlass_mla --watchdog-timeout 1000000  --init-expert-location /root/.cache/huggingface/attachment_ep_statistics/decode_in1000out1000.json --disable-cuda-graph --chunked-prefill-size 16384 --max-total-tokens 32768 --moe-a2a-backend deepep --deepep-mode low_latency --deepep-config /root/.cache/huggingface/deepep.json --ep-dispatch-algorithm dynamic
```

# D nodes
- 假设使用 `10.255.240.109` 作为预填充主节点（prefill master node），`10.255.240.110` 作为预填充工作节点（prefill worker node）。
- 将 `--cuda-graph-max-bs` 设置为 768，然后将 `--max-running-request` 设置为 6144；计算逻辑：6144 = 768 * 4 * 2，其中 4*2 是解码器 GPU 的总数。若在 GB200 上使用 12 个节点作为解码器，则应设置为：768 * 4 * 12 = 36864。

### 在 10.255.240.109 上执行：
```bash
MC_FORCE_MNNVL=1 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768 MC_TE_METRIC=true SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True SGLANG_LOCAL_IP_NIC=enP5p9s0 GLOO_SOCKET_IFNAME=enP5p9s0 NCCL_SOCKET_IFNAME=enP5p9s0 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 PYTHONUNBUFFERED=1 python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --trust-remote-code --disaggregation-mode decode --dist-init-addr 10.255.240.109:5757 --nnodes 2 --node-rank 0 --tp-size 8 --dp-size 8 --enable-dp-attention --host 0.0.0.0 --decode-log-interval 1 --max-running-requests 6144 --context-length 2176 --disable-radix-cache --moe-dense-tp-size 1 --enable-dp-lm-head --disable-shared-experts-fusion --ep-num-redundant-experts 32 --eplb-algorithm deepseek --attention-backend cutlass_mla --watchdog-timeout 1000000  --init-expert-location  /root/.cache/huggingface/attachment_ep_statistics/decode_in1000out1000.json --chunked-prefill-size 36864 --mem-fraction-static 0.82 --moe-a2a-backend deepep --deepep-mode low_latency --ep-dispatch-algorithm static --cuda-graph-bs 768 --num-reserved-decode-tokens 512
```

### 在 10.255.240.110 上执行：
```bash
MC_FORCE_MNNVL=1 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768 MC_TE_METRIC=true SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True SGLANG_LOCAL_IP_NIC=enP5p9s0 GLOO_SOCKET_IFNAME=enP5p9s0 NCCL_SOCKET_IFNAME=enP5p9s0 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 PYTHONUNBUFFERED=1 python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --trust-remote-code --disaggregation-mode decode --dist-init-addr 10.255.240.109:5757 --nnodes 2 --node-rank 1 --tp-size 8 --dp-size 8 --enable-dp-attention --host 0.0.0.0 --decode-log-interval 1 --max-running-requests 6144 --context-length 2176 --disable-radix-cache --moe-dense-tp-size 1 --enable-dp-lm-head --disable-shared-experts-fusion --ep-num-redundant-experts 32 --eplb-algorithm deepseek --attention-backend cutlass_mla --watchdog-timeout 1000000  --init-expert-location  /root/.cache/huggingface/attachment_ep_statistics/decode_in1000out1000.json --chunked-prefill-size 36864 --mem-fraction-static 0.82 --moe-a2a-backend deepep --deepep-mode low_latency --ep-dispatch-algorithm static --cuda-graph-bs 768 --num-reserved-decode-tokens 512
```

# LB
```bash
python3 -m sglang.srt.disaggregation.launch_lb --prefill "http://10.255.240.108:30000" --decode "http://10.255.240.109:30000" --host 0.0.0.0 --port 8000 --timeout 3600
```

# slow down
```bash
curl -H "Content-Type: application/json" -d '{"forward_sleep_time": 180}' -X POST "http://10.255.240.109:30000/slow_down"
```

# start benchmark
- 无需等待此命令执行完成，即可运行下一条命令。
```bash
python3 -m sglang.bench_one_batch_server --model deepseek-ai/DeepSeek-R1 --base-url http://10.255.240.108:8000 --batch-size 128 --input-len 1000 --output-len 1000 --skip-warmup
```

# finish slowing down D nodes
- 等待一段时间（例如 10 分钟），待 D 节点达到饱和后，执行此命令。
```bash
curl -H "Content-Type: application/json" -d '{"forward_sleep_time": null}' -X POST "http://10.255.240.109:30000/slow_down"
```
