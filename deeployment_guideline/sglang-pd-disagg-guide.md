# DeepSeek-R1 基于 SGLang 的部署与压测脚本

## 1. Docker 环境部署
```bash
# 1. 设置 Hugging Face Token（需替换为个人实际 Token）
export HF_TOKEN=${YOUR_HUGGIFACE_TOKEN}

# 2. 启动 Docker 容器（挂载缓存、网络、GPU 等资源）
docker run -it  \
  --ipc=host \
  --privileged \
  --gpus all \
  --network=host \
  --env "HF_TOKEN=$HF_TOKEN" \
  -v /raid/minih/:/root/.cache/huggingface/ \
  --ipc=host \
  --name deepseek_r1_host \
  lmsysorg/sglang:v0.4.10.post2-cu128-gb200  \
  bash 
```
Notice: 当前版本支持NV-FP4的IFB mode, 不支持NVFP4的PD分离部署


## 2. 依赖与资源补丁（容器内执行）
```bash
# 1. 安装基础依赖
pip install netifaces

# 2. 安装 SGLang 内核（CUDA 12.8 版本）
python3 -m pip install --no-cache-dir \
  https://github.com/sgl-project/whl/releases/download/v0.2.8/sgl_kernel-0.2.8+cu128-cp39-abi3-manylinux2014_$(uname -m).whl \
  --force-reinstall --no-deps

# 3. 安装 NVIDIA CuDNN 依赖
pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend

# 4. 下载并解压专家位置配置文件
cd /root/.cache/huggingface/
wget https://github.com/user-attachments/files/20036217/attachment_ep_statistics.zip
unzip attachment_ep_statistics.zip
```

> **注意**：需提前下载 `deepep.json` 配置文件，参考地址：  
> [https://github.com/ai-dynamo/dynamo/blob/4a11fb264b14dbaa67a872f24767828ed17db7e9/examples/sglang/configs/deepseek_r1/wideep/deepep.json](https://github.com/ai-dynamo/dynamo/blob/4a11fb264b14dbaa67a872f24767828ed17db7e9/examples/sglang/configs/deepseek_r1/wideep/deepep.json)  
> 下载后放置到 `/root/.cache/huggingface/` 目录下。


## 3. P 节点（预填充节点）部署
### 3.1 部署说明
- **节点规划**：`10.255.240.108`（预填充主节点）、`10.255.240.107`（预填充工作节点）
- **参数逻辑**：`--max-running-requests 6144` = `--cuda-graph-max-bs 768` × 8（预填充 GPU 总数）, 如果要跑更长的ISL/OSL, 需要增加'--context-length'
- **日志输出**：命令后台执行，日志写入对应文件（便于问题排查）

### 3.2 主节点（10.255.240.108）执行
```bash
MC_FORCE_MNNVL=1 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048 \
MC_TE_METRIC=true \
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
SGLANG_LOCAL_IP_NIC=enP5p9s0 \
GLOO_SOCKET_IFNAME=enP5p9s0 \
NCCL_SOCKET_IFNAME=enP5p9s0 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-R1 \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --dist-init-addr 10.255.240.108:5757 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --host 0.0.0.0 \
  --decode-log-interval 1 \
  --max-running-requests 6144 \
  --context-length 2176 \
  --disable-radix-cache \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --disable-shared-experts-fusion \
  --ep-num-redundant-experts 32 \
  --eplb-algorithm deepseek \
  --attention-backend cutlass_mla \
  --watchdog-timeout 1000000 \
  --init-expert-location /root/.cache/huggingface/attachment_ep_statistics/decode_in1000out1000.json \
  --disable-cuda-graph \
  --chunked-prefill-size 16384 \
  --max-total-tokens 32768 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --deepep-config /root/.cache/huggingface/deepep.json \
  --ep-dispatch-algorithm dynamic \
  > prefill-1.log 2>&1 &
```

### 3.3 工作节点（10.255.240.107）执行
```bash
MC_FORCE_MNNVL=1 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048 \
MC_TE_METRIC=true \
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
SGLANG_LOCAL_IP_NIC=enP5p9s0 \
GLOO_SOCKET_IFNAME=enP5p9s0 \
NCCL_SOCKET_IFNAME=enP5p9s0 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-R1 \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --dist-init-addr 10.255.240.108:5757 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --host 0.0.0.0 \
  --decode-log-interval 1 \
  --max-running-requests 6144 \
  --context-length 2176 \
  --disable-radix-cache \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --disable-shared-experts-fusion \
  --ep-num-redundant-experts 32 \
  --eplb-algorithm deepseek \
  --attention-backend cutlass_mla \
  --watchdog-timeout 1000000 \
  --init-expert-location /root/.cache/huggingface/attachment_ep_statistics/decode_in1000out1000.json \
  --disable-cuda-graph \
  --chunked-prefill-size 16384 \
  --max-total-tokens 32768 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --deepep-config /root/.cache/huggingface/deepep.json \
  --ep-dispatch-algorithm dynamic \
  > prefill-2.log 2>&1 &
```


## 4. D 节点（解码节点）部署
### 4.1 部署说明
- **节点规划**：`10.255.240.109`（解码主节点）、`10.255.240.110`（解码工作节点）
- **参数逻辑**：
  - 2 节点场景：`--max-running-requests 6144` = `--cuda-graph-max-bs 768` × 4 × 2（解码器 GPU 总数），如果要跑更长的ISL/OSL, 需要增加'--context-length'
  - 12 节点场景（GB200）：`--max-running-requests 36864` = `--cuda-graph-max-bs 768` × 4 × 12

### 4.2 主节点（10.255.240.109）执行
```bash
MC_FORCE_MNNVL=1 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768 \
MC_TE_METRIC=true \
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
SGLANG_LOCAL_IP_NIC=enP5p9s0 \
GLOO_SOCKET_IFNAME=enP5p9s0 \
NCCL_SOCKET_IFNAME=enP5p9s0 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-R1 \
  --trust-remote-code \
  --disaggregation-mode decode \
  --dist-init-addr 10.255.240.109:5757 \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --host 0.0.0.0 \
  --decode-log-interval 1 \
  --max-running-requests 6144 \
  --context-length 2176 \
  --disable-radix-cache \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --disable-shared-experts-fusion \
  --ep-num-redundant-experts 32 \
  --eplb-algorithm deepseek \
  --attention-backend cutlass_mla \
  --watchdog-timeout 1000000 \
  --init-expert-location /root/.cache/huggingface/attachment_ep_statistics/decode_in1000out1000.json \
  --chunked-prefill-size 36864 \
  --mem-fraction-static 0.82 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --ep-dispatch-algorithm static \
  --cuda-graph-bs 768 \
  --num-reserved-decode-tokens 512 \
  > decoder-1.log 2>&1 &
```

### 4.3 工作节点（10.255.240.110）执行
```bash
MC_FORCE_MNNVL=1 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768 \
MC_TE_METRIC=true \
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
SGLANG_LOCAL_IP_NIC=enP5p9s0 \
GLOO_SOCKET_IFNAME=enP5p9s0 \
NCCL_SOCKET_IFNAME=enP5p9s0 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-R1 \
  --trust-remote-code \
  --disaggregation-mode decode \
  --dist-init-addr 10.255.240.109:5757 \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --host 0.0.0.0 \
  --decode-log-interval 1 \
  --max-running-requests 6144 \
  --context-length 2176 \
  --disable-radix-cache \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --disable-shared-experts-fusion \
  --ep-num-redundant-experts 32 \
  --eplb-algorithm deepseek \
  --attention-backend cutlass_mla \
  --watchdog-timeout 1000000 \
  --init-expert-location /root/.cache/huggingface/attachment_ep_statistics/decode_in1000out1000.json \
  --chunked-prefill-size 36864 \
  --mem-fraction-static 0.82 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --ep-dispatch-algorithm static \
  --cuda-graph-bs 768 \
  --num-reserved-decode-tokens 512 \
  > decoder-2.log 2>&1 &
```


## 5. 负载均衡（LB）启动
> ⚠️ **关键前提**：必须确认 P 节点（预填充）和 D 节点（解码）的服务均已正常启动后，再执行此命令。
```bash
python3 -m sglang.srt.disaggregation.launch_lb \
  --prefill "http://10.255.240.108:30000" \
  --decode "http://10.255.240.109:30000" \
  --host 0.0.0.0 \
  --port 8000 \
  --timeout 3600 \
  > lb.log 2>&1 &
```


## 6. 压测相关操作
### 6.1 减缓 D 节点速度（避免初期过载）
```bash
curl -H "Content-Type: application/json" \
  -d '{"forward_sleep_time": 180}' \
  -X POST "http://10.255.240.109:30000/slow_down" \
  > slow.log 2>&1 &
```

### 6.2 启动压测（无需等待完成，可直接执行后续命令）
```bash
python3 -m sglang.bench_one_batch_server \
  --model deepseek-ai/DeepSeek-R1 \
  --base-url http://10.255.240.108:8000 \
  --batch-size 128 \
  --input-len 1000 \
  --output-len 1000 \
  --skip-warmup \
  > bench.log 2>&1 &
```

### 6.3 恢复 D 节点速度（压测饱和后执行）
> 等待约 10 分钟，确认 D 节点达到饱和状态后执行。
```bash
curl -H "Content-Type: application/json" -d '{"forward_sleep_time": null}' -X POST "http://10.255.240.111:30000/slow_down"
```

## 7. bench 脚本
```python
import os
import subprocess
import time
import pickle
import numpy as np
import io,sys
import requests


def benchmark(num_prompt,ISL,OSL,max_concurrency,output_file):
    test_cmd = [
                'python3','-m','sglang.bench_serving','--model', 'deepseek-ai/DeepSeek-R1',
                '--dataset-name','random',
                '--num-prompt',
                f'{num_prompt}',
                '--random-input',
                f'{ISL}',
                '--random-output',
                f'{OSL}',
                '--max-concurrency',
                f'{max_concurrency}',
                '--random-range-ratio','1.0','--base-url', 'http://10.255.240.108:8000', '--output-file',f'{output_file}'
                ]
    subprocess.run(test_cmd,env=os.environ.copy())

input_output = [
    [6000,1000],
    #[25000,1000]
]
concurrencies = [8,64, 128, 256, 384,512]

n=0
pid = -1
skip = 0
for ISL,OSL in input_output:
    for concurrency in concurrencies:
        num_requests = concurrency * 4
        if n >= skip:
            time.sleep(5)
            with open('tmp.out','a') as fw:
                fw.write(f'max_prefill:8192,max_running_requests:128,torch_compile:False,is_dp:False\n')
            benchmark(num_requests,ISL,OSL,concurrency,'tmp.out')
            print('finish 1 benchmark')
        else:
            print('skip')
        n+=1

```
