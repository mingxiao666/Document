# SGL Deployment and Testing Guide

## 1. Environment Preparation

### Docker Setup
```bash
export HF_TOKEN={your token}

docker run -it \
  --ipc=host \
  --privileged \
  --gpus all \
  --network=host \
  --env "HF_TOKEN=$HF_TOKEN" \
  -v /raid/minih:/root/.cache/huggingface \
  --ipc=host \
  --name deepseek_r1_host \
  lmsysorg/sglang:latest bash
```

## 2. Test Launch

### Single Node
```bash
# Server launch
python3 -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-R1 \
  --tp 8 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e5m2 \
  --trust-remote-code \
  --enable-torch-compile \
  --disable-cuda-graph \
  --enable-dp-attention

# Benchmarking command
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --random-range-ratio 1 \
  --num-prompt 600 \
  --request-rate 2 \
  --random-input 1024 \
  --random-output 1024 \
  --output-file deepseek_v3_8xh200_FP8_online_output.jsonl
```

### Multi Nodes (2 nodes example)
```bash
# Node 0
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1 \
  --tp 16 \
  --dist-init-addr 10.6.131.6:20000 \
  --nnodes 2 \
  --node-rank 0 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 40000 \
  --enable-torch-compile \
  --quantization fp8 \
  --kv-cache-dtype fp8_e5m2 \
  --disable-cuda-graph \
  --enable-dp-attention

# Node 1
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1 \
  --tp 16 \
  --dist-init-addr 10.6.131.6:20000 \
  --nnodes 2 \
  --node-rank 1 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 40000 \
  --enable-torch-compile \
  --quantization fp8 \
  --kv-cache-dtype fp8_e5m2 \
  --disable-cuda-graph \
  --enable-dp-attention

# Benchmarking command on client
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --random-range-ratio 1 \
  --num-prompt 600 \
  --request-rate 2 \
  --random-input 1024 \
  --random-output 1024 \
  --output-file deepseek_v3_8xh200_FP8_online_output.jsonl
```

## 3. Profile with Nsight

### Nsys Installation
```bash
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

### Profiling Commands
```bash
# Server profiling
nsys profile --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  -o sglang.out \
  --delay 300 \
  --duration 70 \
  python3 -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-R1 \
  --tp 8 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e5m2 \
  --trust-remote-code \
  --enable-torch-compile \
  --disable-cuda-graph \
  --enable-dp-attention

# Client benchmarking
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --random-range-ratio 1 \
  --num-prompt 600 \
  --request-rate 2 \
  --random-input 1024 \
  --random-output 1024 \
  --output-file deepseek_v3_8xh200_FP8_online_output.jsonl
```

## 4. Parameters Reference

### Server Parameters
- **FP8**: 
  - `--quantization fp8`
  - `--kv-cache-dtype fp8_e5m2`

- **TP/DP/EP**:
  - Tensor Parallelism (TP) is base
  - Data Parallelism (DP): add `--enable-dp-attention`
  - Expert Parallelism (EP): add `--enable-ep-moe` (not supported for DeepSeek)
  - Note: When enabled, dp-size = tp-size (ep-size = tp-size)

- **Triton**:
  - `--enable-torch-compile`
  - `--disable-cuda-graph`

- **Other Options**:
  - Multinodes: `--nnodes`, `--node-rank`
  - RadixCache: `--disable-radix-cache`
  - Allreduce: `--disable-custom-all-reduce`
  - Memory: `--mem-fraction-static 0.8`

### Client Parameters
- **Input/Output Length**:
  - `--random-input 1024`
  - `--random-output 1024`

- **Performance Settings**:
  - Concurrency: `--max-concurrency 2`
  - Traffic: 
    - `--num-prompt 300`
    - `--request-rate 1`

- **Dataset**:
  - `--dataset-name random`
