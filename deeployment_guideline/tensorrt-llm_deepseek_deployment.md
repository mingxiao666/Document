# TensorRT-LLM Deployment and Benchmarking Guide

## 1. Environment Preparation

### Build Docker
```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull
make -C docker release_build
```

### Enter Docker
```bash
docker run -it \
  --ipc=host \
  --privileged \
  --shm-size=32g \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus=all \
  --volume /raid/minih:/workspace \
  -v ~/.ssh:/root/.ssh \
  -w /workspace \
  --name trtllm_server \
  tensorrt_llm/release:latest bash
```

### Prepare Config Files
```json
# hf_quant_config.json
{
  "quantization": {
    "quant_algo": "FP8_BLOCK_SCALES",
    "kv_cache_quant_algo": null
  }
}
```

```yaml
# extra-llm-api-config.yml
enable_attention_dp: true
pytorch_backend_config:
  enable_overlap_scheduler: true
  use_cuda_graph: false
  cuda_graph_max_batch_size: 64
```

## 2. Server Launch

### Single Node
```bash
trtllm-serve \
  --backend pytorch \
  --tp_size 8 \
  --ep_size 4 \
  --kv_cache_free_gpu_memory_fraction 0.95 \
  --trust_remote_code \
  --max_batch_size 128 \
  --max_num_tokens 4096 \
  --extra_llm_api_options extra-llm-api-config.yml \
  /raid/minih/hub/models--deepseek-ai--DeepSeek-R1/snapshots/8a58a132790c9935686eb97f042afa8013451c9f/
```

### Multi Nodes Setup (Two Nodes Example)

#### SSH Configuration
```bash
# Install SSH
apt update
apt install -y openssh-server

# Update SSH config
vim /etc/ssh/sshd_config
```

Add to sshd_config:
```
PermitRootLogin yes 
PubkeyAuthentication yes
Port 12233
```

```bash
service ssh start

# Validate SSH
ssh <IP_OF_THE_OTHER_CONTAINER> -p 12233
```

#### Launch Multi-Node Server
```bash
/usr/local/mpi/bin/mpirun -np 16 \
  --hostfile /workspace/torch_bench/my_hostfile \
  -mca plm_rsh_args "-p 12233" \
  --allow-run-as-root \
  trtllm-llmapi-launch trtllm-serve \
  --backend pytorch \
  --tp_size 16 \
  --ep_size 8 \
  --kv_cache_free_gpu_memory_fraction 0.95 \
  --trust_remote_code \
  --max_batch_size 128 \
  --max_num_tokens 4096 \
  --extra-llm-api-config extra-llm-api-config.yml \
  /raid/minih/hub/models--deepseek-ai--DeepSeek-R1/snapshots/8a58a132790c9935686eb97f042afa8013451c9f/
```

## 3. Benchmarking

### 3.1 Using curl
```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-R1",
    "prompt": "Who is the president of United States?",
    "max_tokens": 32,
    "temperature": 0
  }'
```

### 3.2 Using OpenAI Interface
```python
import openai

client = openai.Client(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print(response)
```

### 3.3 trtllm_bench

#### Data Generation
```bash
python /workspace/torch_bench/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py \
  --tokenizer=/raid/minih/hub/models--deepseek-ai--DeepSeek-R1/snapshots/8a58a132790c9935686eb97f042afa8013451c9f/ \
  --stdout token-norm-dist \
  --num-requests=${num_dataset_requests} \
  --input-mean=${isl} \
  --output-mean=${osl} \
  --input-stdev=0 \
  --output-stdev=0 > /workspace/torch_bench/flex_dataset2.txt
```

#### Single Node Benchmarking
```bash
# Create config file
echo -e "enable_attention_dp: true\npytorch_backend_config:\n  enable_overlap_scheduler: true\n  use_cuda_graph: false\n  cuda_graph_max_batch_size: 8" > /workspace/torch_bench/extra-llm-api-config_large_bs.yml

# Run benchmark
trtllm-bench \
  --model /raid/minih/hub/models--deepseek-ai--DeepSeek-R1/snapshots/8a58a132790c9935686eb97f042afa8013451c9f/ \
  throughput \
  --backend pytorch \
  --max_batch_size ${batch_size} \
  --max_num_tokens ${num_tokens} \
  --dataset /workspace/torch_bench/flex_dataset2.txt \
  --tp 8 \
  --ep ${ep} \
  --warmup 10 \
  --num_requests ${num_requests} \
  --concurrency ${concurrency} \
  --streaming \
  --kv_cache_free_gpu_mem_fraction 0.9 \
  --extra_llm_api_options /workspace/torch_bench/extra-llm-api-config_large_bs.yml
```

#### Multi-Node Benchmarking
```bash
/usr/local/mpi/bin/mpirun -np 16 \
  --hostfile /workspace/hostfile \
  -mca plm_rsh_args "-p 12233" \
  --verbose -display-map \
  --allow-run-as-root \
  trtllm-llmapi-launch trtllm-bench \
  --model /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1/snapshots/8a58a132790c9935686eb97f042afa8013451c9f \
  throughput \
  --backend pytorch \
  --max_batch_size ${batch_size} \
  --max_num_tokens ${num_tokens} \
  --dataset /workspace/torch_bench/flex_dataset2.txt \
  --tp 16 \
  --ep ${ep} \
  --warmup 10 \
  --num_requests ${num_requests} \
  --concurrency ${concurrency} \
  --streaming \
  --kv_cache_free_gpu_mem_fraction 0.9 \
  --extra_llm_api_options /workspace/torch_bench/extra-llm-api-config_large_bs.yml
```
