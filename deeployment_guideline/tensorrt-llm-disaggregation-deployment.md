# TensorRT-LLM Disaggregated Deployment Guide

## 1. Initial Setup (4 GPU Nodes)
Node IPs: *.*.7.14, *.*.7.17, *.*.7.18, *.*.7.19

### Build Container
```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull
git config --global --add safe.directory '*'
make -C docker release_build
```

### Launch Container
```bash
docker run -it \
  --ipc=host \
  --privileged \
  --shm-size=32g \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus all \
  --volume /mnt/ddn:/workspace \
  -v /root/.ssh:/root/.ssh \
  -w /workspace \
  --name trtllm_server \
  tensorrt_llm/release:latest bash
```

## 2. SSH Configuration
Note: For disaggregation, *.*.7.14, *.*.7.17 are used for context_servers, *.*.7.18, *.*.7.19 for generation_servers.

### Setup SSH Environment
```bash
# Install and init openssh
apt update
apt install -y openssh-server

# Update SSH config
vim /etc/ssh/sshd_config
```

Add the following to sshd_config:
```
PermitRootLogin yes 
PubkeyAuthentication yes
Port 12233
```

```bash
# Start SSH service
service ssh start

# Validate SSH connection
ssh <IP_OF_THE_OTHER_CONTAINER> -p 12233
```

## 3. Configuration Files

### Hostfile Setup
```text
# hostfile
*.*.7.14 slots=8
*.*.7.17 slots=8
*.*.7.18 slots=8
*.*.7.19 slots=8
```

### Disaggregation Configuration
```yaml
# disagg_config_rendered.yaml
model: /workspace/DeepSeek-V3/models--deepseek-ai--DeepSeek-V3/
hostname: localhost
port: 8000
backend: pytorch
free_gpu_memory_fraction: 0.9

context_servers:
  num_instances: 1
  tensor_parallel_size: 16
  pipeline_parallel_size: 1
  moe_expert_parallel_size: 8
  max_num_tokens: 8192
  enable_attention_dp: false
  pytorch_backend_config:
    print_iter_log: true
  urls:
    - "*.17.7.14:8001"

generation_servers:
  num_instances: 1
  tensor_parallel_size: 16
  pipeline_parallel_size: 1
  moe_expert_parallel_size: 8
  max_batch_size: 64
  max_num_tokens: 64
  enable_attention_dp: false
  pytorch_backend_config:
    use_cuda_graph: true
    cuda_graph_padding_enabled: true
    cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32, 64]
    enable_overlap_scheduler: true
  urls:
    - "*.17.7.18:8001"
```

## 4. Launch Services

### Launch Context and Generation Servers
```bash
export TRTLLM_USE_MPI_KVCACHE=1
mpirun --allow-run-as-root -np 32 --hostfile ./hostfile \
    -mca plm_rsh_args "-p 12233" python3 launch_disaggregated_workers.py -c disagg_config_rendered.yaml &
```

### Launch Disaggregated Server
```bash
python3 launch_disaggregated_server.py -c disagg_config_rendered.yaml
```

## 5. Testing

### Using curl
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/workspace/DeepSeek-V3/models--deepseek-ai--DeepSeek-V3/",
        "prompt": "NVIDIA is a great company because",
        "max_tokens": 32,
        "temperature": 0
    }' -w "\n"
```

### Using Python Client
```bash
cd clients
python3 disagg_client.py -c ../disagg_config_rendered.yaml -p prompts.json
```
