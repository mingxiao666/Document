# Dynamo+TRTLLM on GB200 部署指南


## 一、前提条件
1. 容器镜像：`gitlab-master.nvidia.com/dl/ai-dynamo/dynamo:4cbd4f38810f152c81f15f1cc7cc600c354587aa-32804297-tensorrtllm-arm64`
2. 本地目录 `/home/minih` 含：
   - `multinode` 脚本（`start_frontend_services.sh`、`start_trtllm_worker.sh`）, 从这里获取：https://github.com/ai-dynamo/dynamo/blob/main/components/backends/trtllm/multinode/
   - 引擎配置（`/mnt/engine_configs/deepseek_r1/mtp/mtp_prefill.yaml`、`/mnt/engine_configs/deepseek_r1/mtp/mtp_decode.yaml`），从这里获取：https://github.com/ai-dynamo/dynamo/tree/main/components/backends/trtllm/engine_configs
3. 模型路径：`/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-r1_pyt/` 可访问，可自行下载https://huggingface.co/nvidia/DeepSeek-R1-FP4


## 二、申请SLURM资源并配置调试环境
### 申请2节点资源
```bash
salloc \
  --partition="36x2-a01r" \
  --account="general_sa" \
  --job-name="general_sa-dynamo.trtllm" \
  --nodes 2
```

## 三、配置全局环境变量
```bash
# 1. 容器与挂载
export IMAGE="gitlab-master.nvidia.com/dl/ai-dynamo/dynamo:4cbd4f38810f152c81f15f1cc7cc600c354587aa-32804297-tensorrtllm-arm64"
export MOUNTS="/home/minih:/mnt,/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-r1_pyt/:/deepseek-r1_pyt"

# 2. 模型配置
export MODEL_PATH="/deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4/"
export SERVED_MODEL_NAME="hf-574fdb8-nim_fp4"

# 3. 引擎配置
export PREFILL_ENGINE_CONFIG="/mnt/engine_configs/deepseek_r1/mtp/mtp_prefill.yaml"
export DECODE_ENGINE_CONFIG="/mnt/engine_configs/deepseek_r1/mtp/mtp_decode.yaml"

# 4. SLURM基础配置
export PARTITION="36x2-a01r"
export ACCOUNT="general_sa"

# 5. 资源配置
export SLURM_JOB_ID="${SLURM_JOB_ID}"  # 使用salloc自动分配的Job ID
export NUM_PREFILL_NODES=1
export NUM_DECODE_NODES=1
export NUM_GPUS_PER_NODE=4

# 6. 头节点与服务地址（自动获取）
export SLURMD_NODENAME="${SLURMD_NODENAME}"  # 使用SLURM自动设置的头节点名
export HEAD_NODE="${SLURMD_NODENAME}"        # 复用头节点名
export HEAD_NODE_IP="$(hostname -i)"         # 自动获取当前节点（头节点）IP
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export DISAGGREGATION_STRATEGY="decode_first"

# 变量校验
if [[ -z ${IMAGE} || -z ${SLURM_JOB_ID} || -z ${HEAD_NODE_IP} ]]; then
  echo "ERROR: 缺失关键变量（IMAGE/SLURM_JOB_ID/HEAD_NODE_IP）"
  exit 1
fi
```


## 四、启动核心服务
### 启动Frontend（头节点），假设是ptyche0336
```bash
echo "=== 启动Frontend服务 ==="
srun \
  --overlap \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --verbose \
  --label \
  -A "${ACCOUNT}" \
  -J "general_sa-dynamo.trtllm-frontend" \
  --nodelist "${HEAD_NODE}" \
  --nodes 1 \
  --jobid "${SLURM_JOB_ID}" \
  /mnt/multinode/start_frontend_services.sh > /home/minih/start.log 2>&1 &

echo "=== 等待150秒初始化 ==="
sleep 150
```
注意：也不一定要等150s, 只要你在/home/minih/start.log看到如下的log就可以进行下一步：
```bash
0: 2025-08-21T08:17:04.681034Z  INFO dynamo_llm::http::service::service_v2: Starting HTTP service on: 0.0.0.0:8000 address="0.0.0.0:8000"
```

### 启动Prefill（头节点），假设是ptyche0336
```bash
echo -e "\n=== 启动Prefill节点 ==="
DISAGGREGATION_MODE=prefill \
ENGINE_CONFIG="${PREFILL_ENGINE_CONFIG}" \
srun \
  --mpi pmix \
  --oversubscribe \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --container-env "ETCD_ENDPOINTS=${ETCD_ENDPOINTS},NATS_SERVER=${NATS_SERVER},HEAD_NODE_IP=${HEAD_NODE_IP},HEAD_NODE=${HEAD_NODE},DISAGGREGATION_MODE=${DISAGGREGATION_MODE},DISAGGREGATION_STRATEGY=${DISAGGREGATION_STRATEGY},ENGINE_CONFIG=${ENGINE_CONFIG}" \
  --verbose \
  --label \
  -A "${ACCOUNT}" \
  -J "general_sa-dynamo.trtllm-prefill" \
  --nodelist "${HEAD_NODE}" \
  --nodes 1 \
  --ntasks-per-node "${NUM_GPUS_PER_NODE}" \
  --jobid "${SLURM_JOB_ID}" \
  /mnt/multinode/start_trtllm_worker.sh > /home/minih/preff1.log 2>&1 &

echo "=== 等待60秒初始化 ==="
sleep 60
```

### 启动Decode，假设是"ptyche0341"
```bash
echo -e "\n=== 启动Decode节点 ==="
DISAGGREGATION_MODE=decode \
ENGINE_CONFIG="${DECODE_ENGINE_CONFIG}" \
srun \
  --mpi pmix \
  --oversubscribe \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --container-env "ETCD_ENDPOINTS=${ETCD_ENDPOINTS},NATS_SERVER=${NATS_SERVER},HEAD_NODE_IP=${HEAD_NODE_IP},HEAD_NODE=${HEAD_NODE},DISAGGREGATION_MODE=${DISAGGREGATION_MODE},DISAGGREGATION_STRATEGY=${DISAGGREGATION_STRATEGY},ENGINE_CONFIG=${ENGINE_CONFIG}" \
  --verbose \
  --label \
  -A "${ACCOUNT}" \
  -J "general_sa-dynamo.trtllm-decode" \
  --nodelist "ptyche0341" \
  --nodes 1 \
  --ntasks-per-node "${NUM_GPUS_PER_NODE}" \
  --jobid "${SLURM_JOB_ID}" \
  /mnt/multinode/start_trtllm_worker.sh > /home/minih/decoder1.log 2>&1 &

echo "=== 等待60秒初始化 ==="
sleep 60
```


## 五、验证服务就绪
```bash
# 实时查看Frontend日志，等待模型加载成功标志
tail -f /home/minih/start.log
```
**就绪标志**：日志出现 `INFO dynamo_llm::discovery::watcher: added model model_name="hf-574fdb8-nim_fp4"`.


## 六、执行Benchmark测试
### 1 配置Benchmark变量
```bash
export SERVED_MODEL_NAME="hf-574fdb8-nim_fp4"
export HOST=localhost
export PORT=8000
```

### 2 发送Chat请求
```bash
curl -w "%{http_code}" ${HOST}:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "'${SERVED_MODEL_NAME}'",
  "messages": [
    {
      "role": "user",
      "content": "Tell me a story as if we were playing dungeons and dragons."
    }
  ],
  "stream": true,
  "max_tokens": 30
}'
```

### 3 Sglang bench
#### 测试脚本
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
                'python3','-m','sglang.bench_serving','--backend','sglang-oai-chat',
                '--dataset-name','random', '--model', 'hf-574fdb8-nim_fp4',
                '--num-prompt',
                f'{num_prompt}',
                '--random-input',
                f'{ISL}',
                '--random-output',
                f'{OSL}',
                '--max-concurrency',
                f'{max_concurrency}',
                '--random-range-ratio','1.0', '--host', '0.0.0.0', '--port', '8000', '--output-file',f'{output_file}'
                ]
    subprocess.run(test_cmd,env=os.environ.copy())

input_output = [
    [4000,1000],
]
concurrencies = [1,2,4,8,16,32,64,128]

n=0
pid = -1
skip = 0
for ISL,OSL in input_output:
    for concurrency in concurrencies:
        num_requests = concurrency * 4
        if n >= skip:
            time.sleep(5)
            with open('tmp-25k-nodp.4nodes.out','a') as fw:
                fw.write(f'max_prefill:8192,max_running_requests:128,torch_compile:False,is_dp:False\n')
            benchmark(num_requests,ISL,OSL,concurrency,'tmp-25k-nodp.4nodes.out')
            print('finish 1 benchmark')
        else:
            print('skip')
        n+=1

```
#### 关键注意事项
1. **后端选择**  
   必须使用 `sglang-oai-chat` 后端，而非 `sglang-oai`

2. **本地模型命名与路径处理**  
   - 若使用本地模型（如路径 `/deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4/`），需为其指定一个服务端定义的名称（例如 `hf-574fdb8-nim_fp4`）。  
   - 由于该名称不符合 HuggingFace 官方模型路径格式，直接运行脚本会导致程序尝试从 HuggingFace 仓库下载，从而报错。

3. **源码修改（跳过下载步骤）**  
   为解决上述问题，需修改 SGLang 基准测试源码：  
   - 源码路径：`/usr/local/lib/python3.12/dist-packages/sglang/bench_serving.py`L633, 把真实路径填写进去：pretrained_model_name_or_path = get_model("/deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4/")  

   修改后即可使用上述脚本正常运行基准测试。

