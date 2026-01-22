# Dynamo+TRTLLM部署指南(以GB200为例)


## 一、前提条件
1. 容器镜像: `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.0@sha256:c292fd7fe416bcb10a1a0fc599615278c7dc79befda223252c2343a242500a96`
https://nvidia.slack.com/archives/C099TALBD7B/p1760713552921789?thread_ts=1760664642.044459&cid=C099TALBD7B slack上搜到同一个错误说是要升级trtllm version到1.2.0rc0.post1以上
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
export IMAGE="nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.0@sha256:c292fd7fe416bcb10a1a0fc599615278c7dc79befda223252c2343a242500a96"
export MOUNTS="/home/minih:/mnt,/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-r2_pyt/:/deepseek-r1_pyt"

# 2. 模型配置
export MODEL_PATH="/deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4/"
export SERVED_MODEL_NAME="hf-574fdb8-nim_fp4"

# 3. 引擎配置
export PREFILL_ENGINE_CONFIG="/mnt/engine_configs/deepseek_r1/mtp/mtp_prefill.yaml"
export DECODE_ENGINE_CONFIG="/mnt/engine_configs/deepseek_r1/mtp/mtp_decode.yaml"

# 4. SLURM基础配置
export PARTITION="36x2-a01r"
export ACCOUNT="general_sa"
export DECODE_NODE="ptyche0341"


# 5. 资源配置
export SLURM_JOB_ID="${SLURM_JOB_ID}"  # 使用salloc自动分配的Job ID
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
先 export 变量，然后只传递变量名（不带值）：
# 先 export 所有需要的变量
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export DISAGGREGATION_MODE="prefill"
export ENGINE_CONFIG="${PREFILL_ENGINE_CONFIG}"
export DISAGGREGATION_STRATEGY="decode_first"

# --container-env 只传递变量名
echo -e "\n=== 启动Prefill节点 ==="
srun \
  --mpi pmix \
  --oversubscribe \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE,DISAGGREGATION_MODE,DISAGGREGATION_STRATEGY,ENGINE_CONFIG \
  --verbose \
  --label \
  -A "${ACCOUNT}" \
  -J "general_sa-dynamo.trtllm-prefill" \
  --nodelist "${HEAD_NODE}" \
  --nodes 1 \
  --ntasks-per-node "${NUM_GPUS_PER_NODE}" \
  --jobid "${SLURM_JOB_ID}" \
  /mnt/multinode/start_trtllm_worker.sh > /home/minih/preff-1p1d.log 2>&1 &

echo "=== 等待60秒初始化 ==="
sleep 60
```


```bash
##错误处理：
0: ValueError: Overriding KV cache quantization with an invalid type "PyTorchConfig.kv_cache_dtype="bf16" Accepted types are "('fp8', 'nvfp4', 'auto')".
KV Cache 量化类型 bf16 不被支持，只接受以下三种：
• fp8 - FP8 量化（推荐）
• nvfp4 - NVFP4 量化
• auto - 自动选择
✅ 解决方案
你需要修改配置文件 mtp_prefill.yaml（和 mtp_decode.yaml）中的 kv_cache_dtype 设置。
```

### 启动Decode，假设是"ptyche0341"
```bash
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export DISAGGREGATION_MODE="prefill"
export ENGINE_CONFIG="${PREFILL_ENGINE_CONFIG}"
export DISAGGREGATION_STRATEGY="decode_first"


echo -e "\n=== 启动Decode节点 ==="
DISAGGREGATION_MODE="decode" \
ENGINE_CONFIG="${DECODE_ENGINE_CONFIG}" \
srun \
  --mpi pmix \
  --oversubscribe \
  --container-image "${IMAGE}" \
  --container-mounts "${MOUNTS}" \
  --container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE,DISAGGREGATION_MODE,DISAGGREGATION_STRATEGY,ENGINE_CONFIG\
  --verbose \
  --label \
  -A "${ACCOUNT}" \
  -J "general_sa-dynamo.trtllm-decode" \
  --nodelist "${DECODE_NODE}" \
  --nodes 1 \
  --ntasks-per-node "${NUM_GPUS_PER_NODE}" \
  --jobid "${SLURM_JOB_ID}" \
/mnt/multinode/start_trtllm_worker.sh > /home/minih/decoder-1p1d.log 2>&1 &

echo "=== 等待60秒初始化 ==="
sleep 60
```
## 五、验证服务就绪
```bash
# 实时查看Frontend日志，等待模型加载成功标志
tail -l /home/minih/start.log
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

### 3 Sglang backend bench
###查询当前容器作业ID
```bash
$squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             62829 interacti general_   gegong  R    1:13:51      2 poly[0164-0165]

```
###修改bech.sh脚本,脚本的路径是script/dynamo-trtllm/bench.sh
```bash
$vim bench.sh
srun --jobid=62829  --overlap --container-image=nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.0@sha256:c292fd7fe416bcb10a1a0fc599615278c7dc79befda223252c2343a242500a96  --container-mounts=/lustre/fsw/general_sa:/scripts,/home/gegong:/home/gegong,/lustre/fsw/general_sa/gegong/models/DeepSeek-R1-0528-NVFP4/:/DeepSeek-R1-0528-NVFP4  --mpi pmix  bash -c "
cd /home/minih
pip install sglang==0.5.1.post2 pybase64 
cp bench_serving.py /usr/local/lib/python3.12/dist-packages/sglang/bench_serving.py
python benchonly1p1d_mtp.py
"
```
```bash
###错误处理
cp: cannot create regular file '/usr/local/lib/python3.12/dist-packages/sglang/bench_serving.py': No such file or directory
cp: cannot create regular file '/usr/local/lib/python3.12/dist-packages/sglang/bench_serving.py': No such file or directory
1. 修复 bench.sh 中的 sglang 路径
# sglang 路径错误，容器中 sglang 安装在 /opt/dynamo/venv/ 而不是 /usr/local/lib/。修改 bench.sh
vim bench.sh
将：
cp bench_serving.py /usr/local/lib/python3.12/dist-packages/sglang/bench_serving.py
改为：
cp bench_serving.py /opt/dynamo/venv/lib/python3.12/site-packages/sglang/bench_serving.py
```



### 修改benchonly1p1d_mtp.py测试脚本,脚本的路径是script/dynamo-trtllm/benchonly1p1d_mtp.py
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
                '--random-range-ratio','1.0', '--host', '10.66.43.245', '--port', '8000', '--output-file',f'{output_file}'
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
```bash
###错误处理:
服务未运行或连接地址错误
ConnectionRefusedError: [Errno 111] Connect call failed ('0.0.0.0', 8000)
#benchmark 尝试连接 0.0.0.0:8000，但服务没有运行。
确认服务是否运行
在运行 benchmark 之前，先确保推理服务已启动：
# 检查服务状态
curl -s http://10.66.43.245:8000/health

# 或者检查 0.0.0.0:8000（如果在同一容器内）
curl -s http://localhost:8000/health

修复 benchonly1p1d_mtp.py 中的 host 地址
检查 --host 参数是否正确：
# 如果服务在同一节点，使用 localhost
'--host', 'localhost',

# 如果服务在其他节点，使用正确的 IP  ##经过验证，建议使用`hostname -i`命令显示的IP地址参数
'--host', '10.66.43.245',
```

#### 关键注意事项
1. **后端选择**  
   必须使用 `sglang-oai-chat` 后端，而非 `sglang-oai`

2. **本地模型命名与路径处理**  
   - 若使用本地模型（如路径 `/deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4/`），需为其指定一个服务端定义的名称（例如 `hf-574fdb8-nim_fp4`）。  
   - 由于该名称不符合 HuggingFace 官方模型路径格式，直接运行脚本会导致程序尝试从 HuggingFace 仓库下载，从而报错。

3. **源码修改（跳过下载步骤）**  
   为解决上述问题，需修改 SGLang 基准测试源码：  
   - 源码路径：`/opt/dynamo/venv/lib/python3.12/site-packages/sglang/bench_serving.py`L632, 把真实路径填写进去：pretrained_model_name_or_path = get_model("/deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4/")  

   修改后即可使用上述脚本正常运行基准测试。

```bash
###执行benchmark测试,查看测试结果
bash bench.sh







Namespace(backend='sglang-oai-chat', base_url=None, host='10.66.43.245', port=8000, dataset_name='random', dataset_path='', model='DeepSeek-R1-0528-NVFP4', tokenizer=None, num_prompts=4, sharegpt_output_len=None, sharegpt_context_len=None, random_input_len=4000, random_output_len=1000, random_range_ratio=1.0, request_rate=inf, max_concurrency=1, output_file='tmp-4k1k-nodp-mtp.1p1d.out', output_details=False, disable_tqdm=False, disable_stream=False, return_logprob=False, seed=1, disable_ignore_eos=False, extra_request_body=None, apply_chat_template=False, profile=False, lora_name=None, prompt_suffix='', pd_separated=False, flush_cache=False, warmup_requests=1, tokenize_prompt=False, gsp_num_groups=64, gsp_prompts_per_group=16, gsp_system_prompt_len=2048, gsp_question_len=128, gsp_output_len=256)

Downloading from https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json to /tmp/ShareGPT_V3_unfiltered_cleaned_split.json
/tmp/ShareGPT_V3_unfiltered_cleaned_split.json: 100%|██████████| 642M/642M [00:03<00:00, 178MB/s]2M [00:02<00:00, 204MB/s]    | 180M/642M [00:01<00:02, 200MB/s]
/tmp/ShareGPT_V3_unfiltered_cleaned_split.json: 100%|██████████| 642M/642M [00:06<00:00, 98.1MB/s]
#Input tokens: 16000
#Output tokens: 4000
Starting warmup with 1 sequences...
Warmup completed with 1 sequences. Starting main benchmark run...
#Input tokens: 16000
#Output tokens: 4000
Starting warmup with 1 sequences...
Warmup completed with 1 sequences. Starting main benchmark run...
100%|██████████| 4/4 [00:44<00:00, 11.05s/it]

============ Serving Benchmark Result ============
Backend:                                 sglang-oai-chat
Traffic request rate:                    inf
Max request concurrency:                 1
Successful requests:                     4
Benchmark duration (s):                  44.21
Total input tokens:                      16000
Total generated tokens:                  4000
Total generated tokens (retokenized):    4000
Request throughput (req/s):              0.09
Input token throughput (tok/s):          361.93
Output token throughput (tok/s):         90.48
Total token throughput (tok/s):          452.41
Concurrency:                             1.00
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   11051.21
Median E2E Latency (ms):                 11104.40
---------------Time to First Token----------------
Mean TTFT (ms):                          191.38
Median TTFT (ms):                        196.03
P99 TTFT (ms):                           207.42
---------------Inter-Token Latency----------------
Mean ITL (ms):                           10.87
Median ITL (ms):                         10.92
P95 ITL (ms):                            11.04
P99 ITL (ms):                            11.11
Max ITL (ms):                            21.77
==================================================
```


```bash
###附注
在slurm环境中下载模型权重的方法：
一、在容器内下载（推荐）
# 设置变量
export HF_TOKEN="hf_xxxxxx"
export MODEL_DIR="/lustre/fsw/general_sa/minih/models/DeepSeek-R1-0528-NVFP4"

# 创建目录
mkdir -p "${MODEL_DIR}"

# 使用 srun 在容器内下载
srun -A general_sa -N 1 --ntasks-per-node=1 -t 4:00:00 \
  --container-image "nvcr.io/nvidia/pytorch:24.12-py3" \
  --container-mounts "/lustre/fsw/general_sa/minih:/workspace" \
  bash -c "
    export HF_TOKEN='${HF_TOKEN}'
    pip install -q huggingface_hub[cli]
    hf download nvidia/DeepSeek-R1-0528-NVFP4 \
      --local-dir /workspace/models/DeepSeek-R1-0528-NVFP4 \
      --token \${HF_TOKEN} \
      --force-download
  "
srun: WARNING: Please set a name for this job, formatted like this:
srun: 	general_sa-<subproject>.<details>
pyxis: importing docker image: nvcr.io/nvidia/pytorch:24.12-py3
pyxis: imported docker image: nvcr.io/nvidia/pytorch:24.12-py3
DEPRECATION: Loading egg at /usr/local/lib/python3.12/dist-packages/texttable-1.7.0-py3.12.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
DEPRECATION: Loading egg at /usr/local/lib/python3.12/dist-packages/opt_einsum-3.4.0-py3.12.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
DEPRECATION: Loading egg at /usr/local/lib/python3.12/dist-packages/nvfuser-0.2.13a0+0d33366-py3.12-linux-aarch64.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
DEPRECATION: Loading egg at /usr/local/lib/python3.12/dist-packages/looseversion-1.3.0-py3.12.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
DEPRECATION: Loading egg at /usr/local/lib/python3.12/dist-packages/lightning_utilities-0.11.9-py3.12.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
DEPRECATION: Loading egg at /usr/local/lib/python3.12/dist-packages/lightning_thunder-0.2.0.dev0-py3.12.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
DEPRECATION: Loading egg at /usr/local/lib/python3.12/dist-packages/igraph-0.11.8-py3.12-linux-aarch64.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
DEPRECATION: Loading egg at /usr/local/lib/python3.12/dist-packages/dill-0.3.9-py3.12.egg is deprecated. pip 25.1 will enforce this behaviour change. A possible replacement is to use pip for package installation. Discussion can be found at https://github.com/pypa/pip/issues/12330
WARNING: huggingface-hub 1.3.2 does not provide the extra 'cli'
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.

[notice] A new release of pip is available: 24.3.1 -> 25.3
[notice] To update, run: python -m pip install --upgrade pip
Fetching 173 files:   0%|          | 0/173 [00:00<?, ?it/s]Still waiting to acquire lock on /workspace/models/DeepSeek-R1-0528-NVFP4/.cache/huggingface/.gitignore.lock (elapsed: 0.1 seconds)
Fetching 173 files: 100%|██████████| 173/173 [01:52<00:00,  1.54it/s]
/workspace/models/DeepSeek-R1-0528-NVFP4






二、申请2节点资源:
minih@login-ptyche01:~$ salloc \
  --partition="36x2-a01r" \
  --account="general_sa" \
  --job-name="general_sa-dynamo.trtllm" \
  --nodes 2
salloc: Pending job allocation 671684
salloc: job 671684 queued and waiting for resources
salloc: job 671684 has been allocated resources
salloc: Granted job allocation 671684
salloc: Waiting for resource configuration
salloc: Nodes ptyche01[0211,0215] are ready for job
minih@ptyche010211:~$
```
