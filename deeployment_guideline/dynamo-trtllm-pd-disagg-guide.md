# Dynamo+TRTLLM 部署指南


## 一、前提条件
1. 容器镜像：`gitlab-master.nvidia.com/dl/ai-dynamo/dynamo:4cbd4f38810f152c81f15f1cc7cc600c354587aa-32804297-tensorrtllm-arm64`
2. 本地目录 `/home/minih` 含：
   - `multinode` 脚本（`start_frontend_services.sh`、`start_trtllm_worker.sh`）
   - 引擎配置（`/mnt/engine_configs/deepseek_r1/mtp/mtp_prefill.yaml`、`/mnt/engine_configs/deepseek_r1/mtp/mtp_decode.yaml`）
3. 模型路径：`/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-r1_pyt/` 可访问


## 二、申请SLURM资源并配置调试环境
### 申请2节点资源
```bash
salloc \
  --partition="batch" \
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
export SERVED_MODEL_NAME="${MODEL_PATH}"

# 3. 引擎配置
export PREFILL_ENGINE_CONFIG="/mnt/engine_configs/deepseek_r1/mtp/mtp_prefill.yaml"
export DECODE_ENGINE_CONFIG="/mnt/engine_configs/deepseek_r1/mtp/mtp_decode.yaml"

# 4. SLURM基础配置
export PARTITION="batch"
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


## 六、步骤5：执行Benchmark测试
### 6.1 配置Benchmark变量
```bash
export SERVED_MODEL_NAME="hf-574fdb8-nim_fp4"
export HOST=localhost
export PORT=8000
```

### 6.2 发送Chat请求
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
