# 1. 容器与挂载
export IMAGE="gitlab-master.nvidia.com/dl/ai-dynamo/dynamo:4cbd4f38810f152c81f15f1cc7cc600c354587aa-32804297-tensorrtllm-arm64"
export MOUNTS="/home/minih:/mnt,/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-r1_pyt/:/deepseek-r1_pyt"

# 2. 模型配置
export MODEL_PATH="/deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4/"
export SERVED_MODEL_NAME="hf-574fdb8-nim_fp4"

# 3. 引擎配置
export PREFILL_ENGINE_CONFIG="/mnt/engine_configs/deepseek_r1/mtp/mtp_prefill_nodp_1p2d.yaml"
export DECODE_ENGINE_CONFIG="/mnt/engine_configs/deepseek_r1/mtp/mtp_decode_nodp_1p2d.yaml"

# 4. SLURM基础配置
export PARTITION="36x2-a01r"
export ACCOUNT="general_sa"
export DECODE_NODES="ptyche0147,ptyche0149"
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
  /mnt/multinode/start_frontend_services.sh > /home/minih/start1p2d.log 2>&1 &

echo "=== 等待150秒初始化 ==="
sleep 150

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
  /mnt/multinode/start_trtllm_worker.sh > /home/minih/preff-1p2d.log 2>&1 &

echo "=== 等待60秒初始化 ==="
sleep 60


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
  --nodelist "${DECODE_NODES}" \
  --nodes 2 \
  --ntasks-per-node "${NUM_GPUS_PER_NODE}" \
  --jobid "${SLURM_JOB_ID}" \
  /mnt/multinode/start_trtllm_worker.sh > /home/minih/decoder-1p2d-d1.log 2>&1 &

echo "=== 等待60秒初始化 ==="
sleep 60

tail -f /home/minih/start1p2d.log
