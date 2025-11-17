#!/bin/bash
set -eo pipefail
export PS4='+ [PREPARE DEBUG] $(date "+%H:%M:%S"): '
set -x

# ==================== 配置参数 ====================
export NUM_NODES=18
export DOCKER_IMAGE="nvcr.io/nvidia/nemo:25.07.01"
export WORK_DIR=$PWD
export SSH_PORT=12233
export CONTAINER_NAME="nemo_train_minih"
export NODELIST=(
	"10.192.23.158"
	"10.192.23.157"
	"10.192.23.152"
	"10.192.23.156"
	"10.192.23.154"
	"10.192.23.153"
	"10.192.23.155"
	"10.192.23.151"
	"10.192.23.150"
	"10.192.23.149"
	"10.192.23.148"
	"10.192.23.147"
	"10.192.23.146"
	"10.192.23.145"
	"10.192.23.144"
	"10.192.23.143"
	"10.192.23.142"
	"10.192.23.141"
)

# ==================== 前置检查 ====================
echo "[PREPARE] 验证节点数量：配置=$NUM_NODES，实际=${#NODELIST[@]}"
if [ ${#NODELIST[@]} -ne $NUM_NODES ]; then
  echo "[ERROR] 节点数量不匹配" && exit 1
fi

if [ ! -f ~/.ssh/id_rsa ] || [ ! -f ~/.ssh/id_rsa.pub ]; then
  echo "[ERROR] 宿主机未找到SSH密钥对，请先配置节点间免密" && exit 1
fi

# ==================== 启动容器并配置SSH ====================
echo "[PREPARE] 开始在所有节点启动容器并配置SSH（端口$SSH_PORT）"
for idx in "${!NODELIST[@]}"; do
  node=${NODELIST[$idx]}
  echo "[PREPARE] 处理节点 $((idx+1))/$NUM_NODES：$node"

  # 停止旧容器
  ssh $node "docker rm -f $CONTAINER_NAME &>/dev/null || true"

  # 启动新容器（移除注释，修复参数格式）
  echo "[PREPARE] 在$node启动容器：$CONTAINER_NAME"
  ssh $node "docker run -d \
    --name $CONTAINER_NAME --privileged \
    --gpus all \
    --network host \
    --ipc=host \
    -p $SSH_PORT:$SSH_PORT \
    -v /dev/:/dev \
    -v $WORK_DIR:$WORK_DIR \
    -v ~/.ssh:/root/.ssh:ro \
    -w /opt/NeMo \
    $DOCKER_IMAGE \
    tail -f /dev/null"

  # 等待容器启动
  for i in {1..30}; do
    if ssh $node "docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null | grep -q true"; then
      echo "[PREPARE] $node的容器已启动" && break
    fi
    sleep 1
    [ $i -eq 30 ] && { echo "[ERROR] $node容器启动超时" && exit 1; }
  done

  # 配置容器内SSH
  echo "[PREPARE] 在$node的容器内配置SSH"
  ssh $node "docker exec $CONTAINER_NAME bash -c '
    set -x
    apt-get update && apt-get install -y openssh-server net-tools && bash /mnt/nas/minih/nemotron_340b/llm-benchmarking-collection-internal-release-25.07.01/nemotron/setup.sh
    sed -i \"s/#Port 22/Port $SSH_PORT/\" /etc/ssh/sshd_config
    sed -i \"s/#PermitRootLogin prohibit-password/PermitRootLogin yes/\" /etc/ssh/sshd_config
    service ssh start
    netstat -tulpn | grep $SSH_PORT || (echo \"SSH启动失败\" && exit 1)
  '" || { echo "[ERROR] $node的SSH配置失败" && exit 1; }

  # 测试容器内SSH
  if ! ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$node "echo 容器内SSH连接成功"; then
    echo "[ERROR] 无法连接$node的容器（端口$SSH_PORT）" && exit 1
  fi
done

echo "[PREPARE] 所有节点容器和SSH配置完成！"
echo "[PREPARE] 容器名：$CONTAINER_NAME"
echo "[PREPARE] SSH端口：$SSH_PORT"
