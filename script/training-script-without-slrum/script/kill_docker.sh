#!/bin/bash
set -euo pipefail

# 节点IP列表
NODES=(
        "10.192.23.158"
        "10.192.23.157"
        "10.192.23.141"
        "10.192.23.156"
        "10.192.23.155"
        "10.192.23.154"
        "10.192.23.153"
        "10.192.23.152"
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
)

# 循环清理每个节点
for node in "${NODES[@]}"; do
  echo "===== 处理节点: $node ====="
  # 登录节点，查找包含nemo_train的容器并强制删除
  ssh "$node" <<EOF
    echo "当前节点的nemo_train容器列表:"
    docker ps --filter "name=nemo_train" --format "{{.Names}}"
    echo "开始清理..."
    docker rm -f \$(docker ps --filter "name=nemo_train" -q) 2>/dev/null || true
    echo "清理完成"
EOF
  echo "-------------------------"
done

echo "所有节点清理操作已执行完毕"
