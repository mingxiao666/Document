#!/bin/bash

# 定义所有节点IP列表（包含当前登录的节点）
nodes=(
10.192.23.222
10.192.23.221
10.192.23.212
10.192.23.220
10.192.23.219
10.192.23.218
10.192.23.217
10.192.23.216
10.192.23.213
10.192.23.214
10.192.23.215
10.192.23.211
10.192.23.207
10.192.23.210
10.192.23.205
10.192.23.209
10.192.23.206
10.192.23.208
)

# 循环处理每个节点
for node in "${nodes[@]}"; do
    echo "===== 开始配置节点: $node ====="
    
    # 1. 临时设置performance模式（立即生效）
    echo "临时切换CPU模式..."
    ssh $node "sudo sh -c 'for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > \$cpu; done'"
    
    # 2. 创建systemd服务实现永久生效
    echo "配置永久生效服务..."
    ssh $node "sudo sh -c 'cat > /etc/systemd/system/cpu-performance.service << EOF
[Unit]
Description=Set CPU scaling governor to performance

[Service]
Type=oneshot
ExecStart=/bin/sh -c \"for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > \$cpu; done\"

[Install]
WantedBy=multi-user.target
EOF'"
    
    # 3. 启用并启动服务
    ssh $node "sudo systemctl daemon-reload && sudo systemctl enable --now cpu-performance.service"
    
    # 4. 验证配置结果
    echo "验证配置结果..."
    ssh $node "echo 当前CPU模式:; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq"
    
    echo "===== 节点 $node 配置完成 ====="
    echo
done

echo "所有节点已完成performance模式配置，重启后仍会保持生效"
