#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOO_SOCKET_IFNAME=bond0
export HYDRA_FULL_ERROR=1
export MELLANOX_VISIBLE_DEVICES=all
export NCCL_DEBUG=INFO
export NCCL_IBEXT_DISABLE=0
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_3,mlx5_2,mlx5_1,mlx5_0,mlx5_5,mlx5_4,mlx5_7,mlx5_6
#export NCCL_IB_HCA=mlx5_3,mlx5_1,mlx5_5,mlx5_7
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_TC=96
export NCCL_IB_TIMEOUT=20
export NCCL_SOCKET_IFNAME=bond0
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=bond0
export PMIX_MCA_psec=^munge
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib/:/opt/tritonserver/backends/tensorrtllm:/usr/local/tensorrt/lib:/usr/local/cuda/compat/lib:/tool/cublas/cuda-12.8/lib64
