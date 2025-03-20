# NVIDIA Nsight Systems Setup Guide for TRTLLM

## 1. Run Profiling Command
```bash
nsys profile -t cuda,nvtx --nic-metrics=true --gpu-metrics-devices=all -c cudaProfilerApi --capture-range-end="repeat[]" --cuda-graph-trace=node -o trtllm-nsys.out
```

## 2. Install Required Dependencies
To enable `--nic-metrics=true`, install the following dependencies first:
```bash
apt-get install -y libibmad-dev libucx-dev libucx0
```

## 3. Configure TensorRT-LLM Profiling Environment
For TensorRT-LLM profiling, set the environment variable to specify the profiling range:
```bash
export TLLM_PROFILE_START_STOP="35-45"  # Profile iterations from 35 to 45
```
