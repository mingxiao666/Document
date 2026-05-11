# NSYS Profiling Process for vLLM Inference Testing

This document demonstrates how to run NSYS profiling for vLLM inference testing.

## Step 1: Install nsys

```bash
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

## Step 2: Launch the vLLM server under NSYS profiling
Notice: you can change max_iterations and delay_iterations in "--profiler-config '{"profiler": "cuda", "max_iterations": 5, "delay_iterations": 1000}' " according to your need.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node --cuda-event-trace=false --sample=non
e --cpuctxsw=none --capture-range=cudaProfilerApi --trace=cuda,nvtx --capture-range-end repeat --output /minih/dsv4-c256-flash-5it.nsys-rep vllm serve /minih
/DeepSeek-V4-Flash --tensor-parallel-size 1 --data-parallel-size 4 --pipeline-parallel-size 1 --kv-cache-dtype fp8 --trust-remote-code --block-size 256 --no-enable-prefix-caching --enable-expert-parallel --moe-backend deep_gemm_mega_moe --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["a
ll"]}' --attention_config.use_fp4_indexer_cache True --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser
 deepseek_v4 --max-cudagraph-capture-size 2048 --max-model-len 10240 --max-num-batched-tokens 2048  --enable-logging-iteration-details --profiler-config '{"p
rofiler": "cuda", "max_iterations": 5, "delay_iterations": 1000}' > gb300_semi_c256-flash.log 2>&1 &
```

In this step, the vLLM server is started under `nsys profile` so that GPU execution during inference can be captured.

## Step 3: When server is up, run the benchmark workload

```bash
vllm bench serve \
    --model /minih/DeepSeek-V4-Flash  \
    --dataset-name random \
    --num-prompts 320 \
    --random-input-len 8192 \
    --random-output-len 1024 \
    --temperature 0.0 \
    --num-warmups 0 \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --max-concurrency 32 \
    --base-url http://localhost:8000 \
    --profile \
    --save-result \
    --save-detailed \
    --result-filename /minih/bench_withprofile_8192in_1024out_32bs_result.json
```

This step sends benchmark traffic to the vLLM server and triggers the profiling workflow.
