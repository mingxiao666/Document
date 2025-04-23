#! /bin/bash

isl_osl_list="1,1024 128,1024 1000,128 2000,128"
batch_size_list=(128)
ep_list=(1 2 4 8)
#concurrency_list=(1024 1280)
concurrency_list=(1 2 4 8 16 32 64 96 128)

function generate_dataset() {
python /workspace/torch_bench/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py \
--tokenizer=/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1/snapshots/8a58a132790c9935686eb97f042afa8013451c9f \
--stdout token-norm-dist \
--num-requests=${num_dataset_requests} \
--input-mean=${isl} \
--output-mean=${osl} \
--input-stdev=0 \
--output-stdev=0 > /workspace/torch_bench/flex_dataset2.txt
}

function run_benchmark() {
echo -e "enable_attention_dp: false\npytorch_backend_config:\n  enable_overlap_scheduler: true\n  use_cuda_graph: true\n  cuda_graph_max_batch_size: 128" > /workspace/torch_bench/extra-llm-api-config_large_bs.yml

#--model_path  /workspace/hub/models--deepseek-ai--DeepSeek-R1 \
/usr/local/mpi/bin/mpirun -np 16  --hostfile /workspace/hostfile -mca plm_rsh_args "-p 12133" --verbose -display-map --allow-run-as-root trtllm-llmapi-launch trtllm-bench \
--model /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1/snapshots/8a58a132790c9935686eb97f042afa8013451c9f \
throughput \
--backend pytorch \
--max_batch_size ${batch_size} \
--max_num_tokens ${num_tokens} \
--dataset /workspace/torch_bench/flex_dataset2.txt \
--tp 16 \
--ep ${ep} \
--warmup 10 \
--num_requests ${num_requests} \
--concurrency ${concurrency} \
--streaming \
--kv_cache_free_gpu_mem_fraction 0.9 \
--extra_llm_api_options /workspace/torch_bench/extra-llm-api-config_large_bs.yml 2>&1 | tee /workspace/torch_bench/new_logs/trt_bench_i${isl}_o${osl}_bs${batch_size}_ep${ep}_c${concurrency}_r${num_requests}.log
}
#  /workspace/torch_bench/logs2/trt_bench_i${isl}_o${osl}_bs${batch_size}_ep${ep}_c${concurrency}_r${num_requests}.log
for isl_osl in ${isl_osl_list}; do
    IFS=',' read -r isl osl <<< "${isl_osl}"
    num_dataset_requests=10000
    echo "Start generating dataset for isl=${isl}, osl=${osl}"
    generate_dataset
    for ep in ${ep_list[@]}; do
        for batch_size in ${batch_size_list[@]}; do
            for concurrency in ${concurrency_list[@]}; do
                num_requests=$((4*concurrency < 4096 ? 4*concurrency : 4096))
                num_tokens=$((isl+batch_size-1))
                echo "Start running benchmark for isl=${isl}, osl=${osl}, ep=${ep}, batch size ${batch_size}, concurrency ${concurrency}"
                run_benchmark
            done
        done
    done
done

