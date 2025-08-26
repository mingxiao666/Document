        mpirun -n 1 --allow-run-as-root --oversubscribe trtllm-serve /deepseek-r1_pyt/safetensors_mode-instruct/hf-574fdb8-nim_fp4/  \
        --host '0.0.0.0' --port 30000 --backend pytorch --tp_size 4 \
        --ep_size 4 --pp_size 1 \
        --max_batch_size 128 --max_num_tokens 8192 \
        --extra_llm_api_options   ./extra-llm-api-config-nodp-fp4.yml  \
        --kv_cache_free_gpu_memory_fraction 0.85 > 1node.log 2>&1 &
