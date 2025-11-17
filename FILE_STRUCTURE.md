# Repository File Structure

```
.
├── FILE_STRUCTURE.md
├── README.md
├── deeployment_guideline
│   ├── deepseek_takeaway_on_sglang.md
│   ├── dynamo-trtllm-pd-disagg-guide.md
│   ├── sglang-pd-disagg-guide.md
│   ├── sglang_deepseek_deployment.md
│   ├── tensorrt-llm-disaggregation-deployment.md
│   ├── tensorrt-llm_deepseek_deployment.md
│   └── trtllm-profiling.md
├── script
│   ├── dynamo-trtllm
│   │   ├── 1p1d.sh
│   │   ├── 1p1d_mtp.sh
│   │   ├── 1p2d.sh
│   │   ├── 1p2d_mtp.sh
│   │   ├── 2p1d.sh
│   │   ├── 2p1d_mtp.sh
│   │   ├── 2p2d.sh
│   │   ├── 2p2d_mtp.sh
│   │   ├── bench.sh
│   │   ├── bench_serving.py
│   │   ├── benchonly1p1d.py
│   │   ├── benchonly1p1d_mtp.py
│   │   ├── benchonly1p2d.py
│   │   ├── benchonly1p2d_mtp.py
│   │   ├── benchonly2p1d.py
│   │   ├── benchonly2p1d_mtp.py
│   │   ├── benchonly2p2d.py
│   │   ├── benchonly2p2d_mtp.py
│   │   ├── engine_configs
│   │   │   ├── agg.yaml
│   │   │   ├── decode.yaml
│   │   │   ├── deepseek_r1
│   │   │   ├── llama4
│   │   │   └── prefill.yaml
│   │   └── parse.py
│   ├── kill_sglang.sh
│   ├── nccl.sh
│   ├── parse_result_sglang_bench.py
│   ├── parse_result_trtllm_bench.py
│   ├── parse_result_trtllm_bench_pd.py
│   ├── sglang_bench_multinodes.py
│   ├── sglang_bench_singlenode.py
│   ├── test-script-for-sglang0.4.6
│   │   ├── bench.py
│   │   ├── parse_result_sglang_bench.py
│   │   ├── serve.mtp.fa3.sh
│   │   └── serve.mtp.infer.sh
│   ├── training-script-without-slrum
│   │   ├── README.md
│   │   └── script
│   │       ├── check_all_nodes.sh
│   │       ├── checker.sh
│   │       ├── kill_docker.sh
│   │       ├── launch_docker.sh
│   │       ├── pull_docker.sh
│   │       ├── run_llama405b_16nodes.sh
│   │       ├── set_perf_mode.sh
│   │       └── torch_allreduce_test.py
│   ├── trtllm-slrum
│   │   ├── 1node.sh
│   │   ├── 2nodes.sh
│   │   ├── bench.py
│   │   ├── extra-llm-api-config-nodp-fp4-ck.yml
│   │   ├── extra-llm-api-config-nodp-fp4.yml
│   │   └── parse.py
│   ├── trtllm_bench_multinodes.sh
│   ├── trtllm_bench_pd_multinodes.sh
│   └── trtllm_bench_singlenode.sh
└── template
    ├── disagg_config_4nodes
    └── hostfile_4nodes

12 directories, 62 files
```

Total: 12 directories, 62 files
