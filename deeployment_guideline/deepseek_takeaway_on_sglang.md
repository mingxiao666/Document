# SGL Usage Notes

1. **GPU Cards and tp * dp * ep Relationship**
   - The number of GPU cards is not equal to tp * dp * ep
   - Don't try to set dp-size and ep-size (these parameters exist due to historical reasons)
   - When enabling dp, use `--enable-dp-attention`
   - When enabling ep, use `--enable-ep-moe`
   - Total GPU cards should equal tp number (where tp = dp = ep)

2. **EP Support for DeepSeek r1**
   - EP for DeepSeek r1 is not supported in current version(0.4.3)
   - Reference: [Issue #2740](https://github.com/sgl-project/sglang/issues/2740)

3. **tp>16 Limitation**
   - Not support tp>16 due to bug
   - Reference: [Issue #3345](https://github.com/sgl-project/sglang/issues/3345)

4. **dp-attention Effects**
   - Enable `--enable-dp-attention` will increase concurrency and TFFT
   - Will gain benefit for throughput when traffic request is big
   - However, it is not a linearly growing relationship
   - When traffic is too high, throughput may decrease

5. **Fixed Concurrency Setting**
   - Set `--max-concurrency` for fixed concurrency
   - Setting a reasonable max concurrency value can get better throughput and prevent overloading

6. **OSL Impact on Performance**
   - When traffic request rate is high, with fixed input parameters:
   - Larger OSL leads to less TTFT changes
   - Worse TPOT performance
   - Input throughput deteriorates

7. **torch-compile Impact**
   - Enable `--enable-torch-compile` can usually gain performance benefit
   - However, it may not always guarantee performance boost
   - In some cases, performance benefit might be negligible

8. **KV Cache Pool Setting**
   - `--mem-fraction-static 0.8` is for KV cache pool size proportion
   - Example: 0.8 means 80% memory allocated to KV cache pool
   - If not meet OOM, larger value will gain throughput benefit

9. **Custom Allreduce**
   - Support custom allreduce when tp<8
   - `--disable-custom-all-reduce` won't always harm performance
   - It's a trade-off between compute and communication resources

10. **Radix Cache Setting**
    - Consider using `--disable-radix-cache` when cache hit rate is low
    - Can reduce overhead and improve performance
    - Don't disable when cache hit rate is high, may increase latency and decrease performance

11. **Quantization Settings**
    - `--torchao-config`: Specify torch quantization configuration file
    - `--quantization`: Set weight quantization type (e.g., `--quantization fp8`)
    - `--kv-cache-dtype`: Set KV cache quantization data type (e.g., `--kv-cache-dtype fp8_e5m2`)

12. **DeepSeek R1 Service Crash**
    - Crashes occasionally on 2*H100
    - [Fix branch](https://github.com/nvcastet/sglang/tree/fix_all_gather_cuda_graph) has performance regression

13. **MTP Support**
    - Supported in sglang 0.4.3.post2
    - Performance increment with small concurrency(<=64) only

14. **flashinfer mla Limitations**
    - Currently(0.4.3) not always beneficial
    - Especially poor performance with large OSL
    - Reference: [Issue #3917](https://github.com/sgl-project/sglang/issues/3917)

15. **First Test Case TTFT**
    - After launch kernel, first test case usually takes longer TTFT

16. **Startup Issues Resolution**
    - If stuck in startup, try decreasing chunk_prefill_size and max_running_request

17. **NCCL Warning Resolution**
    - For "NCCL WARN Error: failed to extend /dev/shm/nccl-iYZbWq" issue:
    - Pass `--shm-size=32G` when entering docker image
    - Open maximum limit number of ulimit
