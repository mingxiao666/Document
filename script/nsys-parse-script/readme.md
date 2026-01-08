### Step 1: Verify Kernel Classification in `parse-nsys.py`
First, confirm that the category configuration in `parse-nsys.py` matches your actual kernel naming rules (adjust keywords if needed):
```python
    categories = [
        ('Communication', ['ncclDevKernel_', 'ncclKernel', 'moe_comm']),
        ('GEMM', [
            'gemm', 'Gemm',          # 通用 GEMM 关键词
            'cutlass::Kernel2',     # 6KD Cutlass GEMM 识别
            'cutlass::device_kernel',# 6KD Cutlass GEMM 识别
            'matMul', 'MatMul'      # 其他矩阵乘法命名
            'nvjet'
        ]),
        ('Attention', ['flash_attention', 'mha', 'FlashInfer', 'attention', 'mla']),
        ('MoE', ['moe', 'Expert', 'finalizeMoe', 'expandInputRows', 'fused_moe']),
        ('Elementwise',['elementwise']),
        ('Others', ['.*'])          # 兜底
    ]
```

### Step 2: Execute the Following Commands
```bash
# Generate NSys profiling logs (H200)
nsys stats --report cuda_gpu_trace --report cuda_gpu_kern_sum --report cuda_api_sum --format csv,column --output .,-   profiling-260115/h200-profile-1224/h200.qwen235.tp2ep2.nodp.4k1.c64.nsys-rep >h200.qwen235.tp2ep2.nodp.4k1.c64.log

# Generate NSys profiling logs (H20)
nsys stats --report cuda_gpu_trace --report cuda_gpu_kern_sum --report cuda_api_sum --format csv,column --output .,-   profiling-260115/h20-profile-1224/qwen235.tp2ep2.nodp.4k1.c64.nsys-rep >h20.qwen235.tp2ep2.nodp.4k1.c64.log

# Parse logs to generate structured results (H200)
python parse-nsys.py h200.qwen235.tp2ep2.nodp.4k1.c64.log > h200.qwen235.tp2ep2.nodp.4k1.c64.result

# Parse logs to generate structured results (H20)
python parse-nsys.py h20.qwen235.tp2ep2.nodp.4k1.c64.log > h20.qwen235.tp2ep2.nodp.4k1.c64.result

# Compare results with parse-table.py
python parse-table.py h20.qwen235.tp2ep2.nodp.4k1.c64.result h200.qwen235.tp2ep2.nodp.4k1.c64.result
```
