import os
import subprocess
import time
import pickle
import io,sys
import requests
import json
import numpy as np
def render_config_yaml(model_weight,port,ctx_max_batch_size,gen_max_batch_size,free_gpu_mem_frac,context_tp,context_ep,context_dp,gen_tp,gen_ep,gen_dp,ctx_urls,gen_urls):
    config_str = \
f'''\
hostname: 0.0.0.0
port: {port}
model: {model_weight}
free_gpu_memory_fraction: {free_gpu_mem_frac}
backend: "pytorch"
use_cuda_graph: True
context_servers:
  num_instances: 1
  tensor_parallel_size: {context_tp}
  moe_expert_parallel_size: {context_ep}
  pipeline_parallel_size: 1
  max_batch_size: {ctx_max_batch_size}
#   kv_cache_config:
#     free_gpu_memory_fraction: {free_gpu_mem_frac}
  enable_attention_dp: {context_dp}
  urls:
      - "{ctx_urls}"
generation_servers:
  num_instances: 1
  tensor_parallel_size: {gen_tp}
  moe_expert_parallel_size: {gen_ep}
  pipeline_parallel_size: 1
  max_batch_size: {gen_max_batch_size}
#   kv_cache_config:
#     free_gpu_memory_fraction: {free_gpu_mem_frac}
  urls:
      - "{gen_urls}"
  enable_attention_dp: {gen_dp}
'''
    with open('disagg_config_rendered.yaml','w') as fw:
        fw.write(config_str)
    #os.popen('scp -P 12133 disagg_config_rendered.yaml 172.17.7.13:/workspace/minih/torch_bench/TensorRT-LLM/examples/disaggregated')

# sglang start cmd
def get_cmd(nproc=32,ssh_port=12133):
    cmd = ['mpirun',
           '--allow-run-as-root',
            '-np',
            f'{nproc}',
            '--hostfile',
            '/workspace/minih/torch_bench/TensorRT-LLM/examples/disaggregated/hostfile4nodes',
            '-mca',
            'plm_rsh_args',
            f'"-p {ssh_port}"',
            'python3',
            'launch_disaggregated_workers.py',
            '-c',
            'disagg_config_rendered.yaml'
            ]
    print(cmd)
    return cmd

# render config.pbtxt and start triton
def start_server(nproc=32,port=12133):
    os.popen('pkill -9 mpirun')
    os.popen(f"ps -ef | grep python3 | grep -v grep | grep -v {os.getpid()} | awk '{{print $2}}' | xargs kill -9")
    time.sleep(2)
    print('starting server')
    # generate launch.sh
    cmd = get_cmd(nproc,port)
    print(' '.join(cmd))
    res1 = subprocess.Popen(cmd,env=os.environ.copy())
    pid1 = res1.pid
    res2 = subprocess.Popen([
        "python3",
        'launch_disaggregated_server.py',
        '-t',
        '99999999',
        '-c',
        'disagg_config_rendered.yaml',
        ],
        env=os.environ.copy())
    pid2 = res2.pid

    while True:
        url = f"http://localhost:8000/health"
        try:
            response = requests.get(url)
            print('response',response)
            response.close()
            break
        except:
            pass
        time.sleep(10)

    print('server up!')
    return pid1,pid2

def benchmark_request_rate(num_prompt,ISL,OSL,request_rate,output_file,model_path):
    test_cmd = [
                'python3','-m','sglang.bench_serving','--backend','sglang-oai',
                '--dataset-name','random',
                '--model',f'{model_path}',
                '--num-prompt',
                f'{num_prompt}',
                '--random-input',
                f'{ISL}',
                '--random-output',
                f'{OSL}',
                '--request-rate',
                f'{request_rate}',
                '--random-range-ratio','1','--host','127.0.0.1',
                '--port','8000','--output-file',f'{output_file}'
                ]
    return test_cmd

def benchmark(num_prompt,ISL,OSL,max_concurrency,output_file,model_path):
    test_cmd = [
                'python3','-m','sglang.bench_serving','--backend','vllm',
                '--dataset-name','random',
                '--model',f'{model_path}'
                '--num-prompt',
                f'{num_prompt}',
                '--random-input',
                f'{ISL}',
                '--random-output',
                f'{OSL}',
                '--max-concurrency',
                f'{max_concurrency}',
                '--random-range-ratio','1','--host','127.0.0.1',
                '--port','31000','--output-file',f'{output_file}'
                ]
    return test_cmd


model_weight = '/workspace/DeepSeek-R1/models--deepseek-ai--DeepSeek-R1/'
max_batch_size = 1024
context_tp=16
context_ep = 8
context_dp = False
gen_tp = 16
gen_ep = 8
gen_dp = False
mem_frac = 0.2
ctx_urls = "172.17.7.14:8005"
gen_urls= "172.17.7.18:8005"
render_config_yaml(model_weight,8000,max_batch_size,max_batch_size,mem_frac,context_tp,context_ep,context_dp,gen_tp,gen_ep,gen_dp,ctx_urls,gen_urls)
start_server()
output_file = 'pd_test_result.out'
input_output = [
    [5600,1400],
]

single_batch_latency = 4.3
high_range = np.arange(0.5,33,.5)
low_range = np.arange(32,1025,32)
request_rates_high= [x/single_batch_latency for x in high_range]
request_rates_low= [x/single_batch_latency for x in low_range]
n=0
pid = -1
skip = 0

for ISL,OSL in input_output:
    for ind,request_rate in enumerate(request_rates_high):
        num_requests = int(high_range[ind]+1 ) * 3
        if n >= skip:
            server_config = {
                "ctx_tp":context_tp,
                "ctx_ep":context_ep,
                "ctx_dp":context_dp,
                "gen_tp":gen_tp,
                "gen_ep":gen_ep,
                "gen_dp":gen_dp,
                "server_mbs":max_batch_size,
                "request_rate":request_rate,
                "Equivalent concurrency": high_range[ind],
                "num_requests":num_requests,
            }
            json.dump(server_config,open(output_file,'a'))
            with open (output_file,'a') as fw:
                fw.write('\n')
            test_cmd = benchmark_request_rate(num_requests,ISL,OSL,request_rate,output_file,model_weight)
            subprocess.run(test_cmd,env=os.environ.copy())
            print('finish 1 benchmark')
        else:
            print('skip')
        n+=1
    # is_restart = True
    for ind,request_rate in enumerate(request_rates_low):
        # if low_range[ind] > 128 and is_restart:
        #     context_dp = True
        #     gen_dp = True
        #     render_config_yaml(model_weight,8000,max_batch_size,max_batch_size,mem_frac,context_tp,context_ep,context_dp,gen_tp,gen_ep,gen_dp,ctx_urls,gen_urls)
        #     start_server()
        #     is_restart = False
        num_requests = int(low_range[ind]+1) * 3
        if n >= skip:
            server_config = {
                "ctx_tp":context_tp,
                "ctx_ep":context_ep,
                "ctx_dp":context_dp,
                "gen_tp":gen_tp,
                "gen_ep":gen_ep,
                "gen_dp":gen_dp,
                "server_mbs":max_batch_size,
                "request_rate":request_rate,
                "Equivalent concurrency": float(low_range[ind]),
                "num_requests":num_requests,
            }
            json.dump(server_config,open(output_file,'a'))
            with open (output_file,'a') as fw:
                fw.write('\n')
            test_cmd = benchmark_request_rate(num_requests,ISL,OSL,request_rate,output_file,model_weight)
            subprocess.run(test_cmd,env=os.environ.copy())
            print('finish 1 benchmark')
        else:
            print('skip')
        n+=1

