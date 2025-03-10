import os
import subprocess
import time
import pickle
import numpy as np
import io,sys
import requests


# sglang start cmd
def get_cmd(model_path,tp,max_prefill_tokens,max_running_requests,mem_fraction,torch_compile,is_dp,dist_init_addr,node_rank,nnodes=2):
    cmd = ['python3',
            '-m',
            'sglang.launch_server',
            '--model-path',
            f'{model_path}',
            '--trust-remote-code',
            '--tp',
            f'{tp}',
            '--max-prefill-tokens',
            f'{max_prefill_tokens}',
            '--max-running-requests',
            f'{max_running_requests}',
            '--host',
            '0.0.0.0',
            '--port',
            '31000',
            '--disable-radix-cache',
            '--mem-fraction-static',
            f'{mem_fraction}',
            '--stream-output',
            '--nnodes',
            f'{nnodes}',
            '--node-rank',
            f'{node_rank}',
            '--dist-init-addr',
            f'{dist_init_addr}'
            ]
    if torch_compile:
        cmd += [
            '--enable-torch-compile',
            '--torch-compile-max-bs',
            f'{max_running_requests}'
        ]
    if is_dp:
        cmd += [
            '--dp',
            f'{tp}',
            '--enable-dp-attention'
        ]
    print(cmd)
    return cmd

# render config.pbtxt and start triton
def start_server(model_path,tp,max_prefill_tokens,max_running_requests,mem_fraction,torch_compile,is_dp,dist_init_addr,nnodes=2):
    print('starting server')
    # generate launch.sh

    cmd1 = get_cmd(model_path,tp,max_prefill_tokens,max_running_requests,mem_fraction,torch_compile,is_dp,dist_init_addr,0,nnodes)
    cmd2 = get_cmd(model_path,tp,max_prefill_tokens,max_running_requests,mem_fraction,torch_compile,is_dp,dist_init_addr,1,nnodes)

    launch_str = f'''\
#!/bin/bash
ssh -p 12322 root@10.6.131.4 "source /workspace/nccl.sh && nohup {' '.join(cmd2)} > server.log 2>&1 &" &
source /workspace/nccl.sh && {' '.join(cmd1)}
    '''

    with open('launch_2node.sh','w') as fw:
        fw.write(launch_str)

    res = subprocess.Popen(['bash','launch_2node.sh'],env=os.environ.copy())
    pid = res.pid

    while True:
        url = f"http://localhost:31000/health_generate"
        try:
            response = requests.get(url)
            print('response',response)
            response.close()
            break
        except:
            pass
        time.sleep(10)

    print('server up!')
    return pid
def benchmark(num_prompt,ISL,OSL,max_concurrency,output_file):
    test_cmd = [
                'python3','-m','sglang.bench_serving','--backend','sglang',
                '--dataset-name','random',
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
    subprocess.run(test_cmd,env=os.environ.copy())




model_path = '/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1/snapshots/8a58a132790c9935686eb97f042afa8013451c9f/'
tp=16
input_output = [
    [1,1024],
    [128,1024],
    [1000,128],
    [2000,128],
]
concurrencies = [1,8,16,32,64,96,128,256,512,1024,2048]
dp_set = [False,True]
compile_set = [True]
n=0
pid = -1
skip = 0
for ISL,OSL in input_output:
    for concurrency in concurrencies:
        if ISL == 1000:
            max_prefill_tokens = min(ISL * concurrency,2000)
        else:
            max_prefill_tokens = min(ISL * concurrency,2048)
        max_running_requests = concurrency
        mem_fraction = 0.8
        torch_compile = True
        is_dp = False
        num_request = concurrency
        if concurrency > 128:
            is_dp = True
        if n >= skip:
            os.popen('bash kill_sglang.sh')
            if pid >0:
                os.popen(f'kill -9 {pid}')
            os.popen(f'lsof -t -i:31000 | grep -v "{os.getpid()}" | xargs -r kill -9')
            time.sleep(5)
            print('killed')
            print('model_path,tp,max_prefill_tokens,max_running_requests,mem_fraction,torch_compile,is_dp')
            print(model_path,tp,max_prefill_tokens,max_running_requests,mem_fraction,torch_compile,is_dp)
            print('concurrency*4,ISL,OSL,concurrency')
            print(concurrency*4,ISL,OSL,concurrency)

            pid = start_server(model_path,tp,max_prefill_tokens,max_running_requests,mem_fraction,torch_compile,is_dp,'10.6.131.3:5000',2)
            print(f'server start PID:{pid}')
            with open('tmp.out','a') as fw:
                fw.write(f'max_prefill:{max_prefill_tokens},max_running_requests:{max_running_requests},torch_compile:{torch_compile},is_dp:{is_dp}\n')
            benchmark(num_request,ISL,OSL,concurrency,'tmp.out')
            print('finish 1 benchmark')
        else:
            print('skip')
        n+=1
