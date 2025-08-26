import os
import subprocess
import time
import pickle
import numpy as np
import io,sys
import requests


def benchmark(num_prompt,ISL,OSL,max_concurrency,output_file):
    test_cmd = [
                'python3','-m','sglang.bench_serving','--backend','sglang-oai-chat',
                '--dataset-name','random', '--model', 'hf-574fdb8-nim_fp4',
                '--num-prompt',
                f'{num_prompt}',
                '--random-input',
                f'{ISL}',
                '--random-output',
                f'{OSL}',
                '--max-concurrency',
                f'{max_concurrency}',
                '--random-range-ratio','1.0', '--host', '0.0.0.0', '--port', '8000', '--output-file',f'{output_file}'
                ]
    subprocess.run(test_cmd,env=os.environ.copy())

input_output = [
    [4000,1000],
    [6000,1000]
]
concurrencies = [1,2,4,8,16,32,64,128]

n=0
pid = -1
skip = 0
for ISL,OSL in input_output:
    for concurrency in concurrencies:
        num_requests = concurrency * 4
        if n >= skip:
            time.sleep(5)
            with open('tmp-4k6k-bothnodp.1p1d.out','a') as fw:
                fw.write(f'max_prefill:8192,max_running_requests:128,torch_compile:False,is_dp:False\n')
            benchmark(num_requests,ISL,OSL,concurrency,'tmp-4k6k-bothnodp.1p1d.out')
            print('finish 1 benchmark')
        else:
            print('skip')
        n+=1


