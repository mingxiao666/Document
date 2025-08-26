srun --jobid=819242 --overlap --container-image=gitlab-master.nvidia.com/dl/ai-dynamo/dynamo:4cbd4f38810f152c81f15f1cc7cc600c354587aa-32804297-tensorrtllm-arm64  --container-mounts=/lustre/fsw/general_sa/scripts:/scripts,/home/minih:/mnt,/lustre/share/coreai_dlalgo_ci/artifacts/model/deepseek-r1_pyt/:/deepseek-r1_pyt  --mpi pmix  bash -c "
cd /mnt/pd-0825-perf 
pip install sglang==0.5.1.post2 pybase64 
cp bench_serving.py /usr/local/lib/python3.12/dist-packages/sglang/bench_serving.py
python benchonly1p1d.py
"
