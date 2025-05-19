python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-R1 \
    --trust-remote-code \
    --tp-size 8 \
    --max-prefill-tokens 8192 \
    --chunked-prefill-size 2048 \
    --max-running-requests 128 \
    --disable-radix-cache \
    --mem-fraction-static 0.9 \
    --stream-output \
    --attention-backend flashinfer \
    --speculative-algorithm NEXTN --speculative-num-steps  1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2

