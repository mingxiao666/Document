#! /bin/bash
pkill sglang
ssh -p 12322 root@10.6.131.4 "pkill sglang"
ssh -p 12322 root@10.6.131.4 "kill -9 \`lsof -t -i:31000\`"
