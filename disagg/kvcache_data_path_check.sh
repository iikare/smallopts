#!/bin/bash

instance=$1

# - Decode

if [ "${instance}" == "decode" ]; then
  for C in vllm-nvidia-gpu-dynamo-decode1-tp2 vllm-nvidia-gpu-dynamo-decode2-tp2 vllm-nvidia-gpu-dynamo-decode3-tp2 vllm-nvidia-gpu-dynamo-decode1-tp4; do
    echo "--- $C ---"
    sudo docker logs $C 2>&1 | grep "generation throughput" | tail -10
    echo 
    sudo docker logs $C 2>&1 | grep "KV Transfer metrics" | tail -10
    echo
  done

  echo -e "\nDeocde"
  echo -e "generation throughput = tokens/s the decode is generating (output tokens)"
  echo -e "prompt throughput on a decode instance is near 0 (decode doesn't do prefill)"
  echo -e "Running: 8 reqs = concurrent requests being served"
  echo -e "External prefix cache hit rate: 99.9% = KV cache arriving from prefill via NIXL (not recomputed locally)\n"

fi

# - Prefill

if [ "${instance}" == "prefill" ]; then
  for C in vllm-nvidia-gpu-dynamo-prefill1-tp2 vllm-nvidia-gpu-dynamo-prefill2-tp2 vllm-nvidia-gpu-dynamo-prefill3-tp2 vllm-nvidia-gpu-dynamo-prefill1-tp4; do
    echo "--- $C ---"
    sudo docker logs $C 2>&1 | grep "generation throughput" | grep -v "0.0 tokens" | tail -10
    echo
  done

  echo -e "\nPrefill"
  echo -e "prompt throughput = tokens/s the prefill is processing (input tokens being prefilled)"
  echo -e "generation throughput on a prefill instance is near 0 (prefill doesn't generate, decode does)\n"
fi
