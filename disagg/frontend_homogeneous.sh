#!/bin/bash
# Usage: bash frontend_homogeneous.sh [MODEL_NAME]
MODEL_NAME="${1:-meta-llama/Llama-3.1-70B-Instruct}"

sudo docker stop vllm-nvidia-gpu-dynamo-frontend 2>/dev/null
sudo docker rm vllm-nvidia-gpu-dynamo-frontend 2>/dev/null
sleep 5

sudo docker run --network host --rm -itd \
--name vllm-nvidia-gpu-dynamo-frontend \
-e DYN_DISCOVERY_BACKEND=etcd \
-e DYN_REQUEST_PLANE=tcp \
-e DYN_EVENT_PLANE=nats \
-e DYN_ROUTER_MODE=kv \
-e DYN_ENABLE_DISAGGREGATED_SERVING=true \
-e HUGGING_FACE_HUB_TOKEN=[HF_TOKEN_PLACEHOLDER] \
-e DYN_LOAD_FORMAT=safetensors \
-e ETCD_ENDPOINTS=http://127.0.0.1:2379 \
-e NATS_SERVER=nats://127.0.0.1:4222 \
nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1 \
bash -lc "python3 -m dynamo.frontend --model-name ${MODEL_NAME} --http-port 8000"
