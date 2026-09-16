#!/bin/bash
# decode_homogeneous_4P4D_claude.sh — H200 same-node TP4 decode (optimized)

MODEL_NAME="${1:-meta-llama/Llama-3.1-70B-Instruct}"
QUANTIZATION="${2:-fp8}"
QUANT_FLAG=$([ "${QUANTIZATION}" = "none" ] && echo "" || echo "--quantization ${QUANTIZATION}")

TP=4

sudo docker stop vllm-nvidia-gpu-dynamo-decode-tp4 2>/dev/null
sudo docker rm vllm-nvidia-gpu-dynamo-decode-tp4 2>/dev/null

sudo docker run -itd \
--gpus all \
--cpuset-cpus=56-111,168-223 \
--network=host \
--ipc=host \
--ulimit memlock=-1:-1 \
--cap-add=SYS_ADMIN \
--group-add video \
--workdir /root \
--name vllm-nvidia-gpu-dynamo-decode-tp4 \
-e NATS_SERVER=nats://127.0.0.1:4222 \
-e ETCD_ENDPOINTS=http://127.0.0.1:2379 \
-e DYN_DISCOVERY_BACKEND=etcd \
-e DYN_REQUEST_PLANE=tcp \
-e DYN_LOG=INFO \
-e DYN_TCP_RPC_HOST=127.0.0.1 \
-e DYN_DISCOVERY_TTL_S=300 \
-e LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:/usr/lib:/opt/dynamo/lib \
-e OMP_NUM_THREADS=56 \
-e TORCH_COMPILE_DYNAMIC=0 \
-e TORCHINDUCTOR_TRITON_CUDAGRAPHS=1 \
-e TORCHINDUCTOR_MAX_AUTOTUNE=1 \
-e TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0 \
-e TORCHINDUCTOR_EPILOGUE_FUSION=1 \
-e TORCHINDUCTOR_SHAPE_PADDING=1 \
-e NCCL_P2P_LEVEL=NVL \
-e NCCL_SHM_DISABLE=0 \
-e NCCL_DEBUG=INFO \
-e NIXL_LOG_LEVEL=info \
-e NIXL_CONN_TIMEOUT_MS=90000 \
-e NIXL_TRANSFER_TIMEOUT_MS=90000 \
-e NIXL_BACKEND=UCX \
-e UCX_MAX_CHUNK_LEN=4194304 \
-e UCX_MAX_RNDV_RAILS=${TP} \
-e UCX_RNDV_THRESH=0 \
-e UCX_UD_VERBS_ENABLE=n \
-e UCX_MEMTYPE_CACHE=y \
-e UCX_CUDA_IPC_CACHE=1 \
-e UCX_MEMTYPE_REG_WHOLE_ALLOC_TYPES=cuda \
-e UCX_KEEPALIVE_ENABLE=y \
-e UCX_RCACHE_MAX_UNRELEASED=1024 \
-e UCX_TLS=cuda_ipc,cuda_copy,self,tcp \
-e UCX_NET_DEVICES=all \
-e UCX_LOG_LEVEL=INFO \
-e UCX_CUDA_COPY_MAX_REG_RATIO=1.0 \
-e VLLM_LOGGING_LEVEL=INFO \
-e VLLM_USE_V1=1 \
-e VLLM_KV_CACHE_LAYOUT=HND \
-e VLLM_NIXL_ABORT_REQUEST_TIMEOUT=900 \
-e VLLM_NIXL_BACKEND=UCX \
-e VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
-e VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
-e VLLM_NIXL_DEVICE_TO_DEVICE=1 \
-e VLLM_SSM_CONV_STATE_LAYOUT=DS \
-e CUDA_VISIBLE_DEVICES=4,5,6,7 \
-v /home/ebay/.cache/huggingface/hub:/models \
-e HF_HUB_CACHE=/models \
-e HUGGING_FACE_HUB_TOKEN=[HF_TOKEN_PLACEHOLDER] \
-e MODEL_NAME="${MODEL_NAME}" \
-e QUANT_FLAG="${QUANT_FLAG}" \
-e TP="${TP}" \
vllm-nvidia-gpu-dynamo:my_custom_buildtag_v2 \
bash -lc 'python3 -m dynamo.vllm \
    --model ${MODEL_NAME} \
    --served-model-name ${MODEL_NAME} \
    --compilation-config '"'"'{"mode":"VLLM_COMPILE"}'"'"' \
    --is-decode-worker \
    --tensor-parallel-size ${TP} \
    --generation-config vllm \
    --gpu-memory-utilization    0.95 \
    --max-model-len             36864 \
    --max-num-batched-tokens    8192 \
    --max-num-seqs              128 \
    --block-size                128 \
    ${QUANT_FLAG} \
    --kv-cache-dtype 		fp8 \
    --disable-log-stats \
    --generation-config vllm \
    --no-enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config '"'"'{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda","enable_permute_local_kv":true,"kv_connector_extra_config":{"enforce_handshake_compat":false,"agreed_block_size":128}}'"'"''
