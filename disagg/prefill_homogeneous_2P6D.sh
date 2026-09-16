#!/bin/bash
# prefill_homogeneous_4P4D_claude.sh — H200 same-node TP4 prefill (optimized)

MODEL_NAME="${1:-meta-llama/Llama-3.1-70B-Instruct}"
QUANTIZATION="${2:-fp8}"
QUANT_FLAG=$([ "${QUANTIZATION}" = "none" ] && echo "" || echo "--quantization ${QUANTIZATION}")

TP=2

sudo docker stop vllm-nvidia-gpu-dynamo-prefill-tp2 2>/dev/null
sudo docker rm vllm-nvidia-gpu-dynamo-prefill-tp2 2>/dev/null

sudo docker run -itd \
--gpus all \
--cpuset-cpus=0-27,112-139 \
--network=host \
--ipc=host \
--cap-add=SYS_NICE \
--cap-add=SYS_PTRACE \
--cap-add=SYS_ADMIN \
--group-add video \
--workdir /root \
--name vllm-nvidia-gpu-dynamo-prefill-tp2 \
-e NATS_SERVER=nats://127.0.0.1:4222 \
-e ETCD_ENDPOINTS=http://127.0.0.1:2379 \
-e DYN_DISCOVERY_BACKEND=etcd \
-e DYN_REQUEST_PLANE=tcp \
-e DYN_LOG=INFO \
-e DYN_DISCOVERY_TTL_S=300 \
-e DYN_TCP_RPC_HOST=127.0.0.1 \
-e LD_LIBRARY_PATH=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:/usr/lib:/opt/dynamo/lib \
-e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
-e VLLM_RPC_TIMEOUT=10000000 \
-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
-e VLLM_DELAYED_SAMPLING=true \
-e OMP_NUM_THREADS=56 \
-e TORCH_COMPILE_DYNAMIC=0 \
-e TORCHINDUCTOR_TRITON_CUDAGRAPHS=0 \
-e TORCHINDUCTOR_MAX_AUTOTUNE=0 \
-e TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0 \
-e TORCHINDUCTOR_EPILOGUE_FUSION=1 \
-e TORCHINDUCTOR_SHAPE_PADDING=1 \
-e VLLM_EXPONENTIAL_BUCKETING=true \
-e VLLM_PROMPT_SEQ_BUCKET_STEP=128 \
-e NCCL_P2P_LEVEL=NVL \
-e NCCL_SHM_DISABLE=0 \
-e NCCL_DEBUG=INFO \
-e NIXL_LOG_LEVEL=INFO \
-e NIXL_CONN_TIMEOUT_MS=120000 \
-e NIXL_TRANSFER_TIMEOUT_MS=120000 \
-e NIXL_BACKEND=UCX \
-e UCX_MAX_CHUNK_LEN=4194304 \
-e UCX_MAX_RNDV_RAILS="${TP}" \
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
-e VLLM_NIXL_BACKEND=UCX \
-e VLLM_LOGGING_LEVEL=INFO \
-e VLLM_USE_V1=1 \
-e VLLM_KV_CACHE_LAYOUT=HND \
-e VLLM_HOST_IP=127.0.0.1 \
-e VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
-e VLLM_NIXL_SIDE_CHANNEL_PORT=20096 \
-e VLLM_NIXL_DEVICE_TO_DEVICE=1 \
-e VLLM_SSM_CONV_STATE_LAYOUT=DS \
-e VLLM_SKIP_P2P_CHECK=1 \
-e VLLM_NIXL_ABORT_REQUEST_TIMEOUT=900 \
-e CUDA_VISIBLE_DEVICES=0,1 \
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
    --is-prefill-worker \
    --tensor-parallel-size ${TP} \
    --gpu-memory-utilization    0.95 \
    --max-model-len             36864 \
    --max-num-batched-tokens    16384 \
    --max-num-seqs              128 \
    --block-size                128 \
    ${QUANT_FLAG} \
    --kv-cache-dtype            fp8 \
    --disable-log-stats \
    --generation-config vllm \
    --no-enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config '"'"'{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda","enable_permute_local_kv":true,"kv_connector_extra_config":{"enforce_handshake_compat":false,"agreed_block_size":128}}'"'"''
