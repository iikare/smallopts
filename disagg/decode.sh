#!/bin/bash
# decode.sh — Generic H200 decode script (TP2 or TP4, single or multiple instances)
#
# Usage: bash decode.sh [MODEL_NAME] [QUANTIZATION] [TP] [CUDA_DEVS] [DECODE_ID] [CPUSET]
#
#   MODEL_NAME:   HuggingFace model ID         (default: meta-llama/Llama-3.1-70B-Instruct)
#   QUANTIZATION: fp8 | none                   (default: fp8; use "none" for pre-quantized FP8)
#   TP:           Tensor parallel size          (default: 4)
#   CUDA_DEVS:    GPU assignment                (default: 4,5,6,7 for TP4; use 2,3 / 4,5 / 6,7 for TP2)
#   DECODE_ID:    Instance ID for multi-decode  (default: 1) — sets container name & side channel port
#   CPUSET:       CPU pinning                   (default: 56-111,168-223 for TP4 / 28-55,140-167 for TP2)
#
# Examples:
#   # 1P1D TP4
#   bash decode.sh "meta-llama/Llama-3.1-70B-Instruct" fp8 4 "4,5,6,7" 1 "56-111,168-223"
#
#   # 1P3D TP2 (3 decode instances)
#   bash decode.sh "RedHatAI/Qwen3-Next-80B-A3B-Instruct-FP8" none 2 "2,3" 1 "28-55,140-167"
#   bash decode.sh "RedHatAI/Qwen3-Next-80B-A3B-Instruct-FP8" none 2 "4,5" 2 "56-83,168-195"
#   bash decode.sh "RedHatAI/Qwen3-Next-80B-A3B-Instruct-FP8" none 2 "6,7" 3 "84-111,196-223"
#
#   # 3P1D TP2 (single decode)
#   bash decode.sh "ModelName" none 2 "6,7" 1 "84-111,196-223"

MODEL_NAME="${1:-meta-llama/Llama-3.1-70B-Instruct}"
QUANTIZATION="${2:-fp8}"
TP="${3:-4}"
CUDA_DEVS="${4:-4,5,6,7}"
DECODE_ID="${5:-1}"
CPUSET="${6:-56-111,168-223}"

QUANT_FLAG=$([ "${QUANTIZATION}" = "none" ] && echo "" || echo "--quantization ${QUANTIZATION}")
CONTAINER_NAME="vllm-nvidia-gpu-dynamo-decode${DECODE_ID}-tp${TP}"
SIDE_CHANNEL_PORT=$((20100 + DECODE_ID))

sudo docker stop ${CONTAINER_NAME} 2>/dev/null
sudo docker rm   ${CONTAINER_NAME} 2>/dev/null

sudo docker run -itd \
--gpus all \
--cpuset-cpus=${CPUSET} \
--network=host \
--ipc=host \
--pid=host \
--ulimit nofile=65535:65535 \
--cap-add=SYS_NICE \
--cap-add=SYS_PTRACE \
--cap-add=SYS_ADMIN \
--group-add video \
--workdir /root \
--name ${CONTAINER_NAME} \
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
-e OMP_NUM_THREADS=28 \
-e TORCH_COMPILE_DYNAMIC=0 \
-e TORCHINDUCTOR_TRITON_CUDAGRAPHS=0 \
-e TORCHINDUCTOR_MAX_AUTOTUNE=1 \
-e TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0 \
-e TORCHINDUCTOR_EPILOGUE_FUSION=1 \
-e TORCHINDUCTOR_SHAPE_PADDING=1 \
-e VLLM_EXPONENTIAL_BUCKETING=true \
-e VLLM_PROMPT_SEQ_BUCKET_STEP=128 \
-e NCCL_P2P_LEVEL=NVL \
-e NCCL_SHM_DISABLE=0 \
-e NCCL_DEBUG=INFO \
-e NIXL_LOG_LEVEL=INFO \
-e NIXL_TELEMETRY_EXPORTER=NOP \
-e NIXL_CONN_TIMEOUT_MS=120000 \
-e NIXL_TRANSFER_TIMEOUT_MS=120000 \
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
-e UCX_CUDA_ASYNC_EVENTS=n \
-e UCX_CUDA_COPY_MAX_REG_RATIO=1.0 \
-e VLLM_NIXL_BACKEND=UCX \
-e VLLM_LOGGING_LEVEL=INFO \
-e VLLM_USE_V1=1 \
-e VLLM_KV_CACHE_LAYOUT=HND \
-e VLLM_HOST_IP=127.0.0.1 \
-e VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
-e VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
-e VLLM_NIXL_DEVICE_TO_DEVICE=1 \
-e VLLM_SSM_CONV_STATE_LAYOUT=DS \
-e VLLM_SKIP_P2P_CHECK=1 \
-e VLLM_NIXL_ABORT_REQUEST_TIMEOUT=900 \
-e CUDA_VISIBLE_DEVICES=${CUDA_DEVS} \
-v /home/ebay/.cache/huggingface/hub:/models \
-e HF_HUB_CACHE=/models \
-e HUGGING_FACE_HUB_TOKEN=[HF_TOKEN_PLACEHOLDER] \
-e MODEL_NAME="${MODEL_NAME}" \
-e QUANT_FLAG="${QUANT_FLAG}" \
-e TP="${TP}" \
vllm-nvidia-gpu-dynamo:my_custom_buildtag_vtest_homo \
bash -lc 'python3 -m dynamo.vllm \
    --model ${MODEL_NAME} \
    --served-model-name ${MODEL_NAME} \
    --compilation-config '"'"'{"mode":"VLLM_COMPILE"}'"'"' \
    --is-decode-worker \
    --tensor-parallel-size ${TP} \
    --gpu-memory-utilization    0.90 \
    --max-model-len             36864 \
    --max-num-batched-tokens    8192 \
    --max-num-seqs              128 \
    --block-size                512 \
    ${QUANT_FLAG} \
    --kv-cache-dtype            fp8 \
    --disable-log-stats \
    --generation-config vllm \
    --no-enable-prefix-caching \
    --no-disable-hybrid-kv-cache-manager \
    --kv-transfer-config '"'"'{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda","enable_permute_local_kv":true,"kv_connector_extra_config":{"enforce_handshake_compat":false,"agreed_block_size":512}}'"'"''
