Start nats and etcd services:\
sudo docker compose -f docker/docker-compose_nats_etcd.yml up -d \
sudo docker compose -f docker/docker-compose_nats_etcd.yml ps

Create Base: \
sudo docker build -t vllm-nvidia-gpu:my_custom_buildtag_v1 \
    				-f docker/Dockerfile_v1.gpu . 2>&1 | tee DockerBuild_gpu.log

Create Dynamo capable image: \
sudo docker build -t vllm-nvidia-gpu-dynamo:my_custom_buildtag_v1 \
        --ulimit nofile=65535:65535  --build-arg BASE_IMAGE=vllm-nvidia-gpu:my_custom_buildtag_v1 \
        -f docker/Dockerfile.dynamo_v1_gpu . 2>&1 | tee DockerBuild_dynamo_gpu.log

Manual Cleanup: \
sudo rm -rf /ebay/home/ebay/.cache/flashinfer/ \
sudo rm -rf /dev/shm/vllm_* \
sudo rm -rf /dev/shm/vllm* \
sudo rm -rf /dev/shm/sem.vllm* \
sudo rm -rf /tmp/vllm_* \
sudo rm -rf /dev/shm/nccl-* \
sudo rm -rf  /dev/shm/nixl_*

Manual Monitor: \
nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm \
  --format=csv -l 1

Start prefill instance: \
bash ./prefill_homogeneous_4P4D.sh

Start decode instance: \
bash ./decode_homogeneous_4P4D.sh

Start frontend: \
bash ./frontend_homogeneous.sh

Sample Query: \
Internal IP: \
PROMPT="Summarize the following history of Everest: $(printf 'Mount Everest is the highest mountain on Earth. Exploration is the engine of human progress and knowledge. %.0s' {1..210})"; curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg p "$PROMPT" '{
      model: "meta-llama/Llama-3.1-70B-Instruct",
      prompt: $p,
      min_tokens: 5000,
      max_tokens: 5000,
      temperature: 0 }')" | jq .

External IP: \
PROMPT="Summarize the following history of Everest: $(printf 'Mount Everest is the highest mountain on Earth. Exploration is the engine of human progress and knowledge. %.0s' {1..210})"; curl http://10.254.87.164:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg p "$PROMPT" '{
      model: "meta-llama/Llama-3.1-70B-Instruct",
      prompt: $p,
      min_tokens: 5000,
      max_tokens: 5000,
      temperature: 0 }')" | jq .

Test file: \
python3 ./query.py

Intel Documentation for Reference: \
https://docs.google.com/document/d/18kKQgJb_Gui5RkPq4C4plGQ_8cjRwo48/edit#heading=h.fcid1l2c3ewy

Versions working/compatibility: \
vllm: 0.20.1 => 0.23.0\
NIXL: 0.10.0 => 1.3.0\
UCX: 1.20.1 => 1.21.0 \
Dynamo: 1.2.1 \
RUST: 1.90.1 \
NATS: 2.11.4 \
ETCD: 3.6.1 \
CUDA: 13.1 (13.1.2) \
nvidia driver: 590.48.01 \
nvidia fabricmanager: 590.48.01 \
GUIDELLM: 0.6.0
