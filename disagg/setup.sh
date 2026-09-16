#!/bin/bash

SETUP_TPYE=$1
MODEL=$2
QUANTIZATION=$3

if [[ $# -ne 3 ]]
then
  echo "usage: $0 [SETUP_TPYE] [MODEL] [QUANTIZATION]"
  echo "     | [SETUP_TPYE] : 4P4D (use 4 GPUs for 1P & 1D each in TP4), 2P6D (use 2 GPUs for 1P & 3D each in TP2), 6P2D (use 2 GPUs for 3P, 1D each in TP2)"
  echo "     | [MODEL] : hf slug (RedHatAI/Qwen3-Next-80B-A3B-Instruct-FP8-dynamic, RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic, RedHatAI/Qwen3.5-9B-FP8-dynamic)"
  echo "     | [QUANTIZATION] : none, -fp8"
  exit 1
fi

echo -e "\nSETUP_TPYE = ${SETUP_TPYE}"
echo -e "MODEL = ${MODEL}"
echo -e "QUANTIZATION = ${QUANTIZATION}"

echo -e "\nStopping prefill & decode instances"
echo -e "Stopping frontend\n"

sudo docker stop vllm-nvidia-gpu-dynamo-frontend 2>/dev/null
sudo docker rm -f vllm-nvidia-gpu-dynamo-frontend 2>/dev/null

sudo docker stop vllm-nvidia-gpu-dynamo-decode1-tp2 vllm-nvidia-gpu-dynamo-decode2-tp2 vllm-nvidia-gpu-dynamo-decode3-tp2 vllm-nvidia-gpu-dynamo-decode1-tp4 2>/dev/null
sudo docker stop vllm-nvidia-gpu-dynamo-prefill1-tp2 vllm-nvidia-gpu-dynamo-prefill2-tp2 vllm-nvidia-gpu-dynamo-prefill3-tp2 vllm-nvidia-gpu-dynamo-prefill1-tp4 2>/dev/null

sudo docker rm -f vllm-nvidia-gpu-dynamo-decode1-tp2 vllm-nvidia-gpu-dynamo-decode2-tp2 vllm-nvidia-gpu-dynamo-decode3-tp2 vllm-nvidia-gpu-dynamo-decode1-tp4 2>/dev/null
sudo docker rm -f vllm-nvidia-gpu-dynamo-prefill1-tp2 vllm-nvidia-gpu-dynamo-prefill2-tp2 vllm-nvidia-gpu-dynamo-prefill3-tp2 vllm-nvidia-gpu-dynamo-prefill1-tp4 2>/dev/null

echo -e "\nStarting nats and etcd"

sudo docker stop docker-nats-server-1 docker-etcd-server-1
sleep 5
sudo docker rm docker-nats-server-1 docker-etcd-server-1
sleep 10
sudo docker compose -f docker/docker-compose_nats_etcd.yml up -d
sleep 5
sudo docker compose -f docker/docker-compose_nats_etcd.yml ps
sleep 2

echo -e "\nStarting prefill & decode instances"

if [ "${SETUP_TPYE}" == "4P4D" ]; then
    bash ./prefill.sh ${MODEL} ${QUANTIZATION} 4 "0,1,2,3" 1 "0-55,112-167"
    sleep 5
    echo
    bash ./decode.sh ${MODEL} ${QUANTIZATION} 4 "4,5,6,7" 1 "56-111,168-223"
    echo
    sleep 5
fi

if [ "${SETUP_TPYE}" == "2P6D" ]; then
    bash ./prefill.sh ${MODEL} ${QUANTIZATION} 2 "0,1" 1 "0-27,112-139"
    sleep 5
    echo
    bash ./decode.sh ${MODEL} ${QUANTIZATION} 2 "2,3" 1 "28-55,140-167"
    sleep 5
    echo
    bash ./decode.sh ${MODEL} ${QUANTIZATION} 2 "4,5" 2 "56-83,168-195"
    sleep 5
    echo
    bash ./decode.sh ${MODEL} ${QUANTIZATION} 2 "6,7" 3 "84-111,196-223"
    echo
    sleep 5 
fi

if [ "${SETUP_TPYE}" == "6P2D" ]; then
    bash ./prefill.sh ${MODEL} ${QUANTIZATION} 2 "0,1" 1 "0-27,112-139"
    sleep 5
    echo
    bash ./prefill.sh ${MODEL} ${QUANTIZATION} 2 "2,3" 2 "28-55,140-167"
    sleep 5
    echo
    bash ./prefill.sh ${MODEL} ${QUANTIZATION} 2 "4,5" 3 "56-83,168-195"
    sleep 5
    echo
    bash ./decode.sh ${MODEL} ${QUANTIZATION} 2 "6,7" 1 "84-111,196-223"
    echo
    sleep 5
fi

echo -e "\nStarting frontend"
bash ./frontend_homogeneous.sh ${MODEL}

sleep 5

echo
sudo docker ps -a

echo -e "\nSetup scripts completed succesfully"
echo -e "\nWait for the containers to fully come up\n"

