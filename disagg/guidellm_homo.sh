#!/bin/bash
#
if [[ ! -f "$(dirname $0)/argparse.bash" ]]; then
  echo 'installing script deps'
  rm -rf argparse*
  wget -q https://raw.githubusercontent.com/nhoffman/argparse-bash/master/argparse.bash
  chmod +x argparse.bash
  sed -i 's/python/python3/g' argparse.bash
fi

MODEL="meta-llama/Llama-3.1-70B-Instruct"
IP=10.254.87.164
PORT=8000

OUTPUT_TOKENS=5000

PARALLEL_RUNS=1

NUM_ITERATIONS=1

start=30000
interval=1000
max=$(( start + ( PARALLEL_RUNS * interval )))
max_window=30000

TOKENIZER_CMD="--processor meta-llama/Llama-3.1-70B-Instruct"

TP=$1

clean_flag=true

if $clean_flag; then
  echo -e "\nOne-time install of benchmark dependencies!"

  echo "Deleting old env and test results folders"
  rm -rf bench_venv out
  clean_flag=false

  echo "Creating venv"
  python3 -m venv bench_venv
  source bench_venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel

  echo "Huggingface login"
  pip install huggingface_hub pandas
  python3 hf.py

  #pip install --force-reinstall git+https://github.com/vllm-project/guidellm.git@v0.6.0
  pip install --force-reinstall guidellm==0.6.0
  pip install "tracerite==1.1.3"
fi

source bench_venv/bin/activate
echo -e "\nRun guidellm"

mkdir -p out
echo "MODEL = $MODEL"

#pip install --force-reinstall git+https://github.com/vllm-project/guidellm.git@v0.6.0

echo "run guidellm"
mkdir -p out
echo $MODEL

for OUTPUT_TOKENS in 5000; do
  #for ((INPUT_TOKENS=$start; INPUT_TOKENS<$max; INPUT_TOKENS+=$interval)); do
  #for INPUT_TOKENS in 4000 10000 16000; do
  #for INPUT_TOKENS in 4000 10000 16000 20000 25000 30000; do
  for INPUT_TOKENS in 4000; do
    if [ ${INPUT_TOKENS} -le ${max_window} ]; then
        echo -e "\n==============================="
        echo -e "\nINPUT TOKENS = ${INPUT_TOKENS}"
        echo -e "\nOUTPUT_TOKENS = ${OUTPUT_TOKENS}"
        mkdir -p out/${INPUT_TOKENS}_${OUTPUT_TOKENS}_homo_dynamo_finetune_tp${TP}
        echo $MODEL > out/${INPUT_TOKENS}_${OUTPUT_TOKENS}_homo_dynamo_finetune_tp${TP}/model
        for i in $(seq 1 $NUM_ITERATIONS); do
          for conn in 128; do
          #for conn in 16 32 64 128; do
          #for conn in 128; do
          #for conn in 1 2 4 8 16 32 64 128; do
              guidellm benchmark \
                  --target "http://$IP:$PORT" \
                  --rate-type concurrent \
                  --rate $conn \
                  --max-requests $((10*conn+50)) \
                  --max-seconds 600 \
                  --data "prompt_tokens=${INPUT_TOKENS},output_tokens=${OUTPUT_TOKENS}" \
                  --output-path=out/${INPUT_TOKENS}_${OUTPUT_TOKENS}_homo_dynamo_finetune_tp${TP}/result_${i}_${conn}.csv
          done
        done
        python3 analyze.py "${INPUT_TOKENS}_${OUTPUT_TOKENS}_homo_dynamo_finetune_tp${TP}" "output_${INPUT_TOKENS}_${OUTPUT_TOKENS}_homo_dynamo_finetune_tp${TP}.csv"
        cp "output_${INPUT_TOKENS}_${OUTPUT_TOKENS}_homo_dynamo_finetune_tp${TP}.csv" out/
    fi
  done
done
