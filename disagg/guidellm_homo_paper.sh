#!/bin/bash

if [[ $# -ne 4 ]]
then
  echo "usage: $0 [MODEL] [VERSION] [TP] [CLEANUP YES/NO]"
  echo "     | [MODEL] : HF slug (meta-llama/Llama-3.1-70B-Instruct, RedHatAI/Qwen3-Next-80B-A3B-Instruct-FP8, RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic, RedHatAI/Qwen3.5-9B-FP8-dynamic)"
  echo "     | [SETUP] : Setup type (2P6D, 4P4D, 6P2D)"
  echo "     | [TP] : Tensor Parallelism (1,2,4,8. - int only)"
  echo "     | [CLEANUP YES/NO] : Clean up of test env., defaults to NO"
  exit 1
fi

echo -e "\nStart date/time:" 
date

CLEANUP="NO"
clean_flag=false

MODEL=$1
SETUP=$2
TP=$3
CLEANUP=$4

MODEL_DIR=`echo "${MODEL}" | tr "/" "_"`

# detect model suffix
CSV_PREFIX="h200"
if [[ "$MODEL" == *"80b"* || "$MODEL" == *"80B"* ]]; then
  CSV_PREFIX+="_80b"
elif [[ "$MODEL" == *"35b"* || "$MODEL" == *"35B"* ]]; then
  CSV_PREFIX+="_35b"
elif [[ "$MODEL" == *"9b"* || "$MODEL" == *"9B"* ]]; then
  CSV_PREFIX+="_9b"
elif [[ "$MODEL" == *"70b"* || "$MODEL" == *"70B"* ]]; then
  CSV_PREFIX+="_70b"
else
  echo "Could not autodetect model size!"
  CSV_PREFIX+="_unknown_size"
fi

CSV_PREFIX+="_fp8"

PORT=8000
TOKENIZER_CMD="--processor $MODEL"

if [[ "$CLEANUP" == "YES" ]]; then
  clean_flag=true
fi

echo "Begin testing!"
for token_pair in "1000/500" "5000/2500" "10000/5000" "20000/5000"; do
#for token_pair in "10000/5000" "20000/5000"; do	
    INPUT_TOKENS="${token_pair%%/*}"
    OUTPUT_TOKENS="${token_pair##*/}"

    RUN_ID="${CSV_PREFIX}_${INPUT_TOKENS}_${OUTPUT_TOKENS}_${SETUP}_tp${TP}"
    CSV_ID=${RUN_ID}

    if $clean_flag; then
      echo -e "\nOne-time install of benchmark dependencies!"

      echo "Deleting old env and test results folders"
      rm -rf bench_venv out test_results
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
    echo "RUN_ID = ${RUN_ID}"

    mkdir -p out test_results
    echo "MODEL = $MODEL"

    mkdir -p out/${INPUT_TOKENS}_${OUTPUT_TOKENS}_homo_dynamo_finetune_tp${TP}
    echo $MODEL > out/${INPUT_TOKENS}_${OUTPUT_TOKENS}_homo_dynamo_finetune_tp${TP}/model
    for conn in 8 16 32 64; do 
        guidellm benchmark \
          --target "http://localhost:$PORT" \
          --rate-type concurrent \
          --rate $conn \
          --max-requests 5000 \
          --data "prompt_tokens=${INPUT_TOKENS},output_tokens=${OUTPUT_TOKENS}" \
          --output-path=out/${INPUT_TOKENS}_${OUTPUT_TOKENS}_homo_dynamo_finetune_tp${TP}/result_${conn}.csv
    done
    echo "Concatenating results"
    python3 analyze.py "${INPUT_TOKENS}_${OUTPUT_TOKENS}_homo_dynamo_finetune_tp${TP}" "test_results/results_${RUN_ID}.csv"
done

mkdir -p ${SETUP}_tp${TP}_${MODEL_DIR}
mv out test_results ${SETUP}_tp${TP}_${MODEL_DIR}

echo -e "\nEnd date/time:"
date

echo -e "\nRun complete!"
