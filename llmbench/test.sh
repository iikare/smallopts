#!/bin/bash


if [[ ! -f "$(dirname $0)/argparse.bash" ]]; then
  echo 'installing script deps'
  rm -rf argparse*
  wget -q https://raw.githubusercontent.com/nhoffman/argparse-bash/master/argparse.bash
  chmod +x argparse.bash
  sed -i 's/python/python3/g' argparse.bash
fi

ARGPARSE_DESCRIPTION="llmbench utility for inference benchmarking"
source $(dirname $0)/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1
parser.add_argument('-g', '--guidellm', action='store_true',
                    help='run guidellm (client)')
parser.add_argument('-v', '--vllm', action='store_true',
                    help='run vllm (server) - requires --model parameter')
parser.add_argument('-s', '--sglang', action='store_true',
                    help='run sglang (server) - requires --model parameter')
parser.add_argument('-m', '--model', type=str,
                    help='model (hugging face slug)')
parser.add_argument('-q', '--quantization', type=str, default='None',
                    help='quantization [default %(default)s]')
parser.add_argument('-i', '--input',  default=1000, type=int,
                    help='input token size [default %(default)s]')
parser.add_argument('-o', '--output',  default=500, type=int,
                    help='output token size [default %(default)s]')
parser.add_argument('--tp', default=8, type=int,
                    help='tensor parallel [default %(default)s]')
parser.add_argument('--dp', default=1, type=int,
                    help='data parallel size [default %(default)s]')
parser.add_argument('--port', default=8000, type=int,
                    help='port [default %(default)s]')
parser.add_argument('--install_only', action='store_true', default=False,
                    help='install dependencies without running any tests')
parser.add_argument('--offline', action='store_true',
                    help='use existing venv @ ./bench_venv && existing model @ HF_HOME (implies --skip_venv_creation)')
parser.add_argument('--skip_venv_creation', action='store_true',
                    help='use existing venv @ ./bench_venv')
parser.add_argument('--sleep', action='store_true', default=False,
                    help='wait for model ready instead of using hardcoded sleep time')
parser.add_argument('--lmcache', action='store_true',
                    help='use lmcache and start lmcache server before creating test')
parser.add_argument('--hf_hub', type=str, default='~/.cache/huggingface',
                    help='location of huggingface data [default %(default)s]')
parser.add_argument('--tokenizer_dir', type=str, default='',
                    help='tokenizer location for specified model (required for --offline)')
parser.add_argument('--num_iterations', default=1, type=int,
                    help='guidellm repeat iterations [default %(default)s]')
parser.add_argument('--custom_args', type=str, default='',
                    help='quote-delimited string to pass directly to server instance')
EOF

if [[ ! $GUIDELLM && ! $VLLM && ! $SGLANG ]]; then
  echo "script requires -g and/or one of -v/-s flags to be set"
  echo "use -h for help"
  exit 1
fi

if [[ $VLLM && $SGLANG ]]; then
  echo "cannot run both vllm and sglang, choose one server only"
  exit 1
fi

# set server string
SERVER_STR='vllm'
if [[ $SGLANG ]]; then
  SERVER_STR='sglang'
fi

if [[ $VLLM || $SGLANG ]]; then
  # ensure model slug is set if using server
  if [[ ! $MODEL ]]; then
    if [[ ! $INSTALL_ONLY ]]; then
      echo "model slug not set while using server flag -v/-s, aborting!"
      exit 1
    fi
  fi
  
  # validate TP option only if using server
  if [[ "$TP" -ne 1 && "$TP" -ne 2 && "$TP" -ne 4 && "$TP" -ne 8 ]]; then
    echo "bad tensor-parallel-size: $TP - only 1, 2, 4, or 8 are supported"
    exit 1
  fi
fi

DP_ARG=""
if [[ "$DP" -gt 1 ]]; then
  DP_ARG="--data-parallel-size $DP"
fi

export HF_HOME=$HF_HUB
TOKENIZER_CMD=""
if [[ $OFFLINE ]]; then
  export HF_HUB_OFFLINE=1
  SKIP_VENV_CREATION=true

  # guidellm requires tokenizer directory manually specified if HF_HUB_OFFLINE set
  if [[ "${TOKENIZER_DIR}" == "" ]]; then
    echo "--offline mode requires --tokenizer_dir set, exiting!"
    exit 1
  else
    # insert string into guidellm launch command as needed
    TOKENIZER_CMD="--processor \"${TOKENIZER_DIR}\""
    echo "using tokenizer option: ${TOKENIZER_CMD}"
  fi
fi

# detect model size and extend launch time
SLEEP_TIME="5m"

if [[ ! $SKIP_VENV_CREATION ]]; then
  echo "reset"
  rm -rf bench_venv guidellm
fi

if [[ ! $SKIP_VENV_CREATION ]]; then
  echo "creating venv"
  python3 -m venv bench_venv
  source bench_venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel

  echo "huggingface login"
  pip install huggingface_hub pandas
  python3 hf.py

  if [[ $VLLM || $SGLANG ]]; then
    echo "server installation"
    if [[ $LMCACHE ]]; then
      pip install lmcache
    fi
    if [[ $VLLM ]]; then
      pip install accelerate bitsandbytes vllm==0.23.0
    elif [[ $SGLANG ]]; then
      pip install accelerate sglang
    fi
  fi

  if [[ $GUIDELLM ]]; then
    echo "client installation"
    pip install guidellm==0.6.0
    #pip install git+https://github.com/vllm-project/guidellm.git
  fi

  if [[ $SGLANG ]]; then
    echo "client installation"
    pip install sglang
  fi
else 
  source bench_venv/bin/activate
fi

if [[ $INSTALL_ONLY ]]; then
  echo "installation done - dry-run complete, exiting"
  exit
fi

if [[ $VLLM || $SGLANG ]]; then
  if [[ $LMCACHE ]]; then
    echo "starting lmcache server"
    if [[ "$MODEL" == *"80b"* || "$MODEL" == *"80B"* ]]; then
      if [[ "$TP" -eq 8 ]]; then
        ATTN_BLOCK_SIZE="144"
      elif [[ "$TP" -eq 4 ]]; then
        ATTN_BLOCK_SIZE="272"
      elif [[ "$TP" -eq 2 ]]; then
        ATTN_BLOCK_SIZE="544"
      else
        ATTN_BLOCK_SIZE="544"
      fi
    elif [[ "$MODEL" == *"35b"* || "$MODEL" == *"35B"* ]]; then
      if [[ "$TP" -eq 8 ]]; then
        ATTN_BLOCK_SIZE="272"
      elif [[ "$TP" -eq 4 ]]; then
        ATTN_BLOCK_SIZE="528"
      elif [[ "$TP" -eq 2 ]]; then
        ATTN_BLOCK_SIZE="1056"
      else
        ATTN_BLOCK_SIZE="1056"
      fi
    elif [[ "$MODEL" == *"9b"* || "$MODEL" == *"9B"* ]]; then
      if [[ "$TP" -eq 8 ]]; then
        ATTN_BLOCK_SIZE="272"
      else
        ATTN_BLOCK_SIZE="528"
      fi
    else
      echo "invalid model $MODEL"
      exit 1
    fi


    export LMCACHE_CHUNK_SIZE=$ATTN_BLOCK_SIZE
    lmcache server \
    --l1-size-gb 200 --eviction-policy LRU --chunk-size $ATTN_BLOCK_SIZE --separate-object-groups & # as per test plan
    lmcache_pid=$(echo $!)
    CUSTOM_ARGS="--max-num-batched-tokens $ATTN_BLOCK_SIZE --max-model-len 32768"
  fi
  if [[ "$MODEL" == *"80b"* || "$MODEL" == *"80B"* ]]; then
    if [[ $tp == 8 ]]; then
      CUSTOM_ARGS+="--enable_expert_parallel "
    fi
  fi
fi

if [[ $VLLM ]]; then
  if [[ "$DP" -gt 1 ]]; then
    CUSTOM_ARGS+=' --max-num-seqs 128 --max-model-len 32768 --max-num-batched-tokens 32768'
  fi
  echo "run $SERVER_STR"
  if [[ $GUIDELLM ]]; then
    if [[ $LMCACHE ]]; then
      python3 -m vllm.entrypoints.openai.api_server --disable-log-stats --quantization $QUANTIZATION --tensor-parallel-size $TP $DP_ARG --model=$MODEL --port $PORT $CUSTOM_ARGS --mamba-cache-mode align --enable-prefix-caching --kv-transfer-config \
    '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config":{"kv_load_failure_policy":"ignore", "enable_async_put": false}}' &

    else
      python3 -m vllm.entrypoints.openai.api_server --disable-log-stats --quantization $QUANTIZATION --tensor-parallel-size $TP $DP_ARG --model=$MODEL --port $PORT $CUSTOM_ARGS &
    fi
    wait_pid=$(echo $!)
  else
    if [[ "$QUANTIZATION" == "None" ]]; then
      QUANTIZATION='none'
    elif [[ "$QUANTIZATION" == "fp8" ]]; then
      QUANTIZATION='fp8'
    fi
    if [[ $LMCACHE ]]; then
      python3 -m vllm.entrypoints.openai.api_server --disable-log-stats --quantization $QUANTIZATION --tensor-parallel-size $TP $DP_ARG --model=$MODEL --port $PORT $CUSTOM_ARGS --mamba-cache-mode align --enable-prefix-caching --kv-transfer-config \
    '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'
    else
      python3 -m vllm.entrypoints.openai.api_server --disable-log-stats --quantization $QUANTIZATION --tensor-parallel-size $TP $DP_ARG --model=$MODEL --port $PORT $CUSTOM_ARGS
    fi
  fi
fi

if [[ $VLLM || $SGLANG ]]; then
  if [[ $GUIDELLM ]]; then
    if [[ ! $SLEEP ]]; then
      while [[ $(curl -s http://localhost:$PORT/v1/models | jq '(.data // .list // []) | length' 2>/dev/null) -eq 0 ]]; do
        sleep 0.1
      done
      echo "detected model ready - inference begins!"
    else
      echo "sleeping for $SLEEP_TIME while server initializes!"
      sleep $SLEEP_TIME
    fi
  fi
fi

if [[ $GUIDELLM && $LMCACHE ]]; then
  echo "warming up lmcache..."
  guidellm benchmark \
    --target "http://localhost:$PORT" \
    --rate-type concurrent \
    --rate 64 \
    --max-requests 50 \
    --data "prompt_tokens=$INPUT,output_tokens=$OUTPUT" \
    $TOKENIZER_CMD \
    --output-path=warmup.csv
  echo "lmcache warmup complete"
fi

if [[ $GUIDELLM ]]; then
  echo "run guidellm"
  rm -rf out output.csv
  mkdir -p out
  echo $MODEL > out/model
  MAX_REQUESTS=500
  if [[ $LMCACHE ]]; then
    MAX_REQUESTS=100
  fi
  for i in $(seq 1 $NUM_ITERATIONS); do
    rm -f out/result*
    for conn in 8 16 32 64; do # see test plan for details on concurrency and max requests
    #for conn in 64; do # see test plan for details on concurrency and max requests
      guidellm benchmark \
        --target "http://localhost:$PORT" \
        --rate-type concurrent \
        --rate $conn \
        --max-requests $MAX_REQUESTS \
        --data "prompt_tokens=$INPUT,output_tokens=$OUTPUT" \
        $TOKENIZER_CMD \
        --output-path=out/result_${i}_${conn}.csv 
    done
  done
  python3 analyze.py
fi

if [[ $VLLM || $SGLANG ]]; then
  echo "shutting down $SERVER_STR and lmcache"
  pkill -9 -P $wait_pid 2>/dev/null || true
  kill -9 $wait_pid 2>/dev/null || true
  pkill -9 -f "VLLM::" 2>/dev/null || true
  pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
  kill -9 $lmcache_pid 2>/dev/null || true
  pkill -9 lmcache 2>/dev/null || true
  sleep 15
fi
