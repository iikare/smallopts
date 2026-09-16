#!/bin/bash

# naming scheme
# * iterate over pairs of input/output token counts as a set (signifier - {n1}_{n2})
# * limit access to only 3 models as specified in test plan (signifier - 9b/35b/80b)
# * fix GPU id to h200 (signifier - h200)
# * fix quantization to fp8 (signifier - fp8)
# * force iteration over TP (signifier - tp{n})
# * allow version specifier as input var (signifier - v{n})
#
# NOTE: for disaggregated mode, use tp{n1}{n2} naming scheme instead (will need changes to script)
# final signifier - h200_{n}b_tp{n}_fp8_{n1}_{n2}_v{n}
#

if [[ $# -ne 2 && $# -ne 3 ]]
then
  echo "usage: $0 [MODEL] [LMCACHE] [VERSION]"
  echo "     | [MODEL]    : hf slug (RedHatAI/Qwen3-Next-80B-A3B-Instruct-FP8-dynamic, RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic, RedHatAI/Qwen3.5-9B-FP8-dynamic)"
  echo "     | [LMCACHE]  : use lmcache [Y/n]"
  echo "     | [VERSION]  : version tag (int, or omit when --lmcache; 'lmcache' tag set automatically)"
  exit 1
fi

MODEL=$1
LMCACHE=$2
VERSION_TAG=$3

if [[ $LMCACHE == "y" || $LMCACHE == "Y" ]]; then
  VERSION_TAG="lmcache"
elif [[ $# -eq 3 ]]; then
  re='^[0-9]+$'
  if ! [[ $VERSION_TAG =~ $re ]]; then
     echo "error: version tag must be integer"
     exit 1
  fi
fi

# clean up
rm -rf test_results
mkdir test_results

# detect model suffix
CSV_PREFIX="h200"
if [[ "$MODEL" == *"80b"* || "$MODEL" == *"80B"* ]]; then
  CSV_PREFIX+="_80b"
elif [[ "$MODEL" == *"35b"* || "$MODEL" == *"35B"* ]]; then
  CSV_PREFIX+="_35b"
elif [[ "$MODEL" == *"9b"* || "$MODEL" == *"9B"* ]]; then
  CSV_PREFIX+="_9b"
else
  echo "could not autodetect model size!"
  CSV_PREFIX+="_unknown_size"
fi

# detect fp8
#if [[ "$MODEL" == *"fp8"* || "$MODEL" == *"FP8"* ]]; then
CSV_PREFIX+="_fp8"
#fi

first_flag=true
DP=1 
# run suite
echo "begin testing!"
#for tp in 1 2 4 8; do
#for tp in 1; do
#  DP=8
#for tp in 2; do
#  DP=4
for tp in 4; do
  DP=2
#for tp in 1 2 4; do
#for tp in 4 8; do
#for tp in 2; do
#for tp in 4; do
#for tp in 8; do
  for token_pair in "1000/500" "5000/2500" "10000/5000" "20000/5000"; do
  #for token_pair in "1000/500" "5000/2500" "10000/5000" "20000/5000"; do
  #for token_pair in "20000/5000"; do
  #for token_pair in "10000/5000"; do
  #for token_pair in "1000/500"; do
  #for token_pair in "10000/5000" "20000/5000"; do
    input_tokens="${token_pair%%/*}"
    output_tokens="${token_pair##*/}"

    RUN_ID="${CSV_PREFIX}_${tp}_${input_tokens}_${output_tokens}"
    if [ "$DP" -gt 1 ]; then
      RUN_ID+="_${DP}"
    fi
    CSV_ID=${RUN_ID}
    if [[ -n $VERSION_TAG ]]; then
      if [[ $VERSION_TAG == "lmcache" ]]; then
        CSV_ID+="_${VERSION_TAG}"
      else
        CSV_ID+="_v${VERSION_TAG}"
      fi
    fi

    rm -rf out

    first_params="--skip_venv_creation"
    if $first_flag; then
      echo "one-time install of benchmark dependencies!"
      first_params=""
      first_flag=false
    fi
    echo "${RUN_ID}"

    if [[ $LMCACHE == "y" || $LMCACHE == "Y" ]]; then
      ./test.sh -vg -m $MODEL --tp ${tp} --dp ${DP} -i ${input_tokens} -o ${output_tokens} ${first_params} --lmcache
    else
      ./test.sh -vg -m $MODEL --tp ${tp} --dp ${DP} -i ${input_tokens} -o ${output_tokens} ${first_params}
    fi
    cp output.csv test_results/${RUN_ID}.csv
    perl -pe "s/^.*?,/${CSV_ID},/" -i test_results/${RUN_ID}.csv
  done
done

source bench_venv/bin/activate

echo "concatenating results"

rm -f all_results.csv
python3 concat.py

echo "run complete!"
