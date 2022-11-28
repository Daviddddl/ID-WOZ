#!/bin/bash

env=${1}

ctpu_in_file="gs://bert-bahasa/shards/"
ctpu_out_file="gs://bert-bahasa/tmp/"
ctpu_vocab_file="gs://bert-bahasa/bahasa_vocab.txt"

local_in_file="./corpus/shards/"
local_out_file="./corpus/output/"
local_vocab_file="./models/bahasa_cased_L-24_H-1024_A-16/bahasa_vocab.txt"

if [[ "${env}" == 'ctpu' ]]
then
  echo "Running in Google Cloud TPU !!"
  in_file=${ctpu_in_file}
  out_file=${ctpu_out_file}
  vocab_file=${ctpu_vocab_file}
else
  echo "Running in local PC !!"
  in_file=${local_in_file}
  out_file=${local_out_file}
  vocab_file=${local_vocab_file}

  if [[ ! ${in_file} ]]; then
    mkdir -p ${in_file}
    echo "Shards are vacant! Please Upload corpus files!"
    exit 1
  fi

  if [[ ! -f ${vocab_file} ]]; then
    echo "Vocab file " ${vocab_file} " doesnt exist! "
    exit 1
  fi

  rm -rf ${local_out_file}
  mkdir ${local_out_file}
fi

# shards_path='corpus/shards'
# sum_shards=`ls ${shards_path} | wc -w`
sum_shards=2602

echo "creating pretraining data for ${sum_shards} shards files"

# shellcheck disable=SC2004
for((i=1;i<=${sum_shards};i++));
do
python create_pretraining_data.py \
  --input_file=${in_file}"overall_${i}.txt" \
  --output_file=${out_file}"tf_examples_${i}.tfrecord" \
  --vocab_file=${vocab_file} \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
done
