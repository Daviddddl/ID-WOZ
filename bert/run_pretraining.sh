#!/bin/bash

init_id=${1}
batch_size=${2}
num_steps=${3}
lr=${4}

python run_pretraining.py \
  --input_file=gs://bert-bahasa/tmp/tf_examples_*.tfrecord \
  --output_dir=gs://bert-bahasa/output_test \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=gs://bert-bahasa/bert_config.json \
  --init_checkpoint=gs://bert-bahasa/output_test/model.ckpt-${init_id} \
  --train_batch_size=${batch_size} \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=${num_steps} \
  --num_warmup_steps=10 \
  --learning_rate=${lr} \
  --use_tpu=True \
  --tpu_name=bert-bahasa
