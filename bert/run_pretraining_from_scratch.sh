#!/bin/bash

num_steps=${1}
batch_size=${2}

echo 'Run pre-training from scratch: num_steps='${num_steps}' and batch_size='${batch_size}

python run_pretraining.py \
  --input_file=gs://bert-bahasa/tmp/tf_examples_*.tfrecord \
  --output_dir=gs://bert-bahasa/output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=gs://bert-bahasa/bert_config.json \
  --train_batch_size=${batch_size} \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=${num_steps} \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=True \
  --tpu_name=bert-bahasa
