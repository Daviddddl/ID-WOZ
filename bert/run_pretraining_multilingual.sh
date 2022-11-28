#!/bin/bash

num_steps=${1}

python run_pretraining.py \
  --input_file=gs://bert-bahasa/tmp/tf_examples_*.tfrecord \
  --output_dir=gs://bert-bahasa/output_test \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=gs://bert-bahasa/multi_cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=gs://bert-bahasa/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=${num_steps} \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=True \
  --tpu_name=bert-bahasa
