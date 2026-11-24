#!/bin/bash
# Models, benchmarks and the MME-RealWorld training split used for all selections.
set -e
cd "$(dirname "$0")/.."
for m in Qwen2.5-VL-3B-Instruct Qwen2.5-VL-7B-Instruct Qwen3-VL-8B-Instruct Qwen3.5-9B; do
  huggingface-cli download Qwen/$m --local-dir models/$m
done
huggingface-cli download craigwu/vstar_bench --repo-type dataset --local-dir data_local/raw/vstar
huggingface-cli download DreamMr/HR-Bench --repo-type dataset --local-dir data_local/raw/hrbench
huggingface-cli download yifanzhang114/MME-RealWorld-Lite --repo-type dataset --local-dir data_local/raw/mmerw_lite
# training split: questions and images disjoint from MME-RealWorld-Lite, extracted group by group
R=data_local/raw/mmerw_full
mkdir -p $R && cp data_splits/mmerw_train_split.json $R/train_split.json
for g in remote_sensing ocr_cc AutonomousDriving monitoring_images diagram_and_table; do
  huggingface-cli download yifanzhang114/MME-RealWorld --repo-type dataset --include "$g.tar.gz*" --local-dir $R
  python scripts/extract_mmerw.py $g && rm -f $R/$g.tar.gz*
done
python scripts/restore_labels.py          # tile labels of the training split, shared by all models for head ranking
