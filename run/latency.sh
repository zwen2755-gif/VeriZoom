#!/bin/bash
# Latency of every method on the same questions (every 2nd of V* / HR-Bench, every 19th of MME-RealWorld), one process
# at a time on an otherwise idle GPU. VeriZoom runs only its selected rule and read form; the first question is a
# warm-up and is not timed.
#   bash run/latency.sh 7b   models/Qwen2.5-VL-7B-Instruct "19:22 19:15 19:17" warp 28
#   bash run/latency.sh q3vl models/Qwen3-VL-8B-Instruct  "21:10 21:11 18:21" warp 32
#   bash run/latency.sh q35  models/Qwen3.5-9B            "15:9 11:14 19:11"  solo 32
TAG=$1; M=$2; HEADS=$3; FORM=$4; export FOVEA_PATCH=$5
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
F=runs/fovea/forecasters
for d in vstar hrbench4k hrbench8k mmerw; do
  E=2; [ $d = mmerw ] && E=19
  FC=$F/g1024v2_not_$d.pt; [ $d = mmerw ] && FC=$F/g1024v2_all.pt
  python scripts/fovea_cascade.py --dataset $d --every $E --warm --rules both --refine 0 --forms $FORM --model $M --heads $HEADS --tag _lat_$TAG
  python scripts/baseline_native.py --dataset $d --every $E --warm --model $M --tokens 1600 8192 --tag _lat_$TAG
  python scripts/fovea_e2e.py --dataset $d --every $E --model $M --forecaster $FC --tag _lat_$TAG --no-reference --no-downscale --k 1 --verify
done
