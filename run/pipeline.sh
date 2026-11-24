#!/bin/bash
# Full protocol for one model: localization heads ranked on the training split, exits / weighting / read form
# selected on the training split, then the four benchmarks and the baselines.
#   bash run/pipeline.sh h  models/Qwen2.5-VL-3B-Instruct 28
#   bash run/pipeline.sh 7b models/Qwen2.5-VL-7B-Instruct 28 "--offload embed head"   # 16 GB GPUs
#   bash run/pipeline.sh q3vl models/Qwen3-VL-8B-Instruct 32
#   bash run/pipeline.sh q35 models/Qwen3.5-9B 32                                       # transformers>=5
TAG=$1; M=$2; export FOVEA_PATCH=$3; OFF=$4
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
A="--rules both uniform --refine 0 --forms sep warp solo"
F=runs/fovea/forecasters
L=logs/$TAG; mkdir -p $L
python scripts/check_engine_vs_hf.py --model $M > $L/check_engine.log 2>&1
python scripts/fovea_head_sweep.py --dataset mmerwtrain --model $M $OFF --out head_sweep_$TAG.pt > $L/head_sweep.log 2>&1
H=$(python scripts/fovea_rank_heads.py --sweep head_sweep_$TAG.pt 2>>$L/rank_heads.log)
echo "heads: $H"
python scripts/fovea_cascade.py --dataset mmerwtrain --every 4 $A --model $M $OFF --heads $H --tag _sel_$TAG > $L/sel.log 2>&1
python scripts/fovea_cascade_an.py select runs/fovea/mmerwtrain/cascade_sel_$TAG.jsonl | tee $L/select.log
for d in vstar hrbench4k hrbench4k_c123 hrbench8k hrbench8k_c123 mmerw; do
  python scripts/fovea_cascade.py --dataset $d $A --model $M $OFF --heads $H --tag _test_$TAG > $L/test_$d.log 2>&1
  python scripts/baseline_native.py --dataset $d --model $M $OFF --tokens 1600 8192 --tag _$TAG > $L/native_$d.log 2>&1
done
for d in vstar hrbench4k hrbench8k; do
  for s in $d ${d}_c123; do
    [ "$s" = "vstar_c123" ] && continue
    python scripts/fovea_e2e.py --dataset $s --model $M --forecaster $F/g1024v2_not_$d.pt --tag _$TAG --no-reference --no-downscale --k 1 --verify > $L/vicrop_$s.log 2>&1
  done
done
python scripts/fovea_e2e.py --dataset mmerw --model $M --forecaster $F/g1024v2_all.pt --tag _$TAG --no-reference --no-downscale --k 1 --verify > $L/vicrop_mmerw.log 2>&1
