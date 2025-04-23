
set -euo pipefail


EXP_MODELS="exp_models"
mkdir -p "${EXP_MODELS}/mtl"
mkdir -p "${EXP_MODELS}/tl"

# 1) Pretrain MTL on all subjects (global MTL)
echo "=== 1) Global MTL ==="
python lib/experiment/run_mtl.py \
  --config        config/experiment/transfer.yaml \
  --model-config  config/model/deep4net.yaml \
  --out-dir       "${EXP_MODELS}/mtl/global"

# 2) Pretrain MTL on each cluster
echo "=== 2) Cluster MTL ==="
for c in 0 1 2 3; do
  echo "--- Cluster $c ---"
  python lib/experiment/run_mtl.py \
    --config        config/experiment/transfer.yaml \
    --model-config  config/model/deep4net.yaml \
    --out-dir       "${EXP_MODELS}/mtl/cluster${c}" \
    --restrict-to-cluster \
    --cluster-id    "${c}"
done

# 3) Scratch TL (no MTL backbone)
echo "=== 3) Scratch TL ==="
python lib/experiment/run_tl.py \
  --config             config/experiment/transfer.yaml \
  --preprocessed-data  dump/preprocessed_data.pkl \
  --out-dir            "${EXP_MODELS}/tl/scratch" \
  --init-from-scratch

# 4) Global TL (use global MTL backbone)
echo "=== 4) Global TL ==="
python lib/experiment/run_tl.py \
  --config             config/experiment/transfer.yaml \
  --preprocessed-data  dump/preprocessed_data.pkl \
  --pretrained-mtl-model "${EXP_MODELS}/mtl/global/mtl_weights_all.pth" \
  --out-dir            "${EXP_MODELS}/tl/global" \
  --freeze-backbone

# 5) Cluster‑based TL (matched: only subjects in each cluster)
echo "=== 5) Cluster TL (matched) ==="
for c in 0 1 2 3; do
  echo "--- Cluster $c TL ---"
  python lib/experiment/run_tl.py \
    --config             config/experiment/transfer.yaml \
    --preprocessed-data  dump/preprocessed_data.pkl \
    --pretrained-mtl-model "${EXP_MODELS}/mtl/cluster${c}/mtl_weights_cluster${c}.pth" \
    --out-dir            "${EXP_MODELS}/tl/cluster${c}" \
    --freeze-backbone
done

echo "=== 6) Summarize results ==="
python lib/experiment/summarize.py

echo "Experiments complete. Outputs under ${EXP_MODELS} and evaluation/."
