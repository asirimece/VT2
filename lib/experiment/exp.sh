# chmod +x lib/experiment/exp.sh

set -euo pipefail

# 1) MTL pretraining
MTL_CONFIG="config/experiment/mtl.yaml"
FEATURES_FILE="dump/features.pkl"
MTL_OUT_DIR="models"

# 2) Transfer‑Learning
PREPROC_DATA="dump/preprocessed_data.pkl"
TL_OUT_DIR="tl_results"

# Subjects to test (e.g. all BCI subjects)
SUBJECTS=(1 2 3 4 5 6 7 8 9)

# Number of clusters you used
NUM_CLUSTERS=4

# 3) Summarization
SUMMARIZE_SCRIPT="lib/experiment/summarize.py"
EVAL_OUT_DIR="evaluation"
##################################################################

echo "=== 1) Pretrain global MTL backbone ==="
python train_mtl.py \
  --config-path "${MTL_CONFIG}" \
  --features-file "${FEATURES_FILE}" \
  --out_dir "${MTL_OUT_DIR}/mtl/all"

echo "=== 2) Pretrain per‐cluster backbones (0..$((NUM_CLUSTERS-1))) ==="
for C in $(seq 0 $((NUM_CLUSTERS-1))); do
  echo "--- Cluster $C ---"
  python train_mtl.py \
    --config-path "${MTL_CONFIG}" \
    --features-file "${FEATURES_FILE}" \
    --restrict_to_cluster \
    --cluster_id "${C}" \
    --out_dir "${MTL_OUT_DIR}/mtl/cluster${C}"
done

echo "=== 3) Transfer‑Learning for each subject ==="
for SUBJ in "${SUBJECTS[@]}"; do
  echo "--- Subject $SUBJ: scratch baseline ---"
  python run_tl.py \
    pretrained_mtl_model=null \
    init_from_scratch=true \
    preprocessed_data="${PREPROC_DATA}" \
    subject="${SUBJ}" \
    out_dir="${TL_OUT_DIR}/scratch/${SUBJ}"

  echo "--- Subject $SUBJ: global TL ---"
  python run_tl.py \
    pretrained_mtl_model="${MTL_OUT_DIR}/mtl_all/mtl_model_weights.pth" \
    init_from_scratch=false \
    preprocessed_data="${PREPROC_DATA}" \
    subject="${SUBJ}" \
    out_dir="${TL_OUT_DIR}/global/${SUBJ}"

  # Determine this subject’s cluster ID by re‑running small Python snippet:
  CID=$(python - <<EOF
import pickle
from lib.cluster.cluster import SubjectClusterer
import yaml
cfg = yaml.safe_load(open("${MTL_CONFIG}"))["experiment"]["clustering"]
clusterer = SubjectClusterer("${FEATURES_FILE}", cfg)
wrapper = clusterer.cluster_subjects(method=cfg.get("method","kmeans"))
print(wrapper.get_cluster_for_subject(str(${SUBJ})))
EOF
)
  echo "--- Subject $SUBJ: cluster‐TL (cluster $CID) ---"
  python run_tl.py \
    pretrained_mtl_model="${MTL_OUT_DIR}/mtl_cluster${CID}/mtl_model_weights.pth" \
    init_from_scratch=false \
    preprocessed_data="${PREPROC_DATA}" \
    subject="${SUBJ}" \
    out_dir="${TL_OUT_DIR}/cluster/${SUBJ}"
done

echo "=== 4) Summarize & visualize all results ==="
python "${SUMMARIZE_SCRIPT}"

echo "All experiments done. Results are in:"
echo "  - Per‑run TL outputs: ${TL_OUT_DIR}/"
echo "  - Summary CSV + plots: ${EVAL_OUT_DIR}/"
#!/usr/bin/env bash
set -euo pipefail

# Config & paths
CONFIG="config/experiment/mtl.yaml"
MODEL_CFG="config/model/deep4net.yaml"
FEATURES="dump/features.pkl"
PREPROC="dump/preprocessed_data.pkl"

# Where to dump MTL weights
MTL_GLOBAL_OUT="models/mtl_all"
MTL_CLUSTER_OUT_PREFIX="models/mtl_cluster"

# Where to dump TL results
TL_ROOT="tl_results"

# Subject and cluster lists
SUBJECTS=(1 2 3 4 5 6 7 8 9)
CLUSTERS=(0 1 2 3)

echo "=== 1) GLOBAL MTL ==="
python run_mtl.py \
  -c "$CONFIG" \
  -m "$MODEL_CFG" \
  -o "$MTL_GLOBAL_OUT"

echo "=== 2) CLUSTER‑SPECIFIC MTL ==="
for C in "${CLUSTERS[@]}"; do
  echo "→ cluster $C"
  python run_mtl.py \
    -c "$CONFIG" \
    -m "$MODEL_CFG" \
    -f "$FEATURES" \
    -r -k "$C" \
    -o "${MTL_CLUSTER_OUT_PREFIX}${C}"
done

echo "=== 3) SCRATCH TL ==="
for S in "${SUBJECTS[@]}"; do
  echo "→ subject $S"
  python run_tl.py \
    -c "$CONFIG" \
    --preprocessed-data "$PREPROC" \
    --subject "$S" \
    --init-from-scratch \
    --freeze-backbone \
    -o "${TL_ROOT}/scratch/${S}"
done

echo "=== 4) GLOBAL‑BACKBONE TL ==="
for S in "${SUBJECTS[@]}"; do
  echo "→ subject $S"
  python run_tl.py \
    -c "$CONFIG" \
    --preprocessed-data "$PREPROC" \
    --subject "$S" \
    --pretrained-mtl-model "${MTL_GLOBAL_OUT}/mtl_weights_all.pth" \
    --freeze-backbone \
    -o "${TL_ROOT}/global/${S}"
done

echo "=== 5) CLUSTER‑BACKBONE TL ==="
for C in "${CLUSTERS[@]}"; do
  for S in "${SUBJECTS[@]}"; do
    echo "→ subject $S on cluster $C backbone"
    python run_tl.py \
      -c "$CONFIG" \
      --preprocessed-data "$PREPROC" \
      --subject "$S" \
      --pretrained-mtl-model "${MTL_CLUSTER_OUT_PREFIX}${C}/mtl_weights_cluster${C}.pth" \
      --freeze-backbone \
      -o "${TL_ROOT}/cluster/${S}"
  done
done

echo "=== 6) FINAL EVALUATION ==="
python summarize_tl.py

echo "=== EXPERIMENT COMPLETE ==="
