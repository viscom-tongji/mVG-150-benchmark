#!/bin/bash
# ============================================
#  Script: penet_expert_lite.sh
#  Purpose: Train PrototypeEmbeddingNetwork_Experts_lite in SGDET mode
# ============================================

unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=7
export NUM_GPU=1

echo "==============================="
echo "TRAINING MODE: sgdet"
echo "GPU(s) Used: $CUDA_VISIBLE_DEVICES"
echo "==============================="

MODEL_NAME='PE-NET_EXPERT_SGdet'
CHECKPOINT_DIR=./checkpoints/${MODEL_NAME}

echo "Creating checkpoint directory: ${CHECKPOINT_DIR}"
mkdir -p ${CHECKPOINT_DIR}

echo "Backing up source files..."
cp ./tools/relation_train_net.py ${CHECKPOINT_DIR}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ${CHECKPOINT_DIR}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/hierarchy_utils.py ${CHECKPOINT_DIR}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/relation_head.py ${CHECKPOINT_DIR}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py ${CHECKPOINT_DIR}/
cp ./maskrcnn_benchmark/config/defaults.py ${CHECKPOINT_DIR}/
cp ./configs/penet_expert_lite.yaml ${CHECKPOINT_DIR}/
cp ./penet_expert_lite.sh ${CHECKPOINT_DIR}/

echo "Starting training..."

python3 tools/relation_train_net.py \
  --config-file "expert_config.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  OUTPUT_DIR ${CHECKPOINT_DIR}

echo "Training finished. Check results in: ${CHECKPOINT_DIR}"
