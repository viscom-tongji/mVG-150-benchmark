# PENET Expert Lite for Scene Graph Generation

This repository is a compact research version of PENET for Scene Graph Generation. The relation predictor file keeps the original `PrototypeEmbeddingNetwork` and the standalone expert-lite model `PrototypeEmbeddingNetwork_Experts_lite`.

The expert-lite model adds coarse predicate groups and lightweight owner-specific prototype experts on top of PENET. Evaluation reports the diversity-aware metrics `DA-R` and `DA-mR` in addition to the standard SGG metrics.

## Environment

This code follows the original `maskrcnn-benchmark` / `Scene-Graph-Benchmark.pytorch` environment. A typical setup is:

```bash
conda create -n penet-lite python=3.8
conda activate penet-expert
python setup.py build develop
```

For detailed CUDA, PyTorch, cocoapi, and apex notes, see `INSTALL.md`. Older PyTorch/CUDA versions are usually required by this codebase.

## Data And Paths

Prepare the Visual Genome SGG data following `DATASET.md`. Before training, edit `configs/expert_config.yaml` and make sure these paths match your machine:

```yaml
GLOVE_DIR: /path/to/glove/
MODEL:
  PRETRAINED_DETECTOR_CKPT: /path/to/pretrained_faster_rcnn/model_final.pth
PATHS_CATALOG: /path/to/maskrcnn_benchmark/config/paths_catalog.py
OUTPUT_DIR: ./checkpoints/PE-NET_EXPERTS_LITE_SGdet
```

The active relation predictor is:

```yaml
MODEL:
  ROI_RELATION_HEAD:
    PREDICTOR: PrototypeEmbeddingNetwork_Experts_lite
    EXPERTS_LITE:
      ENABLED: true
    HIERARCHY:
      ENABLED: true
```

## Training

Run SGDet training with the expert-lite configuration:

```bash
export CUDA_VISIBLE_DEVICES=0
export NUM_GPU=1

python3 tools/relation_train_net.py \
  --config-file configs/expert_config.yaml \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  OUTPUT_DIR ./checkpoints/PE-NET_EXPERTS_LITE_SGdet
```

You can also adjust common training options from the command line:

```bash
SOLVER.IMS_PER_BATCH 8 \
SOLVER.MAX_ITER 50000 \
SOLVER.BASE_LR 0.001 \
SOLVER.VAL_PERIOD 5000 \
SOLVER.CHECKPOINT_PERIOD 5000
```

If you use `penet_expert_lite.sh`, keep its config path aligned with this repo version:

```bash
--config-file configs/expert_config.yaml
```

## Evaluation

Evaluate a trained checkpoint with:

```bash
python3 tools/relation_test_net.py \
  --config-file configs/expert_config.yaml \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.WEIGHT ./checkpoints/PE-NET_EXPERTS_LITE_SGdet/model_final.pth \
  OUTPUT_DIR ./checkpoints/PE-NET_EXPERTS_LITE_SGdet \
  TEST.IMS_PER_BATCH 1 \
  TEST.ALLOW_LOAD_FROM_CACHE False
```

The relation log includes standard `R`, `mR`, `ng-R`, and the renamed diversity-aware metrics:

```text
SGG eval: DA-R @ 20/50/100 ...
SGG eval: DA-mR @ 20/50/100 ...
```

## Important Files

- `maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py`: PENET and `PrototypeEmbeddingNetwork_Experts_lite`.
- `maskrcnn_benchmark/modeling/roi_heads/relation_head/hierarchy_utils.py`: predicate hierarchy and expert metadata.
- `configs/expert_config.yaml`: main config for expert-lite SGDet training.
- `maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py`: SGG metrics, including `DA-R` and `DA-mR`.

## Notes

This repository is intended for research reproduction and GitHub demonstration. Dataset files, pretrained detector checkpoints, and GloVe vectors are not included; configure their local paths before running training or evaluation.
