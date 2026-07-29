# Super-Predicate-Guided SGG (SPG-SGG)

This repository contains **Super-Predicate-Guided Scene Graph Generation (SPG-SGG)**, built on the PENET framework. SPG-SGG uses the super-predicate information available in the mVG-150 dataset to guide fine-grained relation prediction and improve the semantic diversity of predicted relations.

## Dataset

Download the mVG-150 dataset from [Click here](https://kaggle.com/datasets/32b51bdeb1e69c36f0c502d4388486d1afa7bc2ab9e8e032a240fa1fe8d38bbd).

Place the dataset files under `datasets/mVG-150/` and update the dataset paths in `maskrcnn_benchmark/config/paths_catalog.py`.

## Model

The SPG-SGG model is included in this repository. Its relation predictor is in:

```text
maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py
```


## Setup

```bash
conda create -n penet-lite python=3.8
conda activate penet-lite
python setup.py build develop
```

See `INSTALL.md` for the required CUDA, PyTorch, cocoapi, and apex versions.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/relation_train_net.py \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  OUTPUT_DIR ./checkpoints/SPG-SGG_SGdet
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/relation_test_net.py \
  MODEL.WEIGHT ./checkpoints/SPG-SGG_SGdet/model_final.pth \
  OUTPUT_DIR ./checkpoints/SPG-SGG_SGdet \
  TEST.IMS_PER_BATCH 1 \
  TEST.ALLOW_LOAD_FROM_CACHE False
```

## Diversity-Aware Metrics

For each subject-object pair, `DA-R@K` measures how many of its valid ground-truth predicates are recovered by the top-K relation predictions. `DA-mR@K` first computes this recall for each predicate class and then averages across classes, reducing the influence of frequent predicates. Predicting more distinct, correct relations therefore improves the DA scores.

The relation log includes standard `R`, `mR`, `ng-R`, and the renamed diversity-aware metrics:

```text
SGG eval: DA-R @ 20/50/100 ...
SGG eval: DA-mR @ 20/50/100 ...
```
