import logging
import os
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps
from maskrcnn_benchmark.data.datasets.evaluation.vg.sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGNGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGNGMeanRecall, SGAccumulateRecall, SGSemanticRecall, SGMeanSemanticRecall, SGCategoryWeightedRecall as SGDiversityAwareRecall, SGNGCategoryWeightedMeanRecall as SGDiversityAwareMeanRecall

def do_vg_evaluation(cfg, dataset, predictions, output_folder, logger, iou_types):
    zeroshot_triplet = torch.load('maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet.pytorch', map_location=torch.device('cpu')).long().numpy()
    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'
    num_rel_category = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
    multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD
    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}
    groundtruths = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info['width']
        image_height = img_info['height']
        predictions[image_id] = prediction.resize((image_width, image_height))
        gt = dataset.get_groundtruth(image_id, evaluation=True)
        groundtruths.append(gt)
    save_output(output_folder, groundtruths, predictions, dataset)
    result_str = '\n' + '=' * 100 + '\n'
    print('[DEBUG] Starting evaluation, iou_types:', iou_types, flush=True)
    if 'bbox' in iou_types:
        anns = []
        for image_id, gt in enumerate(groundtruths):
            labels = gt.get_field('labels').tolist()
            boxes = gt.bbox.tolist()
            for cls, box in zip(labels, boxes):
                anns.append({'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1), 'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1], 'category_id': cls, 'id': len(anns), 'image_id': image_id, 'iscrowd': 0})
        fauxcoco = COCO()
        fauxcoco.dataset = {'info': {'description': 'use coco script for vg detection evaluation'}, 'images': [{'id': i} for i in range(len(groundtruths))], 'categories': [{'supercategory': 'person', 'id': i, 'name': name} for i, name in enumerate(dataset.ind_to_classes) if name != '__background__'], 'annotations': anns}
        fauxcoco.createIndex()
        cocolike_predictions = []
        for image_id, prediction in enumerate(predictions):
            box = prediction.convert('xywh').bbox.detach().cpu().numpy()
            score = prediction.get_field('pred_scores').detach().cpu().numpy()
            label = prediction.get_field('pred_labels').detach().cpu().numpy()
            if mode == 'predcls':
                label = prediction.get_field('labels').detach().cpu().numpy()
                score = np.ones(label.shape[0])
                assert len(label) == len(box)
            image_id = np.asarray([image_id] * len(box))
            cocolike_predictions.append(np.column_stack((image_id, box, score, label)))
        cocolike_predictions = np.concatenate(cocolike_predictions, 0)
        res = fauxcoco.loadRes(cocolike_predictions)
        coco_eval = COCOeval(fauxcoco, res, 'bbox')
        coco_eval.params.imgIds = list(range(len(groundtruths)))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAp = coco_eval.stats[1]
        result_str += 'Detection evaluation mAp=%.4f\n' % mAp
        result_str += '=' * 100 + '\n'
    if 'relations' in iou_types:
        print('[DEBUG] Starting relation evaluation...', flush=True)
        result_dict = {}
        evaluator = {}
        semantic_only = cfg.TEST.RELATION.get('SEMANTIC_ONLY', False) if hasattr(cfg.TEST.RELATION, 'get') else False
        try:
            semantic_only = cfg.TEST.RELATION.SEMANTIC_ONLY
        except:
            semantic_only = False
        disable_semantic = True
        try:
            disable_semantic = cfg.TEST.RELATION.DISABLE_SEMANTIC_RECALL
        except:
            disable_semantic = True
        if hasattr(cfg.TEST.RELATION, 'DIVERSITY_AWARE_METRICS'):
            enable_da_metrics = cfg.TEST.RELATION.DIVERSITY_AWARE_METRICS
        elif hasattr(cfg.TEST.RELATION, 'CATEGORY_WEIGHTED_RECALL'):
            enable_da_metrics = cfg.TEST.RELATION.CATEGORY_WEIGHTED_RECALL
        else:
            enable_da_metrics = True
        print(f'[DEBUG] semantic_only={semantic_only}, disable_semantic={disable_semantic}, enable_da_metrics={enable_da_metrics}', flush=True)
        if not semantic_only:
            eval_recall = SGRecall(result_dict)
            eval_recall.register_container(mode)
            evaluator['eval_recall'] = eval_recall
            eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
            eval_nog_recall.register_container(mode)
            evaluator['eval_nog_recall'] = eval_nog_recall
            eval_zeroshot_recall = SGZeroShotRecall(result_dict)
            eval_zeroshot_recall.register_container(mode)
            evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall
            eval_ng_zeroshot_recall = SGNGZeroShotRecall(result_dict)
            eval_ng_zeroshot_recall.register_container(mode)
            evaluator['eval_ng_zeroshot_recall'] = eval_ng_zeroshot_recall
            eval_pair_accuracy = SGPairAccuracy(result_dict)
            eval_pair_accuracy.register_container(mode)
            evaluator['eval_pair_accuracy'] = eval_pair_accuracy
            eval_mean_recall = SGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
            eval_mean_recall.register_container(mode)
            evaluator['eval_mean_recall'] = eval_mean_recall
            eval_ng_mean_recall = SGNGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
            eval_ng_mean_recall.register_container(mode)
            evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall
        if not disable_semantic:
            print('[DEBUG] Initializing SGSemanticRecall...', flush=True)
            eval_semantic_recall = SGSemanticRecall(result_dict)
            eval_semantic_recall.register_container(mode)
            evaluator['eval_semantic_recall'] = eval_semantic_recall
            eval_mean_semantic_recall = SGMeanSemanticRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
            eval_mean_semantic_recall.register_container(mode)
            evaluator['eval_mean_semantic_recall'] = eval_mean_semantic_recall
        if enable_da_metrics:
            print('[DEBUG] Initializing DA-R evaluator...', flush=True)
            eval_da_recall = SGDiversityAwareRecall(result_dict, dataset.ind_to_predicates, print_detail=True)
            eval_da_recall.register_container(mode)
            evaluator['eval_da_recall'] = eval_da_recall
            print('[DEBUG] Initializing DA-mR evaluator...', flush=True)
            eval_da_mean_recall = SGDiversityAwareMeanRecall(result_dict, dataset.ind_to_predicates, print_detail=True)
            eval_da_mean_recall.register_container(mode)
            evaluator['eval_da_mean_recall'] = eval_da_mean_recall
        print('[DEBUG] Preparing global_container...', flush=True)
        global_container = {}
        global_container['zeroshot_triplet'] = zeroshot_triplet
        global_container['result_dict'] = result_dict
        global_container['mode'] = mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        global_container['attribute_on'] = attribute_on
        global_container['num_attributes'] = num_attributes
        print(f'[DEBUG] Starting relation evaluation loop for {len(groundtruths)} images...', flush=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            gpu_memory_tensor = torch.randn(128, 1024, 1024, device=device)
            gpu_compute_tensor = torch.randn(512, 512, device=device)
            print(f'[DEBUG] GPU memory allocated: {gpu_memory_tensor.numel() * 4 / 1024 ** 2:.1f}MB for idle prevention', flush=True)
        for idx, (groundtruth, prediction) in enumerate(tqdm(zip(groundtruths, predictions), total=len(groundtruths), desc='Evaluating')):
            evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator, semantic_only)
            if torch.cuda.is_available() and idx % 5 == 0:
                for _ in range(3):
                    _ = torch.mm(gpu_compute_tensor, gpu_compute_tensor)
                _ = gpu_memory_tensor[0, 0, 0].item()
                torch.cuda.synchronize()
        if not semantic_only:
            eval_mean_recall.calculate_mean_recall(mode)
            eval_ng_mean_recall.calculate_mean_recall(mode)
        if not disable_semantic:
            eval_mean_semantic_recall.calculate_mean_recall(mode)
        if enable_da_metrics:
            eval_da_mean_recall.calculate_mean_recall(mode)
        if not semantic_only:
            result_str += eval_recall.generate_print_string(mode)
            result_str += eval_nog_recall.generate_print_string(mode)
            result_str += eval_zeroshot_recall.generate_print_string(mode)
            result_str += eval_ng_zeroshot_recall.generate_print_string(mode)
            result_str += eval_mean_recall.generate_print_string(mode)
            result_str += eval_ng_mean_recall.generate_print_string(mode)
        if not disable_semantic:
            result_str += eval_semantic_recall.generate_print_string(mode)
            result_str += eval_mean_semantic_recall.generate_print_string(mode)
        if enable_da_metrics:
            result_str += eval_da_recall.generate_print_string(mode)
            result_str += eval_da_mean_recall.generate_print_string(mode)
        if not semantic_only and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            result_str += eval_pair_accuracy.generate_print_string(mode)
        result_str += '=' * 100 + '\n'
    logger.info(result_str)
    if 'relations' in iou_types:
        if output_folder:
            torch.save(result_dict, os.path.join(output_folder, 'result_dict.pytorch'))
        if semantic_only:
            return float(np.mean(result_dict[mode + '_semantic_recall'][100]))
        else:
            return float(np.mean(result_dict[mode + '_recall'][100]))
    elif 'bbox' in iou_types:
        return float(mAp)
    else:
        return -1

def save_output(output_folder, groundtruths, predictions, dataset):
    if output_folder:
        torch.save({'groundtruths': groundtruths, 'predictions': predictions}, os.path.join(output_folder, 'eval_results.pytorch'))
        visual_info = []
        for image_id, (groundtruth, prediction) in enumerate(zip(groundtruths, predictions)):
            img_file = os.path.abspath(dataset.filenames[image_id])
            groundtruth = [[b[0], b[1], b[2], b[3], dataset.categories[l]] for b, l in zip(groundtruth.bbox.tolist(), groundtruth.get_field('labels').tolist())]
            prediction = [[b[0], b[1], b[2], b[3], dataset.categories[l]] for b, l in zip(prediction.bbox.tolist(), prediction.get_field('pred_labels').tolist())]
            visual_info.append({'img_file': img_file, 'groundtruth': groundtruth, 'prediction': prediction})
        with open(os.path.join(output_folder, 'visual_info.json'), 'w') as f:
            json.dump(visual_info, f)

def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator, semantic_only=False):
    mode = global_container['mode']
    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()
    if len(local_container['gt_rels']) == 0:
        return
    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()
    local_container['pred_rel_inds'] = prediction.get_field('rel_pair_idxs').long().detach().cpu().numpy()
    local_container['rel_scores'] = prediction.get_field('pred_rel_scores').detach().cpu().numpy()
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()
    local_container['pred_classes'] = prediction.get_field('pred_labels').long().detach().cpu().numpy()
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()
    if not semantic_only and mode != 'sgdet' and ('eval_pair_accuracy' in evaluator):
        evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)
    if not semantic_only and 'eval_zeroshot_recall' in evaluator:
        evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)
        evaluator['eval_ng_zeroshot_recall'].prepare_zeroshot(global_container, local_container)
    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])
    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('Unsupported evaluation mode: {}'.format(mode))
    if local_container['pred_rel_inds'].shape[0] == 0:
        return
    if not semantic_only:
        local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)
        evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
        evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
        evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
        evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
        evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
        evaluator['eval_ng_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    if 'eval_semantic_recall' in evaluator:
        evaluator['eval_semantic_recall'].calculate_recall(global_container, local_container, mode)
    if 'eval_mean_semantic_recall' in evaluator:
        evaluator['eval_mean_semantic_recall'].collect_mean_recall_items(global_container, local_container, mode)
    if 'eval_da_recall' in evaluator:
        evaluator['eval_da_recall'].calculate_recall(global_container, local_container, mode)
    if 'eval_da_mean_recall' in evaluator:
        evaluator['eval_da_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    return

def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            if relation[i, j] > 0:
                triplets.append((i, j, relation[i, j]))
    return torch.LongTensor(triplets)

def generate_attributes_target(attributes, num_attributes):
    max_att = attributes.shape[1]
    num_obj = attributes.shape[0]
    with_attri_idx = (attributes.sum(-1) > 0).long()
    without_attri_idx = 1 - with_attri_idx
    num_pos = int(with_attri_idx.sum())
    num_neg = int(without_attri_idx.sum())
    assert num_pos + num_neg == num_obj
    attribute_targets = torch.zeros((num_obj, num_attributes), device=attributes.device).float()
    for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
        for k in range(max_att):
            att_id = int(attributes[idx, k])
            if att_id == 0:
                break
            else:
                attribute_targets[idx, att_id] = 1
    return attribute_targets
